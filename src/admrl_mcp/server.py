"""Admiral fleet MCP server.

Talks to api.admrl.co with a Personal API Token. Tools are aimed at
natural fleet questions (find a tagged device, inspect IPs, pull logs from
offline/crashing nodes and summarise them) rather than a 1:1 swagger dump.

Live log tail over WebSocket (/v1/ws/devices/{id}/logs/stream) authenticates
with a Firebase/JWT session, not a PAT, so this server uses the historical
VictoriaLogs query endpoints instead. Those work for offline devices.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
import sys
import contextvars
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from .client import AdmiralAPIError, AdmiralClient, log_page_cursor, parse_rfc3339
from .config import ConfigError, require_settings
from . import docs
from . import fleetmetrics
from . import memtest
from . import ops
from . import troubleshoot as ts
from .extensions import load_env_extensions
from .distill import distill_events, distill_logs, health_from_stats
from .resolve import DeviceResolver, collect_ips, is_uuid, summarise_device
from .watch import RolloutWatch, StateWatch, summarise_state

INSTRUCTIONS = """
You are talking to an Admiral edge fleet (physical devices running Admiral OS).

How to answer naturally:
- Resolve devices by name, tag, notes, IP, hardware, or UUID via find_device / list_devices.
  Tags live on the device and/or its fleet (effective_tags). A printer is often tagged role=printer or similar.
- Offline and crashing devices still have historical logs in VictoriaLogs. Use get_device_logs or diagnose_device.
  Do not try to open the live websocket tail; PAT cannot authenticate it.
- For "what's the IP of X", use get_device_network after resolving the device.
- For "what's on the screen of X", use get_device_screenshot (live capture; the device must be online).
- Observed state: get_device_state (stored; live=true asks the device), watch_device_state (SSE timeline of
  condition transitions and progress operations). Desired state: get_device_document / render_device_document
  (dry run) / patch_device_document. diagnose_device uses the backend verdict (plus the check-in/restart schedule)
  and falls back to local distillation. Device policy (health/connectivity/telemetry/maintenance limits, layered
  org < fleet < device): get_device_policy; change it with patch_device_document on spec.policy.
- duplicate_fleet / duplicate_configuration create a copy (billed features and secret files are never copied;
  the result lists them under skipped). Only on an explicit request.
- Rollouts: preview_rollout (read-only plan: impact, spec diff, warnings, the exact create_rollout call), create_rollout
  (config, reboot, restart_workload, system_update over one or more fleets), list_rollouts, get_rollout,
  list_rollout_devices, rollout_control, watch_rollout (SSE until terminal).
- Changing what a fleet runs: list_configurations / get_configuration / diff_configuration_versions to read;
  edit_configuration (dry_run first; saves a NEW version; base_version guards against concurrent edits),
  rollback_configuration, update_configuration_metadata, create_configuration, delete_configuration. get_fleet shows
  which configuration a fleet runs ('latest' or pinned). assign_fleet_configuration applies a configuration/version to
  a fleet DIRECTLY (no canary, nothing pushed to connected devices; they pick it up when they reconnect or get a push).
  A fleet that follows 'latest' resolves a new version as soon as it exists. For running fleets the safe path is
  edit_configuration -> preview_rollout -> create_rollout, then follow it with get_rollout. Fleet/device management: create_fleet,
  update_fleet, set_fleet_update_policy (OS update policy/window), update_device, move_device_to_fleet,
  get_device_configuration, set_device_configuration_override, clear_device_configuration_override.
- For how-to / what-is questions about Admiral itself (concepts, setup, provisioning, configuration), use
  search_docs then read_docs_page, answer from them, and always cite the docs URL.
- "What is wrong with my device?" -> troubleshoot_device (one call: state, backend diagnosis, fresh probe, logs, events,
  workload, storage, time, connectivity, desired-vs-observed drift -> ranked findings with concrete steps and a docs
  link). Narrower: check_device_connectivity (DNS/TCP/NTP/NATS/clock/link), explain_workload_failure (crash loops,
  exit codes, pull/signature/USB denials, OOM), fleet_health_report (problems grouped across a fleet). All read-only.
- Fleet / organisation metrics: get_fleet_metrics (avg, max, latest, device count; per_device=true ranks devices; pass
  device to compare one device with its fleet), get_fleet_health, get_fleet_uptime, get_org_metrics (org-wide, by
  fleet or device, top N), query_telemetry_metrics (raw PromQL), get_telemetry_scope. A 402 means the organisation
  lacks the Telemetry add-on (billing gate, not an outage).
- Memory tests: start_memory_test (mode live keeps the workload running; full_online stops it and needs
  confirm=true after the user agrees), cancel_memory_test, get_memory_test, list_memory_test_results.
  A test boot is not available here: the user starts it from the dashboard or the device console.
- Destructive actions (reboot, document patches, local-override adopt/discard, configuration edits, rollbacks and
  deletes, fleet assignment, moving a device, rollouts and rollout control) require an explicit user request.
  Confirm the exact device/fleet/configuration first; ambiguous names return candidates and change nothing. Every
  mutating tool reads the state back from the API and reports it: trust that, not the 200.

Auth is a Personal API Token (X-API-Token-ID + X-API-Secret-Key). Organisation
context is X-Organization-ID; list_organisations if ADMRL_ORG_ID is unset.
""".strip()

_READ_ONLY = ToolAnnotations(readOnlyHint=True)

mcp = FastMCP(
    name="admrl",
    instructions=INSTRUCTIONS,
)

_client: AdmiralClient | None = None


# Hosted (multi-tenant) mode: ``hosted.py`` sets this True. A request-scoped client (carrying only
# that caller's token) then replaces the module-level singleton, which is never used.
_hosted_mode = False
_request_client: contextvars.ContextVar[Any] = contextvars.ContextVar("admrl_request_client", default=None)


def get_client() -> AdmiralClient:
    global _client
    holder = _request_client.get()
    if holder is not None:
        return holder.get()
    if _hosted_mode:
        raise ConfigError("No request credentials: hosted mode never falls back to a shared client.")
    if _client is None:
        _client = AdmiralClient(require_settings())
    return _client


def _parallel_map(fn: Any, items: list[Any], max_workers: int = 8) -> list[Any]:
    """Map fn over items, in parallel where threads exist.

    Pyodide (sys.platform == "emscripten") cannot start threads, so the browser
    build runs the same work sequentially. Order is preserved either way.
    """
    if sys.platform == "emscripten" or max_workers <= 1:
        return [fn(item) for item in items]
    # Worker threads do not inherit contextvars; run each task in a copy of the caller's context so
    # the request-scoped client (hosted mode) is visible to them.
    # (A context can only be entered by one thread at a time, so each task gets its own copy,
    # taken here in the calling thread.)
    contexts = [contextvars.copy_context() for _ in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda pair: pair[0].run(fn, pair[1]), zip(contexts, items)))


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=str, indent=2)


def _err(exc: Exception) -> str:
    if isinstance(exc, NeedsChoice):
        return _dumps(
            {
                "error": str(exc),
                "candidates": exc.candidates,
                "next": "Call again with the UUID of the intended " + exc.kind + "; nothing was changed.",
            }
        )
    if isinstance(exc, ToolRefusal):
        return _dumps(exc.payload)
    if isinstance(exc, AdmiralAPIError):
        out: dict[str, Any] = {"error": str(exc)}
        if exc.status == 402:
            out["billing_gate"] = True
            out["hint"] = (
                "The organisation lacks a feature this request needs (a billing/entitlement gate, not a fault). "
                "Enable it in the dashboard, or remove the setting that needs it."
            )
        elif exc.status == 409:
            out["conflict"] = True
            out["hint"] = "The resource changed or already exists. Re-read it and retry."
        return _dumps(out)
    if isinstance(exc, ConfigError):
        return _dumps({"error": str(exc)})
    return _dumps({"error": f"{type(exc).__name__}: {exc}"})


def _rfc3339_hours_ago(hours: float) -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    return start.replace(microsecond=0).isoformat().replace("+00:00", "Z"), end.replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _log_window(lookback_hours: float, before: str | None = None) -> tuple[str, str]:
    """Query window for log pages. `before` is a next_before cursor from a previous page."""
    if not before:
        return _rfc3339_hours_ago(lookback_hours)
    end = parse_rfc3339(before)
    start = end - timedelta(hours=lookback_hours)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _image_format(raw: bytes, reported: str | None = None) -> str:
    """Actual encoding of screenshot bytes.

    Magic bytes win over the reported `format` field — edges have answered
    jpeg for png requests.
    """
    if raw.startswith(b"\x89PNG"):
        return "png"
    if raw.startswith(b"\xff\xd8"):
        return "jpeg"
    rep = (reported or "").lower()
    return "jpeg" if rep in ("jpg", "jpeg") else "png"


def _resolve_device(query: str, organization_id: str | None = None, fleet_id: str | None = None) -> dict[str, Any]:
    client = get_client()
    resolver = DeviceResolver(client)
    org_id = resolver.resolve_org(organization_id)
    found = resolver.find(query, org_id=org_id, fleet_id=fleet_id)
    found["organization_id"] = org_id
    return found


_DOCS_ANNOTATIONS = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


@mcp.tool(
    description=(
        "Search the public Admiral documentation (docs.admrl.co). Use this to answer how-to and what-is questions "
        "about Admiral (concepts, provisioning, configuration, workloads, the CLI/API) and to point the user at the "
        "right docs page. Returns the top matches as {title, section, url, breadcrumbs, snippet}; always cite the "
        "docs url in your answer. Call read_docs_page on a hit for the full text. Needs no Admiral credentials and "
        "never sends them."
    ),
    annotations=_DOCS_ANNOTATIONS,
)
def search_docs(query: str, limit: int = 5) -> str:
    try:
        hits = docs.search(query, limit=limit)
        result: dict[str, Any] = {"query": query, "results": hits}
        if not hits:
            result["note"] = "No matching docs. Try different or fewer keywords."
        return _dumps(result)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Read one page of the public Admiral documentation (docs.admrl.co) by URL or path (for example a url from "
        "search_docs, or /advanced/ble). Returns the title, breadcrumbs, description and the page's sections in "
        "order, each with its anchor URL and text (truncated to about 12k characters). Use it to answer how-to / "
        "what-is questions about Admiral and cite the docs URL."
    ),
    annotations=_DOCS_ANNOTATIONS,
)
def read_docs_page(url_or_path: str) -> str:
    try:
        return _dumps(docs.read_page(url_or_path))
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="List organisations visible to this PAT. Use when ADMRL_ORG_ID is unset or the user asks which org to use.", annotations=_READ_ONLY)
def list_organisations() -> str:
    try:
        orgs = get_client().list_organisations()
        return _dumps(
            {
                "organisations": [
                    {"id": o.get("id"), "name": o.get("name"), "role": o.get("role")} for o in orgs if isinstance(o, dict)
                ],
                "default": get_client().settings.org_id,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="List fleets in the organisation. Optional search and tag filter (e.g. environment=production).", annotations=_READ_ONLY)
def list_fleets(
    search: str | None = None,
    tag: str | None = None,
    organization_id: str | None = None,
    limit: int = 50,
) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        payload = client.list_fleets(org_id=org_id, search=search, tag=tag, limit=limit)
        return _dumps({"organization_id": org_id, "fleets": payload})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "List devices. Filter by status (online, offline, error, provisioning, claimed, pending, unknown), "
        "fleet_id, or a free-text query (name, notes, tags). Returns compact summaries plus online/offline counts. "
        "status=offline returns every device that is not currently online — claimed, erroring, and stale devices "
        "keep their provisioning status while the platform counts them in counts.offline."
    ),
    annotations=_READ_ONLY,
)
def list_devices(
    query: str | None = None,
    status: str | None = None,
    fleet_id: str | None = None,
    organization_id: str | None = None,
    limit: int = 50,
) -> str:
    try:
        client = get_client()
        resolver = DeviceResolver(client)
        org_id = resolver.resolve_org(organization_id)
        if status == "offline":
            devices = resolver.list_not_online(org_id=org_id, fleet_id=fleet_id, limit=limit)
            payload = {"counts": None}
        else:
            devices = resolver.list_devices(
                org_id=org_id, query=query, status=status, fleet_id=fleet_id, limit=limit
            )
            payload = client.list_devices(
                org_id=org_id, search=None, status=status, fleet_id=fleet_id, limit=limit
            )
        counts = payload.get("counts") if isinstance(payload, dict) else None
        return _dumps(
            {
                "organization_id": org_id,
                "counts": counts,
                "devices": [summarise_device(d) for d in devices],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Resolve a device from a natural query: name, UUID, tag (role=printer), notes, IP, or hardware. "
        "Returns a single match or ranked candidates if ambiguous. Call this before logs/IP/diagnose."
    ),
    annotations=_READ_ONLY,
)
def find_device(
    query: str,
    organization_id: str | None = None,
    fleet_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(query, organization_id, fleet_id)
        out = {k: v for k, v in found.items() if k != "raw"}
        return _dumps(out)
    except Exception as inf:
        return _err(inf)


@mcp.tool(description="Full device detail: status, tags (device/fleet/effective), notes, last seen, hardware, location.", annotations=_READ_ONLY)
def get_device(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client = get_client()
        detail = client.get_device(found["match"]["id"], org_id=found["organization_id"])
        return _dumps({"device": summarise_device(detail if isinstance(detail, dict) else found["raw"] or {}), "detail": detail})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "IP addresses and interfaces for a device (name, tag, or UUID). "
        "Read from the system spec attached to the device object (GET /devices/{id} → systemSpec.network), "
        "the canonical IP source; the list ipAddress is a fallback. No live device probe."
    ),
    annotations=_READ_ONLY,
)
def get_device_network(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client = get_client()
        org_id = found["organization_id"]
        device_id = found["match"]["id"]
        detail = client.get_device(device_id, org_id=org_id)
        detail = detail if isinstance(detail, dict) else found.get("raw") or {}
        ips = collect_ips(detail)
        primary = next((row["ip"] for row in ips if row["version"] == "v4"), ips[0]["ip"] if ips else None)
        return _dumps(
            {
                "device": found["match"],
                "ips": ips,
                "primary_ip": primary,
                "source": "system_spec (attached to device object)",
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="Hardware / OS specification: product, CPUs, disks, network NICs, versions, boot slot.", annotations=_READ_ONLY)
def get_device_specs(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        specs = get_client().get_device_specs(found["match"]["id"], org_id=found["organization_id"])
        return _dumps({"device": found["match"], "specifications": specs})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "What the device's init system (s6) is running and whether each service is stable or crashing. "
        "Each service has state (up/down), health (stable, starting, unstable, flapping = crash loop, "
        "restarting, stopped, unsupervised, unavailable = hardware not present on this board, not a fault), pid, uptime, recent crash counts, and its last exits "
        "(exit code or signal) with a plain-English reason. Use this to see what is going wrong on a "
        "device (a crash-looping dhcpcd, workload or wpa_supplicant). Live query: the device must be online; "
        "offline devices return a 504 error. Not the same as network services (listening ports)."
    ),
    annotations=_READ_ONLY,
)
def get_device_system_services(
    device: str,
    service: str | None = None,
    deaths: int | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        report = get_client().get_device_system_services(
            found["match"]["id"], org_id=found["organization_id"], name=service, deaths=deaths
        )
        services = (report or {}).get("services") or []
        needs_attention = [
            {"name": s.get("name"), "health": s.get("health"), "reason": s.get("reason")}
            for s in services
            if s.get("health") in ("flapping", "unstable", "restarting", "unsupervised")
            or (s.get("health") == "stopped" and s.get("normallyUp", True))
        ]
        return _dumps({"device": found["match"], "needs_attention": needs_attention, "system_services": report})
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="Current workload container state on a device (running/stopped/exited/unknown), plus live configuration identity in running (configurationName, image, version). Offline devices often report state=unknown.", annotations=_READ_ONLY)
def get_device_workload(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        workload = get_client().get_device_workload(found["match"]["id"], org_id=found["organization_id"])
        return _dumps({"device": found["match"], "workload": workload})
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="Latest CPU/memory/disk/health stats. include_metrics=true adds gauge data. Offline devices may have stale last_metric_time.", annotations=_READ_ONLY)
def get_device_stats(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        stats = get_client().get_device_stats(found["match"]["id"], org_id=found["organization_id"])
        return _dumps({"device": found["match"], "health": health_from_stats(stats), "stats": stats})
    except Exception as exc:
        return _err(exc)


_METRIC_CHOICES = ("cpu", "memory", "disk", "network_rx", "network_tx")


@mcp.tool(
    description=(
        "Historical metric timeseries for a device (VictoriaMetrics), for questions about trends "
        "over time — use get_device_stats for the current snapshot. metric is one of cpu, memory, "
        "disk, network_rx, network_tx (percentage gauges except network, which is bytes/second rate). "
        "lookback_hours defaults to 24; the server picks the bucket interval (5m under 24h, 30m to 7d, "
        "1h beyond). Returns meta (metric, unit, start, end, interval) and series with labels and "
        "[unix_ts, value] points. series is null when the window has no data — not an error; "
        "shorten the window or check the device was online and reporting then."
    ),
    annotations=_READ_ONLY,
)
def get_device_metrics(
    device: str,
    metric: str = "cpu",
    lookback_hours: float = 24,
    organization_id: str | None = None,
) -> str:
    try:
        if metric not in _METRIC_CHOICES:
            return _dumps({"error": f"Unknown metric {metric!r}", "choices": list(_METRIC_CHOICES)})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        match = found["match"]
        fleet_id = (match.get("fleet") or {}).get("id") or match.get("fleet_id")
        if not fleet_id:
            return _dumps({"device": match, "error": "Device has no fleet_id; metrics API requires it"})
        start, end = _rfc3339_hours_ago(lookback_hours)
        payload = get_client().get_device_metrics(
            match["id"],
            org_id=found["organization_id"],
            metric_name=metric,
            fleet_id=fleet_id,
            start=start,
            end=end,
        )
        return _dumps({"device": match, "query": {"metric": metric, "start": start, "end": end}, "result": payload})
    except Exception as exc:
        return _err(exc)


def _fleet_target(fleet: str | None, device: str | None, organization_id: str | None) -> tuple[str, dict[str, Any], str | None] | str:
    """(org_id, {id, name}, device_id) or a JSON string to return as-is (ambiguous/unknown)."""
    client = get_client()
    device_id = None
    if device:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        match = found["match"]
        device_id = match["id"]
        org_id = found["organization_id"]
        if not fleet:
            fl = match.get("fleet") or {}
            if not fl.get("id"):
                return _dumps({"device": match, "error": "Device has no fleet; pass fleet explicitly"})
            return org_id, {"id": fl["id"], "name": fl.get("name")}, device_id
    else:
        org_id = DeviceResolver(client).resolve_org(organization_id)
    if not fleet:
        return _dumps({"error": "fleet (name or id) is required unless device is given"})
    res = fleetmetrics.resolve_fleet(client, fleet, org_id)
    if not res.get("match"):
        return _dumps(res)
    return org_id, res["match"], device_id


@mcp.tool(
    description=(
        "Metric statistics for a whole fleet (VictoriaMetrics) in one call: avg, max, latest and device count over "
        "the window, plus (per_device=true) a ranked per-device table with names. Pass fleet as a name or id. To "
        "compare one device with its fleet, pass device (name, tag or id; fleet is then optional): the result adds "
        "that device's avg, rank and difference from the fleet average. metric is cpu, memory, disk (worst "
        "partition), network_rx or network_tx (bytes/second). lookback_hours defaults to 24. Empty windows return "
        "a summary saying no data, not an error. A 402 means the organisation lacks the Telemetry add-on."
    ),
    annotations=_READ_ONLY,
)
def get_fleet_metrics(
    fleet: str | None = None,
    metric: str = "cpu",
    lookback_hours: float = 24,
    per_device: bool = False,
    device: str | None = None,
    limit: int = 25,
    organization_id: str | None = None,
) -> str:
    try:
        if metric not in fleetmetrics.METRICS:
            return _dumps({"error": f"Unknown metric {metric!r}", "choices": list(fleetmetrics.METRICS)})
        target = _fleet_target(fleet, device, organization_id)
        if isinstance(target, str):
            return target
        org_id, fl, device_id = target
        start, end = _rfc3339_hours_ago(lookback_hours)
        try:
            out = fleetmetrics.fleet_metrics(
                get_client(), fleet=fl, org_id=org_id, metric=metric, start=start, end=end,
                per_device=per_device, limit=limit, device_id=device_id,
            )
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Fleet metrics")
            if gated is None:
                raise
            return _dumps({**gated, "fleet": fl})
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Fleet health snapshot: total devices, online, offline and the fleet's average CPU, memory and disk. "
        "Pass fleet as a name or id. For which devices are the problem use fleet_health_report; for trends use "
        "get_fleet_metrics."
    ),
    annotations=_READ_ONLY,
)
def get_fleet_health(fleet: str, organization_id: str | None = None) -> str:
    try:
        target = _fleet_target(fleet, None, organization_id)
        if isinstance(target, str):
            return target
        org_id, fl, _ = target
        try:
            data = get_client().get_fleet_health(fl["id"], org_id=org_id)
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Fleet health")
            if gated is None:
                raise
            return _dumps({**gated, "fleet": fl})
        data = data if isinstance(data, dict) else {}
        total, online = data.get("total_devices"), data.get("online")
        offline = data.get("offline")
        if offline is None and isinstance(total, int) and isinstance(online, int):
            offline = total - online
        stats = {k: data.get(k) for k in ("total_devices", "online", "avg_cpu", "avg_memory", "avg_disk")}
        stats["offline"] = offline
        return _dumps(
            {
                "summary": (
                    f"Fleet {fl.get('name') or fl['id']}: {online}/{total} online, avg CPU {data.get('avg_cpu')}%, "
                    f"memory {data.get('avg_memory')}%, disk {data.get('avg_disk')}%."
                ),
                "fleet": fl,
                "health": stats,
                **(
                    {
                        "note": (
                            "The platform reports 0 devices for this fleet's health snapshot. Cross-check with "
                            "list_devices(fleet_id=...) or get_fleet_metrics before concluding it is empty."
                        )
                    }
                    if not total
                    else {}
                ),
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Fleet uptime: the current uptime percentage (last hour) plus uptime buckets (period_type hourly, daily, "
        "weekly or monthly; periods_back buckets, default 7 daily). Pass fleet as a name or id. Hours before a "
        "device's current boot count as downtime."
    ),
    annotations=_READ_ONLY,
)
def get_fleet_uptime(
    fleet: str,
    period_type: str = "daily",
    periods_back: int = 7,
    organization_id: str | None = None,
) -> str:
    try:
        if period_type not in ("hourly", "daily", "weekly", "monthly"):
            return _dumps({"error": f"Unknown period_type {period_type!r}", "choices": ["hourly", "daily", "weekly", "monthly"]})
        target = _fleet_target(fleet, None, organization_id)
        if isinstance(target, str):
            return target
        org_id, fl, _ = target
        client = get_client()
        try:
            buckets = client.get_fleet_uptime(fl["id"], period_type=period_type, periods_back=periods_back, org_id=org_id)
            current = client.get_fleet_uptime_percentage(fl["id"], org_id=org_id)
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Fleet uptime")
            if gated is None:
                raise
            return _dumps({**gated, "fleet": fl})
        return _dumps(fleetmetrics.fleet_uptime_view(buckets, current, fl, period_type))
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Organisation-wide metric question in one call: 'CPU across the org', 'which fleet runs hottest', 'top 10 "
        "devices by memory'. metric is cpu, memory, disk, network_rx or network_tx (bytes/second); group_by is "
        "fleet or device; stat (avg or max) is what rows are ranked by; fleet optionally narrows to one fleet "
        "(name or id). Returns org_stats (avg, max, latest, device count) plus ranked rows with fleet/device names, "
        "so a device can be compared against its fleet and the org average. Uses the organisation-scoped "
        "Telemetry query API (Telemetry add-on; a 402 or 403 is reported as unavailable, not an error)."
    ),
    annotations=_READ_ONLY,
)
def get_org_metrics(
    metric: str = "cpu",
    group_by: str = "fleet",
    stat: str = "avg",
    fleet: str | None = None,
    lookback_hours: float = 24,
    limit: int = 10,
    ascending: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        for value, choices, label in (
            (metric, tuple(fleetmetrics.METRICS), "metric"),
            (group_by, fleetmetrics.GROUPS, "group_by"),
            (stat, fleetmetrics.STATS, "stat"),
        ):
            if value not in choices:
                return _dumps({"error": f"Unknown {label} {value!r}", "choices": list(choices)})
        if not 0 < lookback_hours <= fleetmetrics.MAX_RANGE_HOURS:
            return _dumps({"error": f"lookback_hours must be between 0 and {fleetmetrics.MAX_RANGE_HOURS}"})
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        fl = None
        if fleet:
            res = fleetmetrics.resolve_fleet(client, fleet, org_id)
            if not res.get("match"):
                return _dumps(res)
            fl = res["match"]
        try:
            out = fleetmetrics.org_metrics(
                client, org_id=org_id, metric=metric, group_by=group_by, stat=stat,
                hours=lookback_hours, limit=min(max(limit, 1), 100), fleet=fl,
                sort="asc" if ascending else "desc",
            )
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Organisation metrics")
            if gated is None:
                raise
            return _dumps(gated)
        out["organization_id"] = org_id
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Advanced: run a raw PromQL/MetricsQL query through the organisation-scoped Telemetry API (read-only; the "
        "organisation and your access are enforced server-side). mode=instant evaluates now; mode=range uses "
        "lookback_hours (max 168) and a step (default auto, at most 500 points). Stored metric names look like "
        "edge_cpu_usagepercent, edge_memory_usagepercent, edge_disk_usagepercent; series carry device_id and "
        "fleet_id labels. Output is bounded (25 series, 60 points each). Prefer get_org_metrics / "
        "get_fleet_metrics for ordinary questions. A 402 means the Telemetry add-on is not enabled."
    ),
    annotations=_READ_ONLY,
)
def query_telemetry_metrics(
    query: str,
    mode: str = "instant",
    lookback_hours: float = 1,
    step: str | None = None,
    fleet: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        if mode not in ("instant", "range"):
            return _dumps({"error": f"Unknown mode {mode!r}", "choices": ["instant", "range"]})
        q = (query or "").strip()
        if not q or len(q) > fleetmetrics.MAX_QUERY_CHARS:
            return _dumps({"error": f"query must be 1-{fleetmetrics.MAX_QUERY_CHARS} characters"})
        if not 0 < lookback_hours <= fleetmetrics.MAX_RANGE_HOURS:
            return _dumps({"error": f"lookback_hours must be between 0 and {fleetmetrics.MAX_RANGE_HOURS}"})
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        extra = None
        if fleet:
            res = fleetmetrics.resolve_fleet(client, fleet, org_id)
            if not res.get("match"):
                return _dumps(res)
            extra = {"scope_fleet_id": res["match"]["id"]}
        start = end = None
        if mode == "range":
            start, end = _rfc3339_hours_ago(lookback_hours)
            floor = max(15, int(lookback_hours * 3600 / fleetmetrics.MAX_RANGE_POINTS))
            step = step if step and step.rstrip("smhd").isdigit() and _step_seconds(step) >= floor else f"{floor}s"
        try:
            payload = client.telemetry_query(q, org_id=org_id, start=start, end=end, step=step, extra=extra)
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Telemetry query")
            if gated is None:
                raise
            return _dumps(gated)
        out = fleetmetrics.compact_prom(payload, mode)
        out = {
            "summary": f"{out['series_total']} series for {mode} query.",
            "query": {"promql": q, "mode": mode, "start": start, "end": end, "step": step},
            **out,
        }
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


def _step_seconds(step: str) -> int:
    return int(step[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[step[-1]] if step[-1] in "smhd" else int(step)


@mcp.tool(
    description=(
        "What telemetry this caller can query in the organisation: org_wide, or the specific fleets/devices "
        "granted (fleet names resolved). Use it to explain empty org-wide results or a 403. A 402 means the "
        "Telemetry add-on is not enabled."
    ),
    annotations=_READ_ONLY,
)
def get_telemetry_scope(organization_id: str | None = None) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        try:
            scope = client.get_telemetry_scope(org_id=org_id)
        except AdmiralAPIError as exc:
            gated = fleetmetrics.gate_result(exc, "Telemetry scope")
            if gated is None:
                raise
            return _dumps(gated)
        scope = scope if isinstance(scope, dict) else {}
        fleet_ids = scope.get("fleet_ids") or []
        device_ids = scope.get("device_ids") or []
        names = fleetmetrics.fleet_names(client, org_id) if fleet_ids else {}
        org_wide = bool(scope.get("org_wide"))
        return _dumps(
            {
                "summary": (
                    "Telemetry is org-wide: all fleets and devices in the organisation are queryable."
                    if org_wide and not fleet_ids and not device_ids
                    else f"Telemetry is limited to {len(fleet_ids)} fleet(s) and {len(device_ids)} device(s)."
                ),
                "organization_id": scope.get("org_id") or org_id,
                "org_wide": org_wide,
                "fleets": [{"id": f, "name": names.get(f)} for f in fleet_ids[:50]],
                "device_ids": device_ids[:50],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Live screenshot of a device's display. Resolves the device first. "
        "This is a live capture round-trip through the platform to the device: it must be "
        "online and reachable, takes seconds, and a 504 means the device did not answer in time. "
        "Returns capture metadata (format, width, height, timestamp) as text plus the image "
        "content block. The device chooses the encoding (png or jpeg) — it cannot be requested — "
        "and the metadata reports the actual one."
    ),
    structured_output=False,
    annotations=_READ_ONLY,
)
def get_device_screenshot(
    device: str,
    display: int = 0,
    organization_id: str | None = None,
) -> list[str | Image]:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return [_dumps(found)]
        shot = get_client().get_device_screenshot(
            found["match"]["id"],
            org_id=found["organization_id"],
            display=display,
        )
        shot = shot if isinstance(shot, dict) else {}
        encoded = shot.get("imageData")
        if not encoded:
            return [_dumps({"device": found["match"], "error": "Empty screenshot data"})]
        raw = base64.b64decode(encoded)
        actual = _image_format(raw, shot.get("format"))
        meta = {
            "device": found["match"],
            "format": actual,
            "width": shot.get("width"),
            "height": shot.get("height"),
            "timestamp": shot.get("timestamp"),
        }
        return [_dumps(meta), Image(data=raw, format=actual)]
    except Exception as exc:
        return [_err(exc)]


@mcp.tool(
    description=(
        "Historical logs for a device from VictoriaLogs (kernel, supervisor, workload). "
        "Works for offline and crashing devices. PAT cannot open the live websocket tail. "
        "level is one of debug, info, warn, error, fatal. lookback_hours defaults to 24. "
        "Pagination: when page.next_before is set, pass it as `before` to fetch the next-older page "
        "(lookback_hours stays the window size); it is None once the oldest page is reached."
    ),
    annotations=_READ_ONLY,
)
def get_device_logs(
    device: str,
    lookback_hours: float = 24,
    level: str | None = None,
    search: str | None = None,
    limit: int = 200,
    before: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        start, end = _log_window(lookback_hours, before)
        client = get_client()
        org_id = found["organization_id"]
        device_id = found["match"]["id"]
        payload = client.fetch_logs(
            device_id=device_id,
            org_id=org_id,
            start=start,
            end=end,
            level=level,
            search=search,
            limit=min(max(limit, 1), 1000),
        )
        distilled = distill_logs(payload)
        logs = payload.get("logs") or []
        has_more = bool(payload.get("has_more"))
        return _dumps(
            {
                "device": found["match"],
                "query": {"start": start, "end": end, "level": level, "search": search},
                "log_source": payload.get("source"),
                "page": {"has_more": has_more, "next_before": log_page_cursor(logs, has_more)},
                "summary": distilled,
                "logs": logs,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="Structured device lifecycle events (online/offline, workload, rollout). Complementary to log lines.", annotations=_READ_ONLY)
def get_device_events(
    device: str,
    lookback_hours: float = 24,
    limit: int = 50,
    event: str | None = None,
    source: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        start, end = _rfc3339_hours_ago(lookback_hours)
        payload = get_client().get_device_events(
            found["match"]["id"],
            org_id=found["organization_id"],
            start=start,
            end=end,
            limit=limit,
            event=event,
            source=source,
        )
        return _dumps(
            {
                "device": found["match"],
                "summary": distill_events(payload),
                "events": payload,
            }
        )
    except Exception as inf:
        return _err(inf)


def _local_diagnosis(client: AdmiralClient, device_id: str, org_id: str, start: str, end: str) -> dict[str, Any]:
    """Client-side diagnosis from workload/stats/events/logs (distill.py).

    Fallback for backends without GET /devices/{id}/diagnose.
    """
    report: dict[str, Any] = {}
    try:
        report["workload"] = client.get_device_workload(device_id, org_id=org_id)
    except AdmiralAPIError as exc:
        report["workload_error"] = str(exc)
    try:
        stats = client.get_device_stats(device_id, org_id=org_id)
        report["health"] = health_from_stats(stats)
    except AdmiralAPIError as exc:
        report["health_error"] = str(exc)
    try:
        events = client.get_device_events(device_id, org_id=org_id, start=start, end=end, limit=50)
        report["events"] = distill_events(events)
    except AdmiralAPIError as exc:
        report["events_error"] = str(exc)
    try:
        logs = client.fetch_logs(
            org_id=org_id,
            device_id=device_id,
            start=start,
            end=end,
            limit=300,
        )
        report["logs"] = distill_logs(logs)
        report["log_source"] = logs.get("source")
    except AdmiralAPIError as exc:
        report["logs_error"] = str(exc)
    return report


# Statuses meaning "this backend has no server-side diagnose" (route missing,
# method not routed, diagnoser not wired) rather than a real failure.
_DIAGNOSE_FALLBACK_STATUSES = (404, 405, 501, 503)


def _diagnose_one(client: AdmiralClient, target: dict[str, Any], org_id: str, start: str, end: str) -> dict[str, Any]:
    report: dict[str, Any] = {"device": target}
    try:
        report["diagnosis"] = client.diagnose(target["id"], org_id=org_id)
        report["source"] = "server"
        # Surface the check-in / restart schedule at the top of the report
        # (absent on backends that predate it).
        diagnosis = report["diagnosis"]
        if isinstance(diagnosis, dict) and isinstance(diagnosis.get("schedule"), dict):
            report["schedule"] = diagnosis["schedule"]
        return report
    except AdmiralAPIError as exc:
        if exc.status not in _DIAGNOSE_FALLBACK_STATUSES:
            raise
        report["source"] = "local_fallback"
        report["fallback_reason"] = str(exc)
    report.update(_local_diagnosis(client, target["id"], org_id, start, end))
    return report


@mcp.tool(
    description=(
        "Diagnose a device (or the offline/error set). Uses the backend's one-call verdict "
        "(GET /devices/{id}/diagnose, stored data only, nothing round-trips to the device): verdict "
        "{online, since, workloadState, transitionAge, healthScore, topIssue}, severity-ordered issues with "
        "evidence, lastReboot, and signals, plus schedule (heartbeat interval, last/next check-in, overdue, and when "
        "the device restarts if it stays offline: restart.at/reason/basis/thenEveryMs, limits) copied to report.schedule. When the backend has no diagnose route it falls back to a "
        "client-side distillation of workload, health stats, events and historical logs (source=local_fallback; "
        "lookback_hours applies only there). If device is omitted, inspects currently offline and error devices."
    ),
    annotations=_READ_ONLY,
)
def diagnose_device(
    device: str | None = None,
    lookback_hours: float = 24,
    organization_id: str | None = None,
    fleet_id: str | None = None,
) -> str:
    try:
        client = get_client()
        resolver = DeviceResolver(client)
        org_id = resolver.resolve_org(organization_id)
        targets: list[dict[str, Any]] = []
        if device:
            found = resolver.find(device, org_id=org_id, fleet_id=fleet_id)
            if not found.get("match"):
                return _dumps(found)
            targets = [found["match"]]
        else:
            # The status=offline filter matches almost nothing; the counts.offline
            # bucket is every device not currently online (claimed/error/stale).
            seen: set[str] = set()
            for raw in resolver.list_not_online(org_id=org_id, fleet_id=fleet_id, limit=100):
                summary = summarise_device(raw)
                if summary["id"] in seen:
                    continue
                seen.add(summary["id"])
                targets.append(summary)
            if not targets:
                return _dumps({"organization_id": org_id, "message": "No offline or error devices.", "devices": []})

        start, end = _rfc3339_hours_ago(lookback_hours)
        reports = []
        for target in targets[:15]:
            try:
                reports.append(_diagnose_one(client, target, org_id, start, end))
            except AdmiralAPIError as exc:
                reports.append({"device": target, "error": str(exc)})

        return _dumps(
            {
                "organization_id": org_id,
                "window": {"start": start, "end": end},
                "devices_inspected": len(reports),
                "reports": reports,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Device policy for a device: status.policy of the device document = {requested (layered org < fleet < device, "
        "with sources), reported (what the device says it runs), clamped (paths the device had to clamp)}; also "
        "spec.policy (the device-level override) and resource_version. Legacy (protocol 0) devices never receive "
        "policy. Change it with patch_device_document ({\"spec\": {\"policy\": {...}}})."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_device_policy(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        response = get_client().get_device_document(found["match"]["id"], org_id=found["organization_id"], fmt="json")
        doc = _json_or_text(response)
        if not isinstance(doc, dict):
            return _dumps({"device": found["match"], "error": "Device document was not a JSON object", "document": doc})
        if isinstance(doc.get("data"), dict) and "status" not in doc:
            doc = doc["data"]
        status = doc.get("status") if isinstance(doc.get("status"), dict) else {}
        spec = doc.get("spec") if isinstance(doc.get("spec"), dict) else {}
        policy = status.get("policy")
        out: dict[str, Any] = {
            "device": found["match"],
            "resource_version": _etag(response),
            "policy": policy,
            "override": spec.get("policy"),
            "protocol": status.get("protocol"),
        }
        if policy is None:
            out["note"] = "No status.policy: the backend does not report policy, or the device is on protocol 0."
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


def _duplicate_source(client: AdmiralClient, ref: str, org_id: str, what: str) -> dict[str, Any]:
    if is_uuid(ref):
        return {"id": ref.strip(), "name": None}
    lister = client.list_fleets if what == "Fleet" else client.list_configurations
    return _pick_by_name(_items(lister(org_id=org_id, search=ref, limit=100)), ref, what)


@mcp.tool(
    description=(
        "Duplicate a fleet (POST /fleets/{id}/duplicate). fleet = name or UUID; name = the new fleet's name (1-100 "
        "chars); description defaults to the source's; copy_configuration (default true) also points the copy at the "
        "same configuration. Copies description, location, tags, groups, update window/policy, USB/security/signature/"
        "device policy and custom metrics; never copies devices or billed features (airdetect, edgewire, image proxy). "
        "Returns {fleet, copied, skipped:[{field, reason}]}: report skipped to the user. Needs fleet viewer + org "
        "fleet_creator. Creates a resource — explicit request only."
    ),
    annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False, readOnlyHint=False),
)
def duplicate_fleet(
    fleet: str,
    name: str,
    description: str | None = None,
    copy_configuration: bool = True,
    organization_id: str | None = None,
) -> str:
    try:
        if not name or not name.strip():
            return _dumps({"error": "name is required"})
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        source = _duplicate_source(client, fleet, org_id, "Fleet")
        result = client.duplicate_fleet(
            source["id"],
            name.strip(),
            org_id=org_id,
            description=description,
            copy_configuration=copy_configuration,
        )
        return _dumps({"organization_id": org_id, "source": source, "result": result})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Duplicate a configuration (POST /configurations/{id}/duplicate). configuration = name or UUID; name = the "
        "new configuration's name; description optional; version = source version to copy (omit/0 = latest). The "
        "copy starts at version 1 with the same tags. Secret files are not copied. Returns {configuration, "
        "sourceVersion, skipped:[{field, reason}]}: report skipped to the user. Needs configuration viewer + org "
        "config_creator. Creates a resource — explicit request only."
    ),
    annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False, readOnlyHint=False),
)
def duplicate_configuration(
    configuration: str,
    name: str,
    description: str | None = None,
    version: int | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        if not name or not name.strip():
            return _dumps({"error": "name is required"})
        if version is not None and int(version) < 0:
            return _dumps({"error": "version must be a positive integer (omit for latest)"})
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        source = _duplicate_source(client, configuration, org_id, "Configuration")
        result = client.duplicate_configuration(
            source["id"],
            name.strip(),
            org_id=org_id,
            description=description,
            version=version,
        )
        return _dumps({"organization_id": org_id, "source": source, "result": result})
    except Exception as exc:
        return _err(exc)


_TICKET_PRIORITY = {"high": 1, "normal": 2, "low": 3}
_TICKET_TYPES = ("technical", "billing", "implementation")


def _ticket_summary(t: dict[str, Any]) -> dict[str, Any]:
    subject = t.get("subject") if isinstance(t.get("subject"), dict) else None
    return {
        "ref": t.get("ref"),
        "id": t.get("id"),
        "title": t.get("title"),
        "status": t.get("status"),
        "priority": t.get("priority"),
        "supportType": t.get("supportType"),
        "subject": ({"type": subject.get("type"), "id": subject.get("id"), "name": subject.get("name")} if subject else None),
        "createdAt": t.get("createdAt"),
    }


def _resolve_named(client: AdmiralClient, kind: str, ref: str, org_id: str) -> dict[str, Any]:
    """Resolve a fleet/configuration by UUID (verified by GET) or name. Ambiguity -> candidates."""
    ref = ref.strip()
    path = "fleets" if kind == "fleet" else "configurations"
    if is_uuid(ref):
        row = client.get(f"{path}/{ref}", org_id=org_id)
        row = row if isinstance(row, dict) else {}
        return {"match": {"id": ref, "name": row.get("name")}, "candidates": []}
    lister = client.list_fleets if kind == "fleet" else client.list_configurations
    rows = _items(lister(org_id=org_id, search=ref, limit=100))
    needle = ref.lower()
    hits = [r for r in rows if str(r.get("name") or "").lower() == needle] or [
        r for r in rows if needle in str(r.get("name") or "").lower()
    ]
    cands = [{"id": r.get("id"), "name": r.get("name")} for r in hits[:10]]
    if len(hits) == 1:
        return {"match": cands[0], "candidates": []}
    return {"match": None, "candidates": cands, "kind": kind, "query": ref}


@mcp.tool(
    description=(
        "Raise a support ticket with the Admiral support team (visible to staff and to the user on the dashboard "
        "Help page). Only call when the user asks for a ticket or agrees to one you offered; never on your own "
        "initiative. Put everything a support engineer needs in description: what is wrong, what the user "
        "expected, what was already tried, and the findings from other tools (device status, errors, log "
        "excerpts, versions, timestamps). support_type: 'technical' (default), 'billing' or 'implementation'. "
        "priority: 'normal' (default), 'high' or 'low'. This tool cannot raise an urgent/Sev 1 ticket: for a "
        "production outage that must page on-call, tell the user to raise it from the dashboard Help page. "
        "Optionally tie the ticket to ONE subject: device (name, UUID or tag; resolved like reboot_device, "
        "ambiguous matches return candidates and nothing is created), fleet (name or UUID) or configuration "
        "(name or UUID). attach_diagnostics=true (device only) also collects device state, logs and system info "
        "into a diagnostics bundle for staff; the device must be online, otherwise the ticket is still created "
        "and diagnostics.status is 'unavailable' with a reason. Returns {summary, ticket, diagnostics, next_step}."
    ),
    annotations=ToolAnnotations(destructiveHint=False, idempotentHint=False, readOnlyHint=False),
)
def create_support_ticket(
    title: str,
    description: str,
    support_type: str = "technical",
    priority: str = "normal",
    device: str | None = None,
    fleet: str | None = None,
    configuration: str | None = None,
    attach_diagnostics: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        title = (title or "").strip()
        description = (description or "").strip()
        if not title or not description:
            return _dumps({"error": "title and description are required"})
        stype = (support_type or "technical").strip().lower()
        if stype not in _TICKET_TYPES:
            return _dumps({"error": f"Unsupported support_type {support_type!r}", "choices": list(_TICKET_TYPES)})
        prio = (priority or "normal").strip().lower()
        if prio not in _TICKET_PRIORITY:
            return _dumps(
                {
                    "error": f"Unsupported priority {priority!r}",
                    "choices": list(_TICKET_PRIORITY),
                    "note": "Urgent/Sev 1 pages on-call and is only available from the dashboard Help page.",
                }
            )
        given = {k: v for k, v in (("device", device), ("fleet", fleet), ("configuration", configuration)) if v and str(v).strip()}
        if len(given) > 1:
            return _dumps({"error": "Pass at most one of device, fleet or configuration", "given": sorted(given)})
        if attach_diagnostics and "device" not in given:
            return _dumps({"error": "attach_diagnostics requires a device subject"})

        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        body: dict[str, Any] = {
            "title": title,
            "description": description,
            "priority": _TICKET_PRIORITY[prio],
            "supportType": stype.upper(),
        }
        subject: dict[str, Any] | None = None
        if "device" in given:
            found = _resolve_device(given["device"], org_id)
            if not found.get("match"):
                return _dumps({**found, "note": "No ticket created. Pass a device id (UUID) to disambiguate."})
            subject = {"type": "device", "id": found["match"]["id"], "name": found["match"].get("name")}
        elif given:
            kind = "fleet" if "fleet" in given else "configuration"
            found = _resolve_named(client, kind, given[kind], org_id)
            if not found.get("match"):
                return _dumps({**found, "note": f"No ticket created. Pass the {kind} id (UUID) to disambiguate."})
            subject = {"type": kind, "id": found["match"]["id"], "name": found["match"].get("name")}
        if subject:
            body["subject"] = {"type": subject["type"], "id": subject["id"]}
            if attach_diagnostics:
                body["attachDiagnostics"] = True

        data = client.post("helpdesk/tickets", org_id=org_id, json=body)
        ticket = _ticket_summary(data if isinstance(data, dict) else {})
        if subject and not ticket["subject"]:
            ticket["subject"] = subject
        elif ticket["subject"] and not ticket["subject"].get("name"):
            ticket["subject"]["name"] = subject["name"] if subject else None
        diag = data.get("diagnostics") if isinstance(data, dict) and isinstance(data.get("diagnostics"), dict) else None
        label = ticket["ref"] or ticket["id"] or "(no ref)"
        text = f"Ticket {label} raised ({stype}, {prio})"
        if subject:
            text += f" about {subject['type']} {subject.get('name') or subject['id']}"
        if diag:
            status = diag.get("status")
            text += (
                "; diagnostics bundle requested"
                if status == "requested"
                else f"; diagnostics unavailable ({diag.get('reason') or 'no reason given'})"
            )
        return _dumps(
            {
                "summary": text,
                "ticket": ticket,
                "diagnostics": diag,
                "organization_id": org_id,
                "next_step": "Follow up on the dashboard Help page; the support team replies there.",
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "List the organisation's support tickets (newest first, compact: ref, title, status, priority, type, "
        "subject, created). Use it to show existing tickets or to check for a duplicate before offering "
        "create_support_ticket. Pass 'after' (next_cursor from a previous page) to page."
    ),
    annotations=_READ_ONLY,
)
def list_support_tickets(limit: int = 20, after: str | None = None, organization_id: str | None = None) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        n = max(1, min(int(limit), 50))
        page = client.get("helpdesk/tickets", org_id=org_id, params={"first": n, "after": after})
        page = page if isinstance(page, dict) else {}
        rows = [_ticket_summary(t) for t in (page.get("tickets") or []) if isinstance(t, dict)]
        return _dumps(
            {
                "summary": f"{len(rows)} of {page.get('totalCount', len(rows))} tickets",
                "tickets": rows,
                "has_next": bool(page.get("hasNext")),
                "next_cursor": page.get("nextCursor") or None,
                "organization_id": org_id,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(description="Organisation-wide search across devices, fleets, and configurations.", annotations=_READ_ONLY)
def search(query: str, organization_id: str | None = None, limit: int = 20) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        return _dumps(client.search(query, org_id=org_id, limit=limit))
    except Exception as inf:
        return _err(inf)


@mcp.tool(
    description="Reboot a device. Destructive. Only call when the user explicitly asked to reboot this specific device.",
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def reboot_device(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        result = get_client().reboot_device(found["match"]["id"], org_id=found["organization_id"])
        return _dumps({"device": found["match"], "result": result})
    except Exception as inf:
        return _err(inf)


@mcp.tool(
    description=(
        "Change a device's workload status: start, stop, restart, or recreate. "
        "Resolves the device first (ambiguous names return candidates and send nothing). "
        "start/stop toggle the current workload; restart relaunches the same image; "
        "recreate tears down the workload bundle and rebuilds it from the current configuration. "
        "Destructive — only on an explicit user request for a uniquely resolved device. "
        "The result is the edge's acknowledgement, not steady state: confirm with "
        "get_device_workload afterwards. Runtime-only: the configuration still points at the "
        "workload, so a config push or rollout can start a stopped workload again."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def change_workload_status(
    device: str,
    action: str,
    organization_id: str | None = None,
) -> str:
    try:
        normalised = (action or "").strip().lower()
        if normalised not in AdmiralClient.WORKLOAD_ACTIONS:
            # Reject before resolving: an invalid action must never reach the edge.
            return _dumps(
                {
                    "error": f"Unsupported action {action!r}",
                    "choices": list(AdmiralClient.WORKLOAD_ACTIONS),
                }
            )
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        result = get_client().workload_command(
            found["match"]["id"], action=normalised, org_id=found["organization_id"]
        )
        return _dumps(
            {
                "device": found["match"],
                "operation": f"workload_{normalised}",
                "result": result,
                "verify_with": "get_device_workload",
            }
        )
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Observed state, documents, diagnostics (CONTRACT §4.2, ADDENDUM-A §A3)
# ---------------------------------------------------------------------------

_MAX_WATCH_S = 600.0


def _etag(response: Any) -> str | None:
    tag = response.headers.get("etag")
    if not tag:
        return None
    return tag.removeprefix("W/").strip('"')


def _json_or_text(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


@mcp.tool(
    description=(
        "Observed-state document for a device (GET /devices/{id}/state): conditions (Converged, WorkloadReady, "
        "StorageOK, TimeSynced, UpdateTrial, DegradedLink, DiagnosticsMode, LocalOverride), observedGeneration, "
        "workload, network, system versions, boot, transport, time, storage, boot reasons, live progress "
        "operations and localOverride. Returns a compact summary plus the full document. Default is the stored "
        "copy (works offline; protocol 0 = legacy firmware, 1 = desired-state firmware). live=true asks the "
        "device for its cached snapshot (online devices only; 503 device_offline / 504 device_timeout)."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_device_state(device: str, live: bool = False, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        payload = get_client().get_device_state(found["match"]["id"], org_id=found["organization_id"], live=live)
        payload = payload if isinstance(payload, dict) else {}
        return _dumps(
            {
                "device": found["match"],
                "source": payload.get("source"),
                "protocol": payload.get("protocol"),
                "receivedAt": payload.get("receivedAt"),
                "presence": payload.get("presence"),
                "summary": summarise_state(payload.get("state")),
                "state": payload.get("state"),
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Watch a device's observed state live (SSE GET /devices/{id}/state/stream) for up to timeout_s seconds "
        "(default 60, max 600). Returns the initial and final state summaries, condition transitions "
        "(e.g. Converged False→True), observedGeneration / workload state / workload transition changes, every "
        "progress operation seen (image_pull, rootfs_download, … with its phase sequence and last bytes/items/"
        "rate/ETA), and update counts by kind (state, workload_report, heartbeat, network, spec, progress). "
        "stop_when_converged ends early once Converged=True (and observedGeneration >= min_generation when given)."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def watch_device_state(
    device: str,
    timeout_s: float = 60,
    stop_when_converged: bool = False,
    min_generation: int | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        watch = StateWatch()

        def on_event(name: str, data: Any) -> bool:
            if name != "state":
                return False
            watch.add(data)
            if not stop_when_converged or not watch.converged():
                return False
            gen = (watch.last or {}).get("observedGeneration") or 0
            return min_generation is None or gen >= min_generation

        timeout = min(max(float(timeout_s), 1.0), _MAX_WATCH_S)
        stream = get_client().stream_events(
            f"devices/{found['match']['id']}/state/stream",
            on_event,
            org_id=found["organization_id"],
            timeout_s=timeout,
        )
        return _dumps({"device": found["match"], "stream": stream, **watch.result()})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Kubernetes-shaped Device document (GET /devices/{id}/document, apiVersion admrl.co/v1): metadata "
        "(generation, resourceVersion, labels), the writable spec (fleet, workload.from/override, network, "
        "system, secrets) and the observed status (observedGeneration, renderedRevision, conditions, presence, "
        "protocol, localOverride). format=json (default) or yaml. resource_version is the ETag to pass as "
        "if_match to patch_device_document / adopt_local_override."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_device_document(device: str, format: str = "json", organization_id: str | None = None) -> str:
    try:
        fmt = (format or "json").strip().lower()
        if fmt not in ("json", "yaml"):
            return _dumps({"error": f"Unsupported format {format!r}", "choices": ["json", "yaml"]})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        response = get_client().get_device_document(found["match"]["id"], org_id=found["organization_id"], fmt=fmt)
        body = response.text if fmt == "yaml" else _json_or_text(response)
        return _dumps(
            {
                "device": found["match"],
                "resource_version": _etag(response),
                "content_type": response.headers.get("content-type"),
                "document": body,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Change a Device document with a JSON merge patch (PATCH /devices/{id}/document, "
        "application/merge-patch+json), e.g. {\"spec\": {\"system\": {\"screenshots\": \"disabled\"}}} or "
        "{\"spec\": {\"fleet\": \"<fleetId>\"}} (fleet move). Writes go through the platform services, bump "
        "metadata.generation and push desired state to the device. spec.workload.from, spec.system.updates and "
        "spec.secrets are read-only. if_match = resource_version from get_device_document (409 on mismatch). "
        "Returns push outcome (X-Admrl-Push: sent|failed|skipped|unavailable), changed paths and the new document. "
        "Mutating — only on an explicit user request for a uniquely resolved device."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def patch_device_document(
    device: str,
    merge_patch: dict[str, Any] | str,
    if_match: str | None = None,
    change_reason: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        patch = json.loads(merge_patch) if isinstance(merge_patch, str) else merge_patch
        if not isinstance(patch, dict) or not patch:
            return _dumps({"error": "merge_patch must be a non-empty JSON object"})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        response = get_client().patch_device_document(
            found["match"]["id"],
            patch,
            org_id=found["organization_id"],
            if_match=if_match,
            change_reason=change_reason,
        )
        changed = response.headers.get("x-admrl-changed")
        return _dumps(
            {
                "device": found["match"],
                "push": response.headers.get("x-admrl-push"),
                "changed": changed.split(",") if changed else [],
                "resource_version": _etag(response),
                "document": _json_or_text(response),
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Render the desired-state bundle a device would receive (POST /devices/{id}/document:render): bundle "
        "(secrets stripped, credentials redacted), revision, diff [{path, from, to}] against what the device "
        "last reported, converged, pushed. Dry run by default (nothing is sent). push=true also pushes the "
        "bundle to the device — only on an explicit request."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True),
)
def render_device_document(device: str, push: bool = False, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        result = get_client().render_device_document(
            found["match"]["id"], org_id=found["organization_id"], dry_run=not push
        )
        return _dumps({"device": found["match"], "dry_run": not push, "render": result})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Adopt a device's local override (made on the device via TUI, BLE or console) into its spec "
        "(POST /devices/{id}/document:adoptLocalOverride): network → spec.network, system fields → spec.system. "
        "Bumps generation and pushes; the device then clears its override. Paths the backend cannot store are "
        "listed in notAdopted. 409 when there is no local override or if_match is stale. Mutating."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def adopt_local_override(device: str, if_match: str | None = None, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        response = get_client().adopt_local_override(
            found["match"]["id"], org_id=found["organization_id"], if_match=if_match
        )
        return _dumps(
            {"device": found["match"], "resource_version": _etag(response), "result": _json_or_text(response)}
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Tell a device to drop its local override (POST /devices/{id}/document:discardLocalOverride). paths "
        "defaults to [\"*\"]; valid paths: network, system.diagnosticsMode, system.timezone, system.hostname. "
        "Can remove connectivity from a device that depends on a locally entered network. Protocol-1 firmware "
        "only (409 otherwise). Destructive — explicit request only."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True, readOnlyHint=False),
)
def discard_local_override(
    device: str, paths: list[str] | None = None, organization_id: str | None = None
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        result = get_client().discard_local_override(
            found["match"]["id"], org_id=found["organization_id"], paths=paths or ["*"]
        )
        return _dumps({"device": found["match"], "result": result})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Run the structured on-device diagnostics probe now (POST /devices/{id}/diagnostics/probe): sections "
        "such as network, time, identity, wireless, storage, boot, thermal, workload, services (default all). "
        "timeout_ms caps the device side (max 20000). Live round-trip: device must be online (503/504 "
        "otherwise). Items marked sensitive come back as [redacted] for non-staff callers. Audited."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True),
)
def probe_device(
    device: str,
    sections: list[str] | None = None,
    timeout_ms: int | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        report = get_client().probe_device(
            found["match"]["id"], org_id=found["organization_id"], sections=sections, timeout_ms=timeout_ms
        )
        return _dumps({"device": found["match"], "report": report})
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Memory tests (online modes only)
# ---------------------------------------------------------------------------


def _memtest_error(exc: Exception) -> str:
    if isinstance(exc, AdmiralAPIError):
        return _dumps(memtest.describe_error(exc))
    return _err(exc)


@mcp.tool(
    description=(
        "Start a RAM test on a device. mode='live': the workload keeps running and a bounded slice of free "
        "memory is tested (quick=true is a shorter, lighter preset). mode='full_online': the workload is STOPPED "
        "for the whole test so nearly all free memory is tested; because that interrupts the workload you must "
        "pass confirm=true after the user agreed. passes sets the number of passes (live default 1, full online "
        "default 2). A test boot (reboot into a dedicated test) is not available here: the user starts that "
        "from the dashboard or the device console. The device must be online. One test per device at a time; "
        "starts are rate-limited to one per 5 seconds. Returns the run id: follow progress with get_memory_test, "
        "stop it with cancel_memory_test. Only on an explicit user request."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def start_memory_test(
    device: str,
    mode: str = "live",
    quick: bool | None = None,
    passes: int | None = None,
    confirm: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        norm = (mode or "").strip().lower().replace("-", "_").replace(" ", "_")
        if norm in ("offline", "test_boot", "testboot", "boot"):
            return _dumps({"error": memtest.OFFLINE_MESSAGE, "code": "operator_required"})
        if norm not in memtest.MODES:
            return _dumps({"error": f"Unknown mode {mode!r}. Use 'live' or 'full_online'."})
        if norm == "full_online" and confirm is not True:
            return _dumps(
                {
                    "error": "confirmation_required",
                    "message": (
                        "mode 'full_online' stops the device's workload for the duration of the test (typically tens of "
                        "minutes or more). Ask the user to confirm, then call again with confirm=true. "
                        "mode 'live' leaves the workload running."
                    ),
                }
            )
        if passes is not None and not 1 <= int(passes) <= 20:
            return _dumps({"error": "passes must be between 1 and 20."})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        body = memtest.start_body(norm, quick, passes)
        result = get_client().start_memory_test(found["match"]["id"], body, org_id=found["organization_id"])
        result = result if isinstance(result, dict) else {}
        out: dict[str, Any] = {
            "device": found["match"],
            "started": True,
            "mode": norm,
            "run_id": result.get("runId"),
            "status": memtest.summarise_status(result.get("status")),
            "next": "Follow progress with get_memory_test; stop it with cancel_memory_test.",
        }
        if norm == "full_online":
            out["note"] = "The workload is stopped while the test runs and restarts when it ends."
        return _dumps(out)
    except Exception as exc:
        return _memtest_error(exc)


@mcp.tool(
    description=(
        "Cancel the memory test running on a device. If a full online test had stopped the workload, the device "
        "restarts it. Optional run_id (from start_memory_test) guards against cancelling a different run. "
        "Answers with a clear message when no test is running."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=True, readOnlyHint=False),
)
def cancel_memory_test(device: str, run_id: str | None = None, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        result = get_client().cancel_memory_test(
            found["match"]["id"], run_id=run_id, org_id=found["organization_id"]
        )
        return _dumps({"device": found["match"], "cancelled": True, "result": result})
    except Exception as exc:
        return _memtest_error(exc)


@mcp.tool(
    description=(
        "Memory test status and memory health for a device: the current or last run (phase, pass, coverage %, "
        "errors, temperature, ETA), the verdict, retired pages, memory fault, hardware error counters and what the "
        "device can do (live / full online / test boot). needs_attention lists a memory fault (replace the "
        "board/RAM) and an interrupted last test. Online devices answer live; offline devices return the last "
        "stored report (source='stored')."
    ),
    annotations=_READ_ONLY,
)
def get_memory_test(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        data = get_client().get_memory_test(found["match"]["id"], org_id=found["organization_id"])
        data = data if isinstance(data, dict) else {}
        history = data.get("history") or []
        supported = bool(data.get("supported"))
        out: dict[str, Any] = {
            "device": found["match"],
            "source": data.get("source"),
            "supported": supported,
            "needs_attention": memtest.attention(supported, data.get("status"), data.get("health"), history),
            "status": memtest.summarise_status(data.get("status")),
            "health": memtest.summarise_health(data.get("health"), history),
            "health_reported_at": data.get("healthReportedAt"),
            "history": [memtest.summarise_result(r) for r in history[:5] if isinstance(r, dict)],
        }
        if not supported:
            out["note"] = "This device does not report memory tests (older firmware or unsupported board)."
        return _dumps(out)
    except Exception as exc:
        return _memtest_error(exc)


@mcp.tool(
    description=(
        "Stored memory test results for a device, newest first (limit 1-100, default 10): outcome, verdict, "
        "coverage, error count, retired pages and a few sample errors per run. Works for offline devices."
    ),
    annotations=_READ_ONLY,
)
def list_memory_test_results(device: str, limit: int = 10, organization_id: str | None = None) -> str:
    try:
        if not 1 <= int(limit) <= 100:
            return _dumps({"error": "limit must be between 1 and 100."})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        data = get_client().list_memory_test_results(
            found["match"]["id"], limit=int(limit), org_id=found["organization_id"]
        )
        results = [r for r in ((data or {}).get("results") or []) if isinstance(r, dict)]
        return _dumps(
            {
                "device": found["match"],
                "count": len(results),
                "results": [memtest.summarise_result(r) for r in results],
            }
        )
    except Exception as exc:
        return _memtest_error(exc)


# ---------------------------------------------------------------------------
# Convergence-driven rollouts (ADDENDUM-A §A3)
# ---------------------------------------------------------------------------

_STRATEGY_KEYS = {
    "canary": "canary",
    "maxinflight": "maxInFlight",
    "max_in_flight": "maxInFlight",
    "maxunavailable": "maxUnavailable",
    "max_unavailable": "maxUnavailable",
    "failurethreshold": "failureThreshold",
    "failure_threshold": "failureThreshold",
    "progressdeadline": "progressDeadline",
    "progress_deadline": "progressDeadline",
}


def _normalise_strategy(strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not strategy:
        return None
    out: dict[str, Any] = {}
    for key, value in strategy.items():
        canonical = _STRATEGY_KEYS.get(str(key).lower()) or _STRATEGY_KEYS.get(str(key))
        if canonical is None:
            raise ValueError(f"Unknown strategy field {key!r}; use {sorted(set(_STRATEGY_KEYS.values()))}")
        if value is not None:
            out[canonical] = value
    return out or None


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("items", "fleets", "configurations", "data"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
    return []


def _pick_by_name(rows: list[dict[str, Any]], name: str, what: str) -> dict[str, Any]:
    needle = name.strip().lower()
    exact = [r for r in rows if str(r.get("name") or "").lower() == needle]
    hits = exact or [r for r in rows if needle in str(r.get("name") or "").lower()]
    if len(hits) != 1:
        raise ValueError(
            f"{what} {name!r} matched {len(hits)}; pass a UUID. "
            f"Candidates: {[{'id': r.get('id'), 'name': r.get('name')} for r in hits[:10]]}"
        )
    return hits[0]


@mcp.tool(
    description=(
        "Create a convergence-driven rollout (POST /rollouts) over one or more fleets. fleet = a fleet name/UUID or a "
        "list of them. type 'config' (default): configuration = name or UUID (required), version = integer or \"latest\"; "
        "a config rollout may also switch a fleet to a different configuration, and on completion the fleet is pinned to "
        "the explicit version. Other types need no configuration: 'reboot' (force=true skips the graceful path), "
        "'restart_workload' (optional container_target) and 'system_update' (OS update: use_current_versions=true or "
        "version_ids [UUID]; final_action reboot|none|restart). strategy (all optional; server defaults canary 1, "
        "maxInFlight 50, maxUnavailable 2, failureThreshold 0.1, progressDeadline 30m): {canary, maxInFlight, "
        "maxUnavailable, failureThreshold (0-1 of admitted), progressDeadline (\"30m\" or seconds)}. Devices are admitted "
        "only while online, canary first, then a window of maxInFlight; the rollout pauses on the failure budget and "
        "stays in_progress until every target converges. A config rollout to the version a device already runs never "
        "converges: run preview_rollout first. Changes devices — explicit request only. Follow with watch_rollout."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def create_rollout(
    fleet: str | list[str],
    configuration: str | None = None,
    version: int | str = "latest",
    strategy: dict[str, Any] | None = None,
    name: str | None = None,
    description: str | None = None,
    organization_id: str | None = None,
    type: str = "config",
    force: bool = False,
    container_target: str | None = None,
    use_current_versions: bool = False,
    version_ids: list[str] | None = None,
    final_action: str | None = None,
) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        kind = _rollout_type(type)
        strat = _normalise_strategy(strategy)
        refs = _as_list(fleet)
        if not refs:
            return _dumps({"error": "fleet is required (a name/UUID or a list)."})
        fleet_rows: list[dict[str, Any]] = []
        for ref in refs:
            if is_uuid(ref):
                fleet_rows.append({"id": ref, "name": None})
            else:
                fleet_rows.append(_unique_by_name(_items(client.list_fleets(org_id=org_id, search=ref, limit=100)), ref, "fleet"))
        if len({str(f["id"]).lower() for f in fleet_rows}) != len(fleet_rows):
            return _dumps({"error": "The same fleet was listed twice."})
        fleet_ids = [str(f["id"]) for f in fleet_rows]
        where = (fleet_rows[0].get("name") or fleet_ids[0]) if len(fleet_rows) == 1 else f"{len(fleet_rows)} fleets"

        body: dict[str, Any]
        if kind == "config":
            if not configuration:
                return _dumps({"error": "configuration is required for type=config."})
            if is_uuid(configuration):
                config_row: dict[str, Any] = {"id": configuration.strip()}
                if str(version).strip().lower() == "latest":
                    detail = client.get(f"configurations/{configuration.strip()}", org_id=org_id)
                    config_row = detail if isinstance(detail, dict) else config_row
            else:
                config_row = _unique_by_name(
                    _items(client.list_configurations(org_id=org_id, search=configuration, limit=100)), configuration, "configuration"
                )
            if str(version).strip().lower() == "latest":
                resolved_version = config_row.get("latest_version") or config_row.get("latestVersion")
                if not resolved_version:
                    return _dumps({"error": "Could not resolve the configuration's latest version; pass version explicitly", "configuration": config_row})
            else:
                resolved_version = int(version)
            config_id = config_row.get("id") or configuration
            label = config_row.get("name") or config_id
            body = {
                "name": name or f"{label} v{resolved_version} → {where}",
                "type": "config",
                "fleet_ids": fleet_ids,
                "config_spec": {"config_id": config_id, "config_version": int(resolved_version)},
            }
        else:
            body = {
                "name": name or f"{kind} → {where}",
                "type": kind,
                "fleet_ids": fleet_ids,
                **_type_spec(kind, force, container_target, use_current_versions, version_ids, final_action),
            }
        if description:
            body["description"] = description
        if strat:
            body["strategy"] = strat
        result = client.create_rollout(body, org_id=org_id)
        return _dumps({"request": body, "rollout": result, "watch_with": "watch_rollout"})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Rollout detail (GET /rollouts/{id}): status (pending, scheduled, in_progress, paused, completed, failed, "
        "cancelled, rolled_back), pausedReason, strategy, counts {targets, admitted, inFlight, converged, failed, "
        "stuck, pendingReconnect}, legacy stats, config/version and timings."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def get_rollout(rollout_id: str, organization_id: str | None = None) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        return _dumps({"rollout": client.get_rollout(rollout_id.strip(), org_id=org_id)})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Per-device rollout rows (GET /rollouts/{id}/devices): device_id, device_name, status, phase "
        "(waiting_online, assigned, pulling_image, extracting, downloading, installing, verifying, staged, "
        "starting, converged, failed, unknown), admittedAt, convergedAt, lastProgress, stuckSince, error. "
        "Optional status filter; paged."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def list_rollout_devices(
    rollout_id: str,
    status: str | None = None,
    page: int = 1,
    limit: int = 100,
    organization_id: str | None = None,
) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        payload = client.list_rollout_devices(rollout_id.strip(), org_id=org_id, status=status, page=page, limit=limit)
        return _dumps({"rollout_id": rollout_id, "devices": payload})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Control a rollout: pause, resume, cancel or rollback (POST /rollouts/{id}/{action}, optional reason ≤ "
        "500 chars). rollback reverts only admitted devices. Returns the signalled acknowledgement (202), not "
        "the new steady state — confirm with get_rollout or watch_rollout. Explicit request only."
    ),
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False, readOnlyHint=False),
)
def rollout_control(
    rollout_id: str, action: str, reason: str | None = None, organization_id: str | None = None
) -> str:
    try:
        normalised = (action or "").strip().lower()
        if normalised not in AdmiralClient.ROLLOUT_ACTIONS:
            return _dumps({"error": f"Unsupported action {action!r}", "choices": list(AdmiralClient.ROLLOUT_ACTIONS)})
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        result = client.rollout_control(rollout_id.strip(), normalised, org_id=org_id, reason=reason)
        return _dumps({"rollout_id": rollout_id, "action": normalised, "result": result, "verify_with": "get_rollout"})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Watch a rollout live (SSE GET /rollouts/{id}/stream) until it reaches a terminal status (completed, "
        "failed, cancelled, rolled_back) or timeout_s elapses (default 120, max 600). Returns rollout status "
        "transitions with counts (incl. paused + pausedReason), per-device phase transitions, the last progress "
        "per device, and final counts. Returns immediately if the rollout is already terminal."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def watch_rollout(rollout_id: str, timeout_s: float = 120, organization_id: str | None = None) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        rid = rollout_id.strip()
        current = client.get_rollout(rid, org_id=org_id)
        current = current if isinstance(current, dict) else {}
        watch = RolloutWatch(initial_status=current.get("status"), initial_paused_reason=current.get("pausedReason"))
        watch.counts = current.get("counts")
        if watch.terminal:
            return _dumps({"rollout_id": rid, "stream": {"ended": "already_terminal", "elapsed_s": 0, "events": 0}, **watch.result()})
        timeout = min(max(float(timeout_s), 1.0), _MAX_WATCH_S)
        stream = client.stream_events(
            f"rollouts/{rid}/stream",
            lambda name, data: watch.add(name, data),
            org_id=org_id,
            timeout_s=timeout,
        )
        return _dumps({"rollout_id": rid, "stream": stream, **watch.result()})
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Operations: configurations, fleets, devices and rollout planning
# ---------------------------------------------------------------------------

_DELIVERY_NOTE = (
    "Changing a configuration version or a fleet's assignment does not push anything to devices. A device pulls its "
    "desired state every time it (re)connects and also receives pushes from rollouts and fleet moves, so connected "
    "devices keep running what they have until one of those happens; a fleet that follows 'latest' therefore picks "
    "the change up unevenly, device by device, with no canary and no health gate. create_rollout is the controlled "
    "way to apply a version (canary, in-flight window, failure budget, convergence check); render_device_document "
    "with push=true pushes the current desired state to one device immediately."
)
_ACTIVE_ROLLOUT_STATUSES = "pending,scheduled,in_progress,paused"
_CONFIG_STATUSES = ("draft", "active", "testing", "deprecated", "archived")  # stored lower-case (swagger says capitalised)
_ROLLOUT_TYPES = ("config", "reboot", "restart_workload", "system_update")
_STRATEGY_DEFAULTS = {
    "canary": 1,
    "maxInFlight": 50,
    "maxUnavailable": 2,
    "failureThreshold": 0.1,
    "progressDeadline": "30m",
}
_WINDOW_DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_OVERRIDE_KEYS = (
    "image",
    "command",
    "environment",
    "mounts",
    "options",
    "volumes",
    "ports",
    "image_cleaning",
    "registry_credential_id",
    "desiredState",
)


class NeedsChoice(Exception):
    """A name matched zero or several rows: the caller must pass a UUID."""

    def __init__(self, kind: str, ref: str, candidates: list[dict[str, Any]]):
        self.kind, self.ref, self.candidates = kind, ref, candidates
        if candidates:
            super().__init__(f"{kind.capitalize()} {ref!r} matched {len(candidates)}; pass a UUID.")
        else:
            super().__init__(f"No {kind} matches {ref!r}. Use list_{kind}s to find it, or pass a UUID.")


class ToolRefusal(Exception):
    """The request is valid JSON but should not be sent (no-op, guard failed). Payload goes back as-is."""

    def __init__(self, message: str, **extra: Any):
        super().__init__(message)
        self.payload = {"error": message, **extra}


def _ctx(organization_id: str | None) -> tuple[AdmiralClient, str]:
    client = get_client()
    return client, DeviceResolver(client).resolve_org(organization_id)


def _paged_items(
    client: AdmiralClient,
    path: str,
    org_id: str,
    *,
    params: dict[str, Any] | None = None,
    limit: int = 200,
    max_pages: int = 25,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        payload = client.get(path, org_id=org_id, params={**(params or {}), "page": page, "limit": limit})
        rows = _items(payload)
        out.extend(rows)
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        if page >= ((pagination or {}).get("totalPages") or 1) or not rows:
            break
        page += 1
    return out


def _unique_by_name(rows: list[dict[str, Any]], ref: str, kind: str) -> dict[str, Any]:
    needle = ref.strip().lower()
    hits = [r for r in rows if str(r.get("name") or "").lower() == needle] or [
        r for r in rows if needle in str(r.get("name") or "").lower()
    ]
    if len(hits) == 1:
        return hits[0]
    raise NeedsChoice(kind, ref, [{"id": r.get("id"), "name": r.get("name")} for r in hits[:10]])


def _find_fleet(client: AdmiralClient, org_id: str, ref: str) -> dict[str, Any]:
    """Fleet row by UUID (GET) or by name (list search). Several or no matches -> NeedsChoice."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("fleet is required (name or UUID)")
    if is_uuid(ref):
        row = client.get(f"fleets/{ref}", org_id=org_id)
        row = row if isinstance(row, dict) else {}
        row.setdefault("id", ref)
        return row
    return _unique_by_name(_items(client.list_fleets(org_id=org_id, search=ref, limit=200)), ref, "fleet")


def _find_config(client: AdmiralClient, org_id: str, ref: str) -> dict[str, Any]:
    """Configuration row (id, name, status, latest_version) by UUID (GET) or by name (list search)."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("configuration is required (name or UUID)")
    if is_uuid(ref):
        row = client.get(f"configurations/{ref}", org_id=org_id)
        row = row if isinstance(row, dict) else {}
        row.setdefault("id", ref)
        return row
    return _unique_by_name(_items(client.list_configurations(org_id=org_id, search=ref, limit=100)), ref, "configuration")


def _ref(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row.get("id"), "name": row.get("name")}


def _latest_of(row: dict[str, Any]) -> int | None:
    value = row.get("latest_version") or row.get("latestVersion")
    return int(value) if value else None


def _safe(fn: Any, warnings: list[str], label: str) -> Any:
    try:
        return fn()
    except Exception as exc:  # a failing side lookup must not hide the main answer
        warnings.append(f"{label}: {exc}")
        return None


def _fleet_usage(fleets: list[dict[str, Any]], config_id: str) -> dict[str, list[dict[str, Any]]]:
    """Fleets assigned to config_id, split into 'follow latest' and 'pinned'."""
    follow, pinned = [], []
    for fleet in fleets:
        if str(fleet.get("configuration_id") or "").lower() != str(config_id).lower():
            continue
        entry = {"id": fleet.get("id"), "name": fleet.get("name"), "devices": fleet.get("devices"), "online": fleet.get("online")}
        if fleet.get("config_version"):
            pinned.append({**entry, "pinned_version": fleet.get("config_version")})
        else:
            follow.append(entry)
    return {"follow_latest": follow, "pinned": pinned}


def _usage_by_config(fleets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for fleet in fleets:
        cid = str(fleet.get("configuration_id") or "").lower()
        if cid:
            out.setdefault(cid, []).append(
                {"id": fleet.get("id"), "name": fleet.get("name"), "version": fleet.get("config_version") or "latest"}
            )
    return out


def _assignment(payload: Any) -> dict[str, Any]:
    """Compact fleet configuration state from GET /fleets/{id}/configuration."""
    payload = payload if isinstance(payload, dict) else {}
    cid = payload.get("configuration_id")
    if not cid:
        return {"configuration": None, "assignment": None, "resolved_version": None}
    cfg = payload.get("configuration") or {}
    resolved = payload.get("config_version")
    return {
        "configuration": {"id": cid, "name": cfg.get("name"), "status": cfg.get("status"), "latest_version": cfg.get("latest_version")},
        "assignment": resolved if payload.get("is_pinned") else "latest",
        "resolved_version": resolved,
    }


def _fleet_assignment(client: AdmiralClient, org_id: str, fleet_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = client.get(f"fleets/{fleet_id}/configuration", org_id=org_id)
    payload = payload if isinstance(payload, dict) else {}
    return _assignment(payload), _norm_or_empty(payload.get("configuration_spec"))


def _norm_or_empty(spec: Any) -> dict[str, Any]:
    return ops.norm_spec(spec) if isinstance(spec, dict) and spec else {}


def _rollout_row(r: dict[str, Any]) -> dict[str, Any]:
    names = r.get("fleet_names") if isinstance(r.get("fleet_names"), dict) else {}
    fleets = [names.get(str(fid)) or fid for fid in (r.get("fleet_ids") or [])]
    row = {
        "id": r.get("id"),
        "name": r.get("name"),
        "type": r.get("type"),
        "status": r.get("status"),
        "progress_pct": r.get("progress_pct"),
        "fleets": fleets,
        "configuration": r.get("config_name") or r.get("config"),
        "version": r.get("version"),
        "disruptive": r.get("is_disruptive"),
        "paused_reason": r.get("pausedReason"),
        "counts": r.get("counts"),
        "created_at": r.get("created_at"),
    }
    return {k: v for k, v in row.items() if v not in (None, "", [], {})}


def _active_rollouts(client: AdmiralClient, org_id: str, fleet_ids: list[str]) -> list[dict[str, Any]]:
    payload = client.get(
        "rollouts",
        org_id=org_id,
        params={"status": _ACTIVE_ROLLOUT_STATUSES, "fleet_id": ",".join(fleet_ids), "limit": 50},
    )
    return [_rollout_row(r) for r in _items(payload)]


def _version_info(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "version": row.get("version_number"),
        "changed_by": row.get("changed_by"),
        "change_reason": row.get("change_reason"),
        "created_at": row.get("created_at"),
    }
    if row.get("is_rollback"):
        out["rollback"] = {"from": row.get("rolled_back_from"), "to": row.get("rolled_back_to")}
    return {k: v for k, v in out.items() if v not in (None, "")}


def _config_next(config: dict[str, Any], version: int | None, usage: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    fleets = [f["id"] for f in usage["follow_latest"] + usage["pinned"]]
    if fleets:
        return [
            {
                "tool": "preview_rollout",
                "why": "Check the impact and warnings, then roll the new version out under a canary.",
                "arguments": {"fleets": fleets, "configuration": config.get("id"), "version": version or "latest"},
            }
        ]
    return [
        {
            "tool": "assign_fleet_configuration",
            "why": "No fleet uses this configuration yet.",
            "arguments": {"fleet": "<fleet name or UUID>", "configuration": config.get("id"), "version": version or "latest"},
        }
    ]


def _delivery_warnings(usage: dict[str, list[dict[str, Any]]], new_version: int | None) -> list[str]:
    warnings = []
    if usage["follow_latest"]:
        names = ", ".join(f["name"] or f["id"] for f in usage["follow_latest"][:5])
        warnings.append(
            f"Fleets following 'latest' ({names}) resolve v{new_version} as soon as the version exists, without a canary. "
            "Devices pick it up when they reconnect or next receive a push."
        )
    return warnings


# ------------------------------------------------------- configurations ---


@mcp.tool(
    description=(
        "List configurations (GET /configurations): id, name, status (draft, active, testing, deprecated, archived; matched case-insensitively), "
        "latest_version and which fleets use each one (fleet name + 'latest' or the pinned version). Filter by search "
        "(name/description) and status. Read-only. Next: get_configuration for the spec, edit_configuration to change it."
    ),
    annotations=_READ_ONLY,
)
def list_configurations(
    search: str | None = None,
    status: str | None = None,
    limit: int = 50,
    page: int = 1,
    include_fleets: bool = True,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        if status:
            match = [s for s in _CONFIG_STATUSES if s.lower() == status.strip().lower()]
            if not match:
                return _dumps({"error": f"Unknown status {status!r}", "choices": list(_CONFIG_STATUSES)})
            status = match[0]
        payload = client.get(
            "configurations",
            org_id=org_id,
            params={"search": search, "status": status, "limit": max(1, min(int(limit), 200)), "page": max(1, int(page))},
        )
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        usage = _usage_by_config(_paged_items(client, "fleets", org_id)) if include_fleets else {}
        rows = []
        for cfg in _items(payload):
            row = {
                "id": cfg.get("id"),
                "name": cfg.get("name"),
                "status": cfg.get("status"),
                "latest_version": cfg.get("latest_version"),
                "description": cfg.get("description"),
                "tags": cfg.get("tags"),
                "updated_at": cfg.get("updated_at"),
            }
            if include_fleets:
                row["fleets"] = usage.get(str(cfg.get("id")).lower(), [])
            fleets_col = row.pop("fleets", None)
            row = {k: v for k, v in row.items() if v not in (None, "", [])}
            if include_fleets:
                row["fleets"] = fleets_col
            rows.append(row)
        total = (pagination or {}).get("total", len(rows))
        return _dumps(
            {
                "summary": f"{len(rows)} of {total} configuration(s)" + (f" matching {search!r}" if search else ""),
                "organization_id": org_id,
                "configurations": rows,
                "pagination": pagination,
                "next": [{"tool": "get_configuration", "why": "Read the spec and version history of one configuration."}],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "One configuration in full: metadata, the spec of the latest (or the requested) version, recent version history "
        "and the fleets assigned to it (GET /configurations/{id}/details, /versions, /versions/{n}). configuration = name "
        "or UUID (ambiguous names return candidates). Environment values whose names look like credentials are masked "
        "unless show_secret_env=true. Read-only."
    ),
    annotations=_READ_ONLY,
)
def get_configuration(
    configuration: str,
    version: int | None = None,
    include_spec: bool = True,
    show_secret_env: bool = False,
    history_limit: int = 10,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        cid = cfg["id"]
        warnings: list[str] = []
        if version is not None:
            if int(version) < 1:
                return _dumps({"error": "version must be 1 or greater (omit for the latest)"})
            main = lambda: client.get(f"configurations/{cid}/versions/{int(version)}", org_id=org_id)  # noqa: E731
        else:
            main = lambda: client.get(f"configurations/{cid}/details", org_id=org_id)  # noqa: E731
        detail, versions, fleets = _parallel_map(
            lambda fn: fn(),
            [
                main,
                lambda: _safe(lambda: client.get(f"configurations/{cid}/versions", org_id=org_id), warnings, "version history"),
                lambda: _safe(lambda: _paged_items(client, "fleets", org_id), warnings, "fleets"),
            ],
        )
        detail = detail if isinstance(detail, dict) else {}
        if version is not None:
            meta, spec, info = cfg, detail.get("spec"), detail
        else:
            meta = detail.get("configuration") or cfg
            spec, info = detail.get("latest_spec"), detail.get("version_info") or {}
        latest = _latest_of(meta) or _latest_of(cfg)
        shown = int(version) if version is not None else latest
        history = sorted(
            [_version_info(v) for v in (versions or []) if isinstance(v, dict)], key=lambda v: v.get("version", 0), reverse=True
        )[: max(1, int(history_limit))]
        usage = _fleet_usage(fleets or [], cid)
        out: dict[str, Any] = {
            "summary": (
                f"{meta.get('name')} ({meta.get('status')}): latest v{latest}, showing v{shown}; "
                f"{len(usage['follow_latest'])} fleet(s) follow latest, {len(usage['pinned'])} pinned."
            ),
            "configuration": {
                "id": cid,
                "name": meta.get("name"),
                "description": meta.get("description"),
                "tags": meta.get("tags"),
                "status": meta.get("status"),
                "latest_version": latest,
                "updated_at": meta.get("updated_at"),
            },
            "version": shown,
            "is_latest": shown == latest,
            "version_info": _version_info(info) if info else None,
            "history": history,
            "fleets": usage,
        }
        if include_spec:
            norm = ops.norm_spec(spec)
            out["spec"] = norm if show_secret_env else ops.mask_spec(norm)
            if not show_secret_env and ops.mask_spec(norm) != norm:
                out["note"] = "Credential-looking environment values are masked; pass show_secret_env=true to reveal them."
        else:
            out["spec_summary"] = ops.spec_summary(spec)
        if warnings:
            out["warnings"] = warnings
        out["next"] = [
            {"tool": "diff_configuration_versions", "why": "See what changed between versions."},
            {"tool": "edit_configuration", "why": "Change the image, environment or any spec field (creates a new version)."},
        ]
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Structured diff between two versions of a configuration spec (GET /configurations/{id}/versions/{n}). Defaults: "
        "to_version = latest, from_version = the one before. Returns one row per changed field (image, environment.NAME, "
        "ports[tcp/80], volumes.NAME, options.*, mounts[...], command, ...) with old/new values; credential-looking "
        "environment values are masked. Read-only."
    ),
    annotations=_READ_ONLY,
)
def diff_configuration_versions(
    configuration: str,
    from_version: int | None = None,
    to_version: int | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        latest = _latest_of(cfg)
        target = int(to_version) if to_version is not None else latest
        if not target or target < 1:
            return _dumps({"error": "Could not determine the version to compare; pass to_version.", "configuration": _ref(cfg)})
        source = int(from_version) if from_version is not None else target - 1
        if source < 1:
            return _dumps({"error": f"{cfg.get('name')} only has version {target}; there is nothing earlier to compare with.", "configuration": _ref(cfg)})
        if source == target:
            return _dumps({"error": "from_version and to_version are the same.", "configuration": _ref(cfg)})
        old, new = _parallel_map(
            lambda v: client.get(f"configurations/{cfg['id']}/versions/{v}", org_id=org_id), [source, target]
        )
        old, new = (old if isinstance(old, dict) else {}), (new if isinstance(new, dict) else {})
        rows = ops.spec_diff(old.get("spec"), new.get("spec"))
        return _dumps(
            {
                "summary": f"{cfg.get('name')} v{source} → v{target}: {ops.diff_summary(rows)}",
                "configuration": _ref(cfg),
                "from": _version_info(old) or {"version": source},
                "to": _version_info(new) or {"version": target},
                "identical": not rows,
                "changes": rows,
            }
        )
    except Exception as exc:
        return _err(exc)


def _spec_from_args(
    image: str | None,
    env: dict[str, Any] | None,
    ports: list[Any] | None,
    command: list[str] | None,
    desired_state: str | None,
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    if spec is not None:
        if any((image, env, ports, command, desired_state)):
            raise ValueError("Pass either spec (a full spec) or the individual fields, not both.")
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object.")
        built = dict(spec)
    else:
        built = {}
        if image:
            built["image"] = image
        if env:
            built["environment"] = {str(k): "" if v is None else str(v) for k, v in env.items()}
        if command:
            built["command"] = list(command)
        if desired_state:
            built["desiredState"] = desired_state
        if ports:
            parsed = []
            for p in ports:
                if isinstance(p, dict):
                    parsed.append({"port": int(p["port"]), "protocol": p.get("protocol", "tcp")})
                    continue
                num, _, proto = str(p).partition("/")
                parsed.append({"port": int(num), "protocol": (proto or "tcp").lower()})
            built["ports"] = parsed
    built = ops.norm_spec(built)
    if not built.get("image"):
        raise ValueError("A configuration needs an image (pass image=... or spec={'image': ...}).")
    if built["desiredState"] not in ("RUNNING", "STOPPED"):
        raise ValueError("desired_state must be RUNNING or STOPPED.")
    return built


@mcp.tool(
    description=(
        "Create a configuration (POST /configurations) at version 1. Give a name plus either image (with optional env "
        "{NAME: value}, ports ['8080/tcp', 80], command [...], desired_state RUNNING|STOPPED) or a full spec object. "
        "Does not assign it to any fleet: use assign_fleet_configuration afterwards. 409 means the name already exists. "
        "Creates a resource, explicit request only. Do not put real secrets in env: use the dashboard's secret files."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
def create_configuration(
    name: str,
    image: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    env: dict[str, Any] | None = None,
    ports: list[Any] | None = None,
    command: list[str] | None = None,
    desired_state: str | None = None,
    spec: dict[str, Any] | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        if not name or not name.strip():
            return _dumps({"error": "name is required"})
        built = _spec_from_args(image, env, ports, command, desired_state, spec)
        client, org_id = _ctx(organization_id)
        body: dict[str, Any] = {"name": name.strip(), "spec": built}
        if description:
            body["description"] = description
        if tags:
            body["tags"] = list(tags)
        created = client.post("configurations", org_id=org_id, json=body)
        created = created if isinstance(created, dict) else {}
        cid = created.get("id")
        back = client.get(f"configurations/{cid}/details", org_id=org_id) if cid else {}
        back = back if isinstance(back, dict) else {}
        meta = back.get("configuration") or created
        return _dumps(
            {
                "summary": f"Created configuration {meta.get('name')} (v{_latest_of(meta) or 1}, {meta.get('status')}) with image {built['image']}.",
                "configuration": {**_ref(meta), "status": meta.get("status"), "latest_version": _latest_of(meta)},
                "spec": ops.mask_spec(ops.norm_spec(back.get("latest_spec") or built)),
                "confirmed": bool(back),
                "next": [
                    {
                        "tool": "assign_fleet_configuration",
                        "why": "Nothing runs this configuration until a fleet is assigned to it.",
                        "arguments": {"fleet": "<fleet name or UUID>", "configuration": cid, "version": "latest"},
                    }
                ],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Edit a configuration: starts from the latest spec, applies the edits and saves a NEW version (PUT "
        "/configurations/{id}/spec). Edits: image (full reference) or image_tag (swap the tag only), env_set {NAME: value}, "
        "env_unset [NAME], merge_patch (RFC 7386 JSON merge patch over the spec, null deletes a key, lists replace), or spec "
        "(whole replacement, exclusive with the others). change_reason is required to write. base_version refuses the edit "
        "if the latest version is no longer that number (someone else edited). dry_run=true returns the diff and the "
        "effect without writing. No-op edits are refused. The result lists old → new version, the diff, which fleets follow "
        "'latest' (they resolve the new version immediately but devices only receive it when they reconnect or get a push; "
        "no canary) versus pinned (unaffected until a rollout or assignment), and the next step. A 402 means the "
        "organisation lacks an entitlement used by the spec (for example an image signature policy). Changes what "
        "devices will run — explicit request only; prefer dry_run first."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
)
def edit_configuration(
    configuration: str,
    change_reason: str | None = None,
    image: str | None = None,
    image_tag: str | None = None,
    env_set: dict[str, Any] | None = None,
    env_unset: list[str] | None = None,
    merge_patch: dict[str, Any] | None = None,
    spec: dict[str, Any] | None = None,
    base_version: int | None = None,
    dry_run: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        reason = (change_reason or "").strip()
        if not dry_run and not reason:
            return _dumps({"error": "change_reason is required (it is recorded in the version history). Use dry_run=true to preview without one."})
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        cid = cfg["id"]
        detail = client.get(f"configurations/{cid}/details", org_id=org_id)
        detail = detail if isinstance(detail, dict) else {}
        meta = detail.get("configuration") or cfg
        latest = _latest_of(meta)
        if base_version is not None and int(base_version) != latest:
            raise ToolRefusal(
                f"{meta.get('name')} is at v{latest}, not v{int(base_version)}: someone changed it since you read it. "
                "Nothing was written. Re-read with get_configuration (or diff_configuration_versions) and retry.",
                latest_version=latest,
                base_version=int(base_version),
            )
        current = ops.norm_spec(detail.get("latest_spec"))
        new_spec, notes = ops.apply_spec_edits(
            current, image=image, image_tag=image_tag, env_set=env_set, env_unset=env_unset, patch=merge_patch, replace=spec
        )
        rows = ops.spec_diff(current, new_spec)
        if not rows:
            raise ToolRefusal(
                "No change: the edit produces the same spec as the latest version, so no version was created.",
                latest_version=latest,
                notes=notes,
            )
        usage = _fleet_usage(_paged_items(client, "fleets", org_id), cid)
        new_version = (latest or 0) + 1
        warnings = _delivery_warnings(usage, new_version)
        if dry_run and usage["follow_latest"]:
            warnings.append(
                "To stage this safely, pin those fleets to the current version first (assign_fleet_configuration with "
                f"version={latest}), save, then apply with create_rollout."
            )
        result: dict[str, Any] = {
            "configuration": {**_ref(meta), "status": meta.get("status")},
            "old_version": latest,
            "diff": rows,
            "fleets": usage,
            "notes": notes,
            "warnings": warnings,
        }
        if dry_run:
            result.update(
                {
                    "summary": f"DRY RUN {meta.get('name')} v{latest} → v{new_version}: {ops.diff_summary(rows)}. Nothing was written.",
                    "dry_run": True,
                    "new_version": new_version,
                    "next": [{"tool": "edit_configuration", "why": "Repeat without dry_run to save it.", "arguments": {"configuration": cid, "base_version": latest}}],
                }
            )
            return _dumps({k: v for k, v in result.items() if v not in ([], None)} | {"dry_run": True})
        client.put(f"configurations/{cid}/spec", org_id=org_id, json={"spec": new_spec, "change_reason": reason})
        back = client.get(f"configurations/{cid}/details", org_id=org_id)
        back = back if isinstance(back, dict) else {}
        back_meta = back.get("configuration") or {}
        saved_version = _latest_of(back_meta)
        drift = ops.spec_diff(new_spec, back.get("latest_spec"), mask=True)
        confirmed = saved_version == new_version and not drift
        if saved_version is not None and saved_version != new_version:
            warnings.append(
                f"Expected v{new_version} but the configuration is now at v{saved_version}: another edit landed at the same time. Check diff_configuration_versions."
            )
        if drift:
            warnings.append("The stored spec differs from what was sent (server-side normalisation or a concurrent edit): " + ops.diff_summary(drift))
        result.update(
            {
                "summary": f"{meta.get('name')} v{latest} → v{saved_version}: {ops.diff_summary(rows)}.",
                "new_version": saved_version,
                "change_reason": reason,
                "read_back": {"latest_version": saved_version, "confirmed": confirmed},
                "delivery": _DELIVERY_NOTE if (usage["follow_latest"] or usage["pinned"]) else None,
                "warnings": warnings,
                "next": _config_next(meta, saved_version, usage),
            }
        )
        return _dumps({k: v for k, v in result.items() if v not in ([], None, "")})
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Change a configuration's metadata and/or lifecycle status: name, description, tags (replaces the list; [] clears) "
        "via PUT /configurations/{id}, status (draft, active, testing, deprecated, archived; matched case-insensitively) via PUT "
        "/configurations/{id}/status. Does not touch the spec or any device. The backend ignores empty name/description, "
        "so those cannot be cleared. Returns before/after read back from the API. Explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True),
)
def update_configuration_metadata(
    configuration: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    status: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        for label, value in (("name", name), ("description", description)):
            if value is not None and not value.strip():
                return _dumps({"error": f"{label} cannot be empty (the backend ignores empty values)."})
        canonical = None
        if status is not None:
            canonical = next((s for s in _CONFIG_STATUSES if s.lower() == status.strip().lower()), None)
            if canonical is None:
                return _dumps({"error": f"Unknown status {status!r}", "choices": list(_CONFIG_STATUSES)})
        if name is None and description is None and tags is None and canonical is None:
            return _dumps({"error": "Nothing to change: pass name, description, tags or status."})
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        cid = cfg["id"]
        before = client.get(f"configurations/{cid}", org_id=org_id)
        before = before if isinstance(before, dict) else {}
        body: dict[str, Any] = {}
        if name is not None and name.strip() != before.get("name"):
            body["name"] = name.strip()
        if description is not None and description.strip() != (before.get("description") or ""):
            body["description"] = description.strip()
        if tags is not None and list(tags) != list(before.get("tags") or []):
            body["tags"] = list(tags)
        status_change = canonical is not None and canonical != before.get("status")
        if not body and not status_change:
            raise ToolRefusal("No change: the configuration already has these values.", configuration=_ref(before))
        if body:
            client.put(f"configurations/{cid}", org_id=org_id, json=body)
        if status_change:
            client.put(f"configurations/{cid}/status", org_id=org_id, json={"status": canonical})
        after = client.get(f"configurations/{cid}", org_id=org_id)
        after = after if isinstance(after, dict) else {}
        fields = [k for k in ("name", "description", "tags", "status") if before.get(k) != after.get(k)]
        return _dumps(
            {
                "summary": f"Updated {after.get('name')}: " + (", ".join(f"{k} {before.get(k)!r} → {after.get(k)!r}" for k in fields) or "no visible change"),
                "configuration": _ref(after),
                "before": {k: before.get(k) for k in ("name", "description", "tags", "status")},
                "after": {k: after.get(k) for k in ("name", "description", "tags", "status")},
                "confirmed": all(
                    (list(after.get(k) or []) if k == "tags" else after.get(k)) == v
                    for k, v in {**body, **({"status": canonical} if status_change else {})}.items()
                ),
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Roll a configuration back to an earlier version (POST /configurations/{id}/rollback). This does not rewrite "
        "history: it creates a NEW latest version that is a copy of target_version, so fleets following 'latest' resolve "
        "it immediately (devices receive it when they reconnect or get a push) and pinned fleets are unaffected until a "
        "rollout. reason is recorded. dry_run=true shows the diff only. Returns old → new version and the diff, read back. "
        "Changes what devices will run — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
)
def rollback_configuration(
    configuration: str,
    target_version: int,
    reason: str | None = None,
    dry_run: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        cid = cfg["id"]
        latest = _latest_of(cfg)
        target = int(target_version)
        if target < 1 or (latest and target > latest):
            return _dumps({"error": f"target_version must be between 1 and {latest}.", "configuration": _ref(cfg)})
        if target == latest:
            raise ToolRefusal(f"v{target} is already the latest version of {cfg.get('name')}; there is nothing to roll back.", latest_version=latest)
        current, wanted = _parallel_map(lambda v: client.get(f"configurations/{cid}/versions/{v}", org_id=org_id), [latest, target])
        current, wanted = (current if isinstance(current, dict) else {}), (wanted if isinstance(wanted, dict) else {})
        rows = ops.spec_diff(current.get("spec"), wanted.get("spec"))
        usage = _fleet_usage(_paged_items(client, "fleets", org_id), cid)
        new_version = (latest or 0) + 1
        warnings = _delivery_warnings(usage, new_version)
        if not rows:
            warnings.append(f"v{target} has the same spec as v{latest}; the rollback creates an identical version.")
        base = {"configuration": _ref(cfg), "old_version": latest, "target_version": target, "diff": rows, "fleets": usage, "warnings": warnings}
        if dry_run:
            return _dumps(
                {
                    "summary": f"DRY RUN {cfg.get('name')}: v{latest} → copy of v{target} as v{new_version}: {ops.diff_summary(rows)}. Nothing was written.",
                    "dry_run": True,
                    **base,
                }
            )
        body: dict[str, Any] = {"target_version": target}
        if reason and reason.strip():
            body["reason"] = reason.strip()
        client.post(f"configurations/{cid}/rollback", org_id=org_id, json=body)
        back = client.get(f"configurations/{cid}/details", org_id=org_id)
        back = back if isinstance(back, dict) else {}
        saved = _latest_of(back.get("configuration") or {})
        drift = ops.spec_diff(wanted.get("spec"), back.get("latest_spec"))
        return _dumps(
            {
                "summary": f"{cfg.get('name')} rolled back: v{latest} → v{saved} (a copy of v{target}): {ops.diff_summary(rows)}.",
                **base,
                "new_version": saved,
                "read_back": {"latest_version": saved, "matches_target": not drift, "confirmed": saved == new_version and not drift},
                "delivery": _DELIVERY_NOTE if (usage["follow_latest"] or usage["pinned"]) else None,
                "next": _config_next(cfg, saved, usage),
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Delete a configuration and all its versions (DELETE /configurations/{id}). Irreversible. The backend would "
        "silently detach fleets that still use it, so this tool refuses while any fleet is assigned to the configuration or "
        "an unfinished rollout references it, and lists them; reassign those first (assign_fleet_configuration). Reads "
        "back that it is gone. Destructive — explicit request for a named configuration only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
)
def delete_configuration(configuration: str, organization_id: str | None = None) -> str:
    try:
        client, org_id = _ctx(organization_id)
        cfg = _find_config(client, org_id, configuration)
        cid = cfg["id"]
        fleets, rollouts = _parallel_map(
            lambda fn: fn(),
            [
                lambda: _paged_items(client, "fleets", org_id),
                lambda: _items(client.get("rollouts", org_id=org_id, params={"status": _ACTIVE_ROLLOUT_STATUSES, "limit": 200})),
            ],
        )
        usage = _fleet_usage(fleets, cid)
        using = usage["follow_latest"] + usage["pinned"]
        busy = [_rollout_row(r) for r in rollouts if str(r.get("config") or "").lower() == str(cid).lower()]
        if using or busy:
            raise ToolRefusal(
                f"{cfg.get('name')} is still in use ({len(using)} fleet(s), {len(busy)} unfinished rollout(s)); nothing was deleted.",
                configuration=_ref(cfg),
                fleets=using,
                rollouts=busy,
                next=[{"tool": "assign_fleet_configuration", "why": "Move each fleet to another configuration first."}],
            )
        client.delete(f"configurations/{cid}", org_id=org_id)
        gone = False
        try:
            client.get(f"configurations/{cid}", org_id=org_id)
        except AdmiralAPIError as exc:
            gone = exc.status == 404
        return _dumps(
            {
                "summary": f"Deleted configuration {cfg.get('name')}." if gone else f"Delete sent for {cfg.get('name')} but it still reads back; check list_configurations.",
                "configuration": _ref(cfg),
                "deleted": gone,
                "read_back": {"confirmed": gone},
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------- fleets ---


def _tag_rows(tags: Any) -> list[dict[str, str]]:
    return [{"key": t.get("key"), "value": t.get("value", "")} for t in (tags or []) if isinstance(t, dict)]


def _tags_arg(tags: dict[str, Any] | list[Any] | None) -> list[dict[str, str]]:
    """Tags given as {key: value} or [{key, value}] -> API rows. Values are strings."""
    if not tags:
        return []
    if isinstance(tags, dict):
        return [{"key": str(k).strip(), "value": "" if v is None else str(v)} for k, v in tags.items()]
    rows = []
    for t in tags:
        if isinstance(t, dict) and t.get("key"):
            rows.append({"key": str(t["key"]).strip(), "value": str(t.get("value", ""))})
        elif isinstance(t, str) and t.strip():
            key, _, value = t.partition("=")
            rows.append({"key": key.strip(), "value": value.strip()})
        else:
            raise ValueError(f"Cannot read tag {t!r}: use {{key: value}} or 'key=value'.")
    return rows


@mcp.tool(
    description=(
        "One fleet in full (GET /fleets/{id}, /configuration, /update-policy, /configuration/history, /rollouts): device "
        "counts and online, tags, the assigned configuration with how it is assigned ('latest' = follows the newest "
        "version, or a pinned version number) and the version that resolves to now, the OS update policy and update "
        "window, recent configuration assignments and any unfinished rollouts. fleet = name or UUID (ambiguous names "
        "return candidates). Read-only."
    ),
    annotations=_READ_ONLY,
)
def get_fleet(fleet: str, organization_id: str | None = None) -> str:
    try:
        client, org_id = _ctx(organization_id)
        found = _find_fleet(client, org_id, fleet)
        fid = found["id"]
        warnings: list[str] = []
        detail, assignment, policy, history, active = _parallel_map(
            lambda fn: fn(),
            [
                lambda: client.get(f"fleets/{fid}", org_id=org_id) if not is_uuid(fleet.strip()) else found,
                lambda: _safe(lambda: _fleet_assignment(client, org_id, fid)[0], warnings, "configuration"),
                lambda: _safe(lambda: client.get(f"fleets/{fid}/update-policy", org_id=org_id), warnings, "update policy"),
                lambda: _safe(lambda: client.get(f"fleets/{fid}/configuration/history", org_id=org_id), warnings, "configuration history"),
                lambda: _safe(lambda: _active_rollouts(client, org_id, [fid]), warnings, "rollouts"),
            ],
        )
        detail = detail if isinstance(detail, dict) else found
        assignment = assignment or {"configuration": None, "assignment": None, "resolved_version": None}
        cfg = assignment["configuration"]
        if cfg:
            how = "follows latest" if assignment["assignment"] == "latest" else f"pinned to v{assignment['assignment']}"
            cfg_text = f"{cfg.get('name')} {how} (v{assignment['resolved_version']})"
        else:
            cfg_text = "no configuration"
        history_rows = sorted(
            [h for h in (history or []) if isinstance(h, dict)], key=lambda h: str(h.get("assigned_at") or ""), reverse=True
        )[:5]
        out = {
            "summary": f"{detail.get('name')}: {detail.get('devices', 0)} device(s), {detail.get('online', 0)} online; {cfg_text}.",
            "fleet": {
                "id": fid,
                "name": detail.get("name"),
                "description": detail.get("description"),
                "location": detail.get("location"),
                "tags": _tag_rows(detail.get("tags")),
                "created_at": detail.get("createdAt"),
                "last_update": detail.get("lastUpdate"),
                "ssh_enabled": detail.get("sshEnabled"),
                "signature_policy": detail.get("signaturePolicy"),
            },
            "devices": {
                k: v
                for k, v in (
                    ("total", detail.get("devices")),
                    ("online", detail.get("online")),
                    ("offline", detail.get("offline")),
                    ("compliance", detail.get("compliance")),
                )
                if v is not None
            },
            "configuration": assignment,
            "update_policy": policy,
            "update_window": detail.get("update_window"),
            "recent_assignments": [
                {k: h.get(k) for k in ("configuration_id", "config_version", "assigned_by", "assigned_at", "reason") if h.get(k) is not None}
                for h in history_rows
            ],
            "active_rollouts": active or [],
            "warnings": warnings,
            "next": [
                {"tool": "list_devices", "why": "List the devices in this fleet.", "arguments": {"fleet_id": fid}},
                {"tool": "assign_fleet_configuration", "why": "Change which configuration the fleet runs (applied directly)."},
                {"tool": "preview_rollout", "why": "Plan a controlled change for this fleet."},
            ],
        }
        return _dumps({k: v for k, v in out.items() if v not in (None, [], {})})
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Assign a configuration to a fleet (POST /fleets/{id}/configuration). version = an integer (pin that version) or "
        "'latest' (follow the newest version; default). THIS IS APPLIED DIRECTLY: it changes the fleet's desired state at "
        "once, with no canary, no failure budget and no convergence check. Nothing is pushed to connected devices; each "
        "device picks the new desired state up when it reconnects or next receives a push, so a running fleet changes "
        "unevenly. For a fleet with live devices use create_rollout (preview_rollout first) instead. Returns before/after "
        "read back from the API. Refuses no-ops and unknown versions. Changes what devices run — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False),
)
def assign_fleet_configuration(
    fleet: str,
    configuration: str,
    version: int | str = "latest",
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        fl = _find_fleet(client, org_id, fleet)
        cfg = _find_config(client, org_id, configuration)
        fid, cid = fl["id"], cfg["id"]
        latest = _latest_of(cfg)
        wants_latest = str(version).strip().lower() == "latest"
        pin = None if wants_latest else int(version)
        if pin is not None and (pin < 1 or (latest and pin > latest)):
            return _dumps({"error": f"version must be between 1 and {latest} (or 'latest').", "configuration": _ref(cfg)})
        before, before_spec = _fleet_assignment(client, org_id, fid)
        before_cfg = before["configuration"]
        same = bool(before_cfg) and str(before_cfg["id"]).lower() == str(cid).lower() and (
            before["assignment"] == "latest" if wants_latest else before["assignment"] == pin
        )
        if same:
            raise ToolRefusal(
                f"No change: {fl.get('name')} is already assigned {cfg.get('name')} ({'latest' if wants_latest else 'v' + str(pin)}).",
                before=before,
            )
        warnings = [
            "Applied directly with no canary or health gate; connected devices pick it up when they reconnect or receive a push.",
        ]
        active = _safe(lambda: _active_rollouts(client, org_id, [fid]), warnings, "rollouts") or []
        if active:
            warnings.append(f"{len(active)} unfinished rollout(s) target this fleet; they may override or conflict with this assignment.")
        if fl.get("online"):
            warnings.append(f"{fl.get('online')} device(s) online now: use create_rollout for a staged change.")
        body: dict[str, Any] = {"configuration_id": cid}
        if pin is not None:
            body["version"] = pin
        client.post(f"fleets/{fid}/configuration", org_id=org_id, json=body)
        after, after_spec = _fleet_assignment(client, org_id, fid)
        after_cfg = after["configuration"] or {}
        confirmed = str(after_cfg.get("id", "")).lower() == str(cid).lower() and (
            after["assignment"] == "latest" if wants_latest else after["assignment"] == pin
        )
        rows = ops.spec_diff(before_spec, after_spec)
        text = lambda a: f"{(a['configuration'] or {}).get('name')} @ {a['assignment']}" if a["configuration"] else "none"  # noqa: E731
        return _dumps(
            {
                "summary": f"{fl.get('name')}: {text(before)} → {text(after)}" + (" (confirmed)." if confirmed else " (NOT confirmed by read-back)."),
                "fleet": _ref(fl),
                "before": before,
                "after": after,
                "diff": rows,
                "read_back": {"confirmed": confirmed},
                "warnings": warnings,
                "delivery": _DELIVERY_NOTE,
                "next": [{"tool": "get_fleet", "why": "Check the fleet's state."}],
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "History of configuration assignments for a fleet (GET /fleets/{id}/configuration/history): newest first, with the "
        "configuration name, version (blank = followed latest), who assigned it, when and why. Includes assignments made "
        "by completed rollouts. Read-only."
    ),
    annotations=_READ_ONLY,
)
def get_fleet_configuration_history(fleet: str, limit: int = 20, organization_id: str | None = None) -> str:
    try:
        client, org_id = _ctx(organization_id)
        fl = _find_fleet(client, org_id, fleet)
        history = client.get(f"fleets/{fl['id']}/configuration/history", org_id=org_id)
        rows = sorted(_items(history), key=lambda h: str(h.get("assigned_at") or ""), reverse=True)
        rows = rows[: max(1, int(limit))]
        names: dict[str, str] = {}
        wanted = {str(h.get("configuration_id")) for h in rows if h.get("configuration_id")}
        if wanted:
            try:
                for c in _paged_items(client, "configurations", org_id, limit=100, max_pages=10):
                    names[str(c.get("id"))] = c.get("name")
            except AdmiralAPIError:
                pass
        out = [
            {
                "configuration": {"id": h.get("configuration_id"), "name": names.get(str(h.get("configuration_id")))},
                "version": h.get("config_version") or "latest",
                "assigned_by": h.get("assigned_by"),
                "assigned_at": h.get("assigned_at"),
                "reason": h.get("reason"),
            }
            for h in rows
        ]
        return _dumps(
            {
                "summary": f"{fl.get('name')}: {len(out)} assignment(s)" + (f", latest: {out[0]['configuration']['name']} @ {out[0]['version']}" if out else ""),
                "fleet": _ref(fl),
                "history": out,
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Create a fleet (POST /fleets). name required; description, location, tags ({key: value} or ['key=value']) and an "
        "optional configuration (name or UUID; follows its latest version, applied directly with no rollout) are optional. "
        "Read back from the API. A new fleet has no devices: move devices in with move_device_to_fleet. Creates a "
        "resource — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
def create_fleet(
    name: str,
    description: str | None = None,
    location: str | None = None,
    tags: dict[str, Any] | list[Any] | None = None,
    configuration: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        if not name or not name.strip():
            return _dumps({"error": "name is required"})
        client, org_id = _ctx(organization_id)
        warnings: list[str] = []
        body: dict[str, Any] = {"name": name.strip()}
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        rows = _tags_arg(tags)
        if rows:
            body["tags"] = rows
        cfg = _find_config(client, org_id, configuration) if configuration else None
        if cfg:
            body["configuration_id"] = cfg["id"]
        same = [r for r in _items(client.list_fleets(org_id=org_id, search=name.strip(), limit=200)) if str(r.get("name") or "").lower() == name.strip().lower()]
        if same:
            warnings.append(f"{len(same)} fleet(s) already use this name; by-name lookups will need a UUID.")
        created = client.post("fleets", org_id=org_id, json=body)
        created = created if isinstance(created, dict) else {}
        fid = created.get("id")
        back = client.get(f"fleets/{fid}", org_id=org_id) if fid else {}
        back = back if isinstance(back, dict) else {}
        return _dumps(
            {
                "summary": f"Created fleet {back.get('name') or name.strip()}" + (f" with configuration {cfg.get('name')}" if cfg else " (no configuration)") + ".",
                "fleet": {
                    "id": fid,
                    "name": back.get("name"),
                    "description": back.get("description"),
                    "location": back.get("location"),
                    "tags": _tag_rows(back.get("tags")),
                    "configuration_id": back.get("configuration_id"),
                },
                "read_back": {"confirmed": bool(back)},
                "warnings": warnings,
                "next": [{"tool": "move_device_to_fleet", "why": "Add devices to the new fleet."}],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Update a fleet's name, description, location and tags. name/description/location go to PUT /fleets/{id} (empty "
        "values are ignored by the backend, so they cannot be cleared). Tags: tags_add {key: value} adds or updates keys "
        "(POST /fleets/{id}/tags), tags_remove [key] removes keys (DELETE), tags_replace {key: value} replaces the whole "
        "set (PUT; exclusive with add/remove; {} is refused, use tags_remove). Returns before/after read back. Fleet tags "
        "become effective tags on its devices. Explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
)
def update_fleet(
    fleet: str,
    name: str | None = None,
    description: str | None = None,
    location: str | None = None,
    tags_add: dict[str, Any] | list[Any] | None = None,
    tags_remove: list[str] | None = None,
    tags_replace: dict[str, Any] | list[Any] | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        for label, value in (("name", name), ("description", description), ("location", location)):
            if value is not None and not value.strip():
                return _dumps({"error": f"{label} cannot be empty (the backend ignores empty values)."})
        add_rows, replace_rows = _tags_arg(tags_add), _tags_arg(tags_replace)
        if tags_replace is not None and not replace_rows:
            return _dumps({"error": "tags_replace is empty; to remove tags pass tags_remove=[...]."})
        if tags_replace is not None and (add_rows or tags_remove):
            return _dumps({"error": "tags_replace cannot be combined with tags_add or tags_remove."})
        if name is None and description is None and location is None and not add_rows and not tags_remove and not replace_rows:
            return _dumps({"error": "Nothing to change: pass name, description, location or a tag edit."})
        client, org_id = _ctx(organization_id)
        fl = _find_fleet(client, org_id, fleet)
        fid = fl["id"]
        before = client.get(f"fleets/{fid}", org_id=org_id)
        before = before if isinstance(before, dict) else {}
        body = {
            k: v.strip()
            for k, v in (("name", name), ("description", description), ("location", location))
            if v is not None and v.strip() != (before.get(k) or "")
        }
        old_tags = {t["key"]: t["value"] for t in _tag_rows(before.get("tags"))}
        add_rows = [t for t in add_rows if old_tags.get(t["key"]) != t["value"]]
        remove = [k for k in (tags_remove or []) if k in old_tags]
        missing = [k for k in (tags_remove or []) if k not in old_tags]
        if replace_rows and {t["key"]: t["value"] for t in replace_rows} == old_tags:
            replace_rows = []
        if not body and not add_rows and not remove and not replace_rows:
            raise ToolRefusal("No change: the fleet already has these values.", fleet=_ref(before), ignored_tag_removals=missing)
        if body:
            client.put(f"fleets/{fid}", org_id=org_id, json=body)
        if replace_rows:
            client.put(f"fleets/{fid}/tags", org_id=org_id, json={"tags": replace_rows})
        if add_rows:
            client.post(f"fleets/{fid}/tags", org_id=org_id, json={"tags": add_rows})
        if remove:
            client.request("DELETE", f"fleets/{fid}/tags", org_id=org_id, params={"key": remove})
        after = client.get(f"fleets/{fid}", org_id=org_id)
        after = after if isinstance(after, dict) else {}
        new_tags = {t["key"]: t["value"] for t in _tag_rows(after.get("tags"))}
        confirmed = all(after.get(k) == v for k, v in body.items())
        confirmed = confirmed and all(new_tags.get(t["key"]) == t["value"] for t in add_rows + replace_rows)
        confirmed = confirmed and not any(k in new_tags for k in remove)
        if replace_rows:
            confirmed = confirmed and new_tags == {t["key"]: t["value"] for t in replace_rows}
        return _dumps(
            {
                "summary": f"Updated fleet {after.get('name')}: " + ", ".join(
                    [f"{k} → {v!r}" for k, v in body.items()]
                    + ([f"tags +{[t['key'] for t in add_rows]}"] if add_rows else [])
                    + ([f"tags -{remove}"] if remove else [])
                    + (["tags replaced"] if replace_rows else [])
                ),
                "fleet": _ref(after),
                "before": {"name": before.get("name"), "description": before.get("description"), "location": before.get("location"), "tags": old_tags},
                "after": {"name": after.get("name"), "description": after.get("description"), "location": after.get("location"), "tags": new_tags},
                "ignored_tag_removals": missing,
                "read_back": {"confirmed": confirmed},
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


def _clean_window(window: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(window, dict):
        raise ValueError("update_window must be an object {enabled, days, start_time, end_time, timezone}.")
    unknown = set(window) - {"enabled", "days", "start_time", "end_time", "timezone"}
    if unknown:
        raise ValueError(f"Unknown update_window field(s) {sorted(unknown)}; use enabled, days, start_time, end_time, timezone.")
    out = {
        "enabled": bool(window.get("enabled", True)),
        "days": [str(d).strip().capitalize() for d in window.get("days") or []],
        "start_time": str(window.get("start_time") or ""),
        "end_time": str(window.get("end_time") or ""),
        "timezone": str(window.get("timezone") or ""),
    }
    if out["enabled"]:
        if not out["days"] or any(d not in _WINDOW_DAYS for d in out["days"]):
            raise ValueError(f"days must be a non-empty list of {', '.join(_WINDOW_DAYS)}.")
        for key in ("start_time", "end_time"):
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", out[key]):
                raise ValueError(f"{key} must be HH:MM (24-hour).")
        if not out["timezone"]:
            raise ValueError("timezone is required (IANA name, e.g. Australia/Sydney or UTC).")
    return out


def _policy_view(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    return {"mode": payload.get("mode"), "targets": payload.get("targets") or [], "update_window": payload.get("update_window")}


@mcp.tool(
    description=(
        "Set a fleet's OS update policy and/or update window. This governs Admiral system/OS image updates, not the "
        "workload configuration. mode 'latest' tracks the current release; 'pinned' uses targets "
        "[{architecture, board?, system_version_id?, initram_version_id?}] (a missing version id follows latest for that "
        "hardware). update_window {enabled, days:[Monday..], start_time:'02:00', end_time:'06:00', timezone:'Australia/Sydney'} "
        "limits when devices apply updates; disable_update_window=true turns it off. Policy via PUT "
        "/fleets/{id}/update-policy (replaces the whole policy; the existing targets are kept when you only change the "
        "window), window alone via PUT /fleets/{id}/update-window. Returns before/after read back. Changes what OS "
        "versions devices install — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
def set_fleet_update_policy(
    fleet: str,
    mode: str | None = None,
    targets: list[dict[str, Any]] | None = None,
    update_window: dict[str, Any] | None = None,
    disable_update_window: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        if update_window is not None and disable_update_window:
            return _dumps({"error": "Pass update_window or disable_update_window, not both."})
        if mode is None and targets is None and update_window is None and not disable_update_window:
            return _dumps({"error": "Nothing to change: pass mode, targets, update_window or disable_update_window."})
        want_mode = None
        if mode is not None:
            want_mode = mode.strip().lower()
            if want_mode not in ("latest", "pinned"):
                return _dumps({"error": f"Unknown mode {mode!r}", "choices": ["latest", "pinned"]})
        if want_mode == "latest" and targets:
            return _dumps({"error": "targets only apply to mode 'pinned'."})
        window = _clean_window(update_window) if update_window is not None else None
        client, org_id = _ctx(organization_id)
        fl = _find_fleet(client, org_id, fleet)
        fid = fl["id"]
        before = _policy_view(client.get(f"fleets/{fid}/update-policy", org_id=org_id))
        if disable_update_window:
            old = before["update_window"] or {}
            window = {
                "enabled": False,
                "days": old.get("days") or [],
                "start_time": old.get("start_time") or "",
                "end_time": old.get("end_time") or "",
                "timezone": old.get("timezone") or "",
            }
        policy_change = want_mode is not None or targets is not None
        if policy_change:
            new_mode = want_mode or ("pinned" if targets is not None else before["mode"] or "latest")
            new_targets = targets if targets is not None else (before["targets"] if new_mode == "pinned" else [])
            if new_mode == "pinned" and not new_targets:
                return _dumps({"error": "mode 'pinned' needs targets [{architecture, board?, system_version_id?, initram_version_id?}]."})
            for i, t in enumerate(new_targets):
                if not isinstance(t, dict) or not t.get("architecture"):
                    return _dumps({"error": f"targets[{i}] needs an architecture."})
            if new_mode == before["mode"] and new_targets == before["targets"] and window in (None, before["update_window"]):
                raise ToolRefusal("No change: the fleet already has this update policy.", before=before)
            body: dict[str, Any] = {"mode": new_mode}
            if new_targets:
                body["targets"] = new_targets
            if window is not None:
                body["update_window"] = window
            client.put(f"fleets/{fid}/update-policy", org_id=org_id, json=body)
        else:
            if window == before["update_window"]:
                raise ToolRefusal("No change: the fleet already has this update window.", before=before)
            client.put(f"fleets/{fid}/update-window", org_id=org_id, json=window)
        after = _policy_view(client.get(f"fleets/{fid}/update-policy", org_id=org_id))
        confirmed = (not policy_change or (after["mode"] == body["mode"])) and (window is None or after["update_window"] == window)
        return _dumps(
            {
                "summary": f"{fl.get('name')} update policy: mode {before['mode']} → {after['mode']}"
                + (f"; window {'on' if (after['update_window'] or {}).get('enabled') else 'off'}" if window is not None else "")
                + ("" if confirmed else " (NOT confirmed by read-back)")
                + ".",
                "fleet": _ref(fl),
                "before": before,
                "after": after,
                "read_back": {"confirmed": confirmed},
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


# --------------------------------------------------------------- devices ---


def _device_detail(client: AdmiralClient, device_id: str, org_id: str) -> dict[str, Any]:
    detail = client.get_device(device_id, org_id=org_id)
    return detail if isinstance(detail, dict) else {}


def _device_tags(detail: dict[str, Any]) -> dict[str, str]:
    return {t["key"]: t["value"] for t in _tag_rows(detail.get("tags"))}


@mcp.tool(
    description=(
        "Update a device's display name, notes, location (latitude + longitude) and tags (PUT /devices/{id}). tags "
        "replaces the whole device tag set; tags_add {key: value} and tags_remove [key] edit it incrementally (cannot be "
        "combined with tags). device = name, UUID, tag or other query (ambiguous matches return candidates and send "
        "nothing). Returns before/after read back. Does not change what the device runs. Explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True),
)
def update_device(
    device: str,
    name: str | None = None,
    notes: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    tags: dict[str, Any] | list[Any] | None = None,
    tags_add: dict[str, Any] | list[Any] | None = None,
    tags_remove: list[str] | None = None,
    screenshots_disabled: bool | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        if name is not None and not name.strip():
            return _dumps({"error": "name cannot be empty."})
        if (latitude is None) != (longitude is None):
            return _dumps({"error": "Pass latitude and longitude together."})
        if tags is not None and (tags_add or tags_remove):
            return _dumps({"error": "tags cannot be combined with tags_add or tags_remove."})
        if latitude is not None and not (-90 <= latitude <= 90 and -180 <= longitude <= 180):  # type: ignore[operator]
            return _dumps({"error": "latitude must be -90..90 and longitude -180..180."})
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client, did, org_id = get_client(), found["match"]["id"], found["organization_id"]
        before = _device_detail(client, did, org_id)
        old_tags = _device_tags(before)
        new_tags: dict[str, str] | None = None
        if tags is not None:
            new_tags = {t["key"]: t["value"] for t in _tags_arg(tags)}
        elif tags_add or tags_remove:
            new_tags = {**old_tags, **{t["key"]: t["value"] for t in _tags_arg(tags_add)}}
            for key in tags_remove or []:
                new_tags.pop(key, None)
        body: dict[str, Any] = {}
        if name is not None and name.strip() != before.get("name"):
            body["name"] = name.strip()
        if notes is not None and notes != (before.get("notes") or ""):
            body["notes"] = notes
        if new_tags is not None and new_tags != old_tags:
            body["tags"] = [{"key": k, "value": v} for k, v in new_tags.items()]
        if latitude is not None:
            body["location"] = {"latitude": latitude, "longitude": longitude}
        if screenshots_disabled is not None:
            body["screenshots_disabled"] = bool(screenshots_disabled)
        if not body:
            raise ToolRefusal("Nothing to change: the device already has these values (or no field was given).", device=found["match"])
        client.put(f"devices/{did}", org_id=org_id, json=body)
        after = _device_detail(client, did, org_id)
        ok = True
        if "name" in body:
            ok = ok and after.get("name") == body["name"]
        if "notes" in body:
            ok = ok and (after.get("notes") or "") == body["notes"]
        if "tags" in body:
            ok = ok and _device_tags(after) == new_tags
        changed = [k for k in body if k != "location"] + (["location"] if "location" in body else [])
        return _dumps(
            {
                "summary": f"Updated {after.get('name') or found['match'].get('name')}: {', '.join(changed)}" + ("" if ok else " (NOT fully confirmed by read-back)") + ".",
                "device": {"id": did, "name": after.get("name")},
                "before": {"name": before.get("name"), "notes": before.get("notes"), "tags": old_tags},
                "after": {"name": after.get("name"), "notes": after.get("notes"), "tags": _device_tags(after), "location": after.get("location")},
                "read_back": {"confirmed": ok},
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Move a device to another fleet (PUT /devices/{id}/fleet). The device takes the new fleet's configuration, which is "
        "pushed to it immediately (the workload changes now, without a rollout), and its fleet-level tags, update window "
        "and policies change. device = name/UUID/tag (ambiguous matches return candidates); fleet = name or UUID. The "
        "result shows the device's old and new fleet and the configuration it now resolves to, read back. Data wipe is not "
        "offered here. Destructive: changes what the device runs — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
def move_device_to_fleet(device: str, fleet: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client, did, org_id = get_client(), found["match"]["id"], found["organization_id"]
        target = _find_fleet(client, org_id, fleet)
        before = _device_detail(client, did, org_id)
        old_fleet = before.get("fleet") or {}
        if str(old_fleet.get("id") or "").lower() == str(target["id"]).lower():
            raise ToolRefusal(f"No change: {found['match'].get('name')} is already in {target.get('name')}.", device=found["match"])
        new_cfg = _safe(lambda: _fleet_assignment(client, org_id, target["id"])[0], [], "target configuration")
        client.put(f"devices/{did}/fleet", org_id=org_id, json={"flotilla_id": target["id"]})
        after = _device_detail(client, did, org_id)
        new_fleet = after.get("fleet") or {}
        confirmed = str(new_fleet.get("id") or "").lower() == str(target["id"]).lower()
        cfg = (new_cfg or {}).get("configuration")
        return _dumps(
            {
                "summary": f"{found['match'].get('name')}: {old_fleet.get('name')} → {new_fleet.get('name') or target.get('name')}"
                + ("" if confirmed else " (NOT confirmed by read-back)")
                + (f"; now runs {cfg.get('name')} @ {new_cfg['assignment']}" if cfg else "; the new fleet has no configuration")
                + ".",
                "device": {"id": did, "name": found["match"].get("name")},
                "from_fleet": {"id": old_fleet.get("id"), "name": old_fleet.get("name")},
                "to_fleet": {"id": new_fleet.get("id") or target["id"], "name": new_fleet.get("name") or target.get("name")},
                "configuration": new_cfg,
                "read_back": {"confirmed": confirmed},
                "next": [{"tool": "get_device_workload", "why": "Confirm the workload that is running after the move.", "arguments": {"device": did}}],
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


def _device_config_view(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    base, merged = ops.norm_spec(payload.get("base_config")), ops.norm_spec(payload.get("merged_config"))
    info = payload.get("configuration_info") or {}
    return {
        "fleet_id": payload.get("fleet_id"),
        "configuration": {"id": payload.get("configuration_id"), "name": info.get("name")} if payload.get("configuration_id") else None,
        "assignment": payload.get("config_version") if payload.get("is_pinned") else ("latest" if payload.get("configuration_id") else None),
        "resolved_version": payload.get("config_version"),
        "has_override": bool(payload.get("has_override")),
        "override": payload.get("host_override"),
        "overridden_fields": ops.spec_diff(base, merged),
        "merged": ops.mask_spec(merged),
    }


@mcp.tool(
    description=(
        "A device's effective configuration (GET /devices/{id}/configuration): which fleet and configuration it inherits "
        "(version, and whether the fleet follows 'latest' or is pinned), the per-device override if any, exactly which "
        "fields the override changes, and the merged spec the device should run. Credential-looking environment values "
        "are masked. This is the desired configuration; get_device_workload shows what is actually running. Read-only."
    ),
    annotations=_READ_ONLY,
)
def get_device_configuration(device: str, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        payload = get_client().get(f"devices/{found['match']['id']}/configuration", org_id=found["organization_id"])
        view = _device_config_view(payload)
        cfg = view["configuration"]
        text = (
            f"{cfg.get('name')} @ {view['assignment']}" + (f" with an override on {len(view['overridden_fields'])} field(s)" if view["has_override"] else ", no override")
            if cfg
            else "no configuration (its fleet has none)"
        )
        return _dumps({"summary": f"{found['match'].get('name')}: {text}.", "device": found["match"], **view})
    except Exception as exc:
        return _err(exc)


def _check_override(override: Any) -> dict[str, Any]:
    if not isinstance(override, dict) or not override:
        raise ValueError("override must be a non-empty JSON object.")
    unknown = sorted(set(override) - set(_OVERRIDE_KEYS))
    if unknown:
        raise ValueError(f"Unknown override field(s) {unknown}; the backend would ignore them. Allowed: {list(_OVERRIDE_KEYS)}.")
    state = override.get("desiredState")
    if state is not None and str(state).strip().upper() not in ("RUNNING", "STOPPED", ""):
        raise ValueError("desiredState must be RUNNING or STOPPED.")
    return override


@mcp.tool(
    description=(
        "Set a per-device configuration override (PUT /devices/{id}/configuration): fields layered over the fleet's "
        "configuration for this one device. override keys: image, command, environment {NAME: value}, mounts, options, "
        "volumes, ports, image_cleaning, registry_credential_id, desiredState (RUNNING|STOPPED; STOPPED stops only this "
        "device's workload). By default the override is merged into the existing one (merge patch: null removes a field); "
        "replace=true replaces it entirely. Unknown keys are rejected. Stores the override; it reaches the device when it "
        "reconnects or receives a push (render_device_document with push=true pushes this one device now). Returns the effective changes, read back. Changes what the device runs — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
def set_device_configuration_override(
    device: str,
    override: dict[str, Any],
    reason: str | None = None,
    replace: bool = False,
    organization_id: str | None = None,
) -> str:
    try:
        override = _check_override(override)
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client, did, org_id = get_client(), found["match"]["id"], found["organization_id"]
        current = client.get(f"devices/{did}/configuration", org_id=org_id)
        before = _device_config_view(current)
        if not before["configuration"]:
            return _dumps({"error": "The device's fleet has no configuration to override. Assign one first (assign_fleet_configuration).", "device": found["match"]})
        existing = (current or {}).get("host_override") or {}
        wanted = override if replace else ops.merge_patch(existing, override)
        if not wanted:
            return _dumps({"error": "The resulting override is empty; use clear_device_configuration_override to remove it."})
        if wanted == existing:
            raise ToolRefusal("No change: the device already has this override.", device=found["match"])
        body: dict[str, Any] = {"override": wanted}
        if reason and reason.strip():
            body["reason"] = reason.strip()
        client.put(f"devices/{did}/configuration", org_id=org_id, json=body)
        after = _device_config_view(client.get(f"devices/{did}/configuration", org_id=org_id))
        confirmed = after["has_override"] and not ops.spec_diff(after["override"], wanted, mask=False)
        return _dumps(
            {
                "summary": f"{found['match'].get('name')}: override set; changes vs fleet configuration: {ops.diff_summary(after['overridden_fields'])}.",
                "device": found["match"],
                "before": {"has_override": before["has_override"], "override": before["override"]},
                "after": {"has_override": after["has_override"], "override": after["override"], "overridden_fields": after["overridden_fields"]},
                "read_back": {"confirmed": confirmed},
                "delivery": "The override is stored now and reaches the device when it reconnects or next receives a push.",
                "next": [{"tool": "get_device_configuration", "why": "Review the effective configuration."}],
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Remove a device's configuration override (DELETE /devices/{id}/configuration/override) so it follows its fleet's "
        "configuration again. Refuses when there is no override. reason is recorded. Returns the read-back state. "
        "Changes what the device will run (it reconnects/pushes to apply) — explicit request only."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True),
)
def clear_device_configuration_override(device: str, reason: str | None = None, organization_id: str | None = None) -> str:
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client, did, org_id = get_client(), found["match"]["id"], found["organization_id"]
        before = _device_config_view(client.get(f"devices/{did}/configuration", org_id=org_id))
        if not before["has_override"]:
            raise ToolRefusal(f"No change: {found['match'].get('name')} has no configuration override.", device=found["match"])
        client.delete(f"devices/{did}/configuration/override", org_id=org_id, params={"reason": (reason or "").strip() or None})
        after = _device_config_view(client.get(f"devices/{did}/configuration", org_id=org_id))
        confirmed = not after["has_override"]
        return _dumps(
            {
                "summary": f"{found['match'].get('name')}: override " + ("cleared; follows the fleet configuration again." if confirmed else "still present after the request (NOT confirmed)."),
                "device": found["match"],
                "removed_override": before["override"],
                "read_back": {"confirmed": confirmed, "has_override": after["has_override"]},
            }
        )
    except ToolRefusal as refusal:
        return _dumps(refusal.payload)
    except Exception as exc:
        return _err(exc)


# -------------------------------------------------------------- rollouts ---


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def _resolve_fleets(client: AdmiralClient, org_id: str, fleets: Any) -> list[dict[str, Any]]:
    refs = _as_list(fleets)
    if not refs:
        raise ValueError("fleet is required (one name/UUID or a list)")
    out: list[dict[str, Any]] = []
    for ref in refs:
        if is_uuid(ref):
            out.append({"id": ref, "name": None})
        else:
            out.append(_unique_by_name(_items(client.list_fleets(org_id=org_id, search=ref, limit=100)), ref, "fleet"))
    ids = [str(f["id"]).lower() for f in out]
    if len(set(ids)) != len(ids):
        raise ValueError("The same fleet was listed twice.")
    return out


def _rollout_type(value: str | None) -> str:
    kind = (value or "config").strip().lower()
    if kind not in _ROLLOUT_TYPES:
        raise ValueError(f"Unsupported rollout type {value!r}; use one of {list(_ROLLOUT_TYPES)}.")
    return kind


def _type_spec(
    kind: str,
    force: bool,
    container_target: str | None,
    use_current_versions: bool,
    version_ids: list[str] | None,
    final_action: str | None,
) -> dict[str, Any]:
    """Body fragment carrying the type-specific spec of a non-config rollout."""
    if kind == "reboot":
        return {"reboot_spec": {"force": True}} if force else {}
    if kind == "restart_workload":
        return {"restart_workload_spec": {"container_target": container_target}} if container_target else {}
    if kind == "system_update":
        ids = _as_list(version_ids)
        if not use_current_versions and not ids:
            raise ValueError("system_update needs version_ids [UUID] or use_current_versions=true.")
        if use_current_versions and ids:
            raise ValueError("Pass version_ids or use_current_versions, not both.")
        action = (final_action or "reboot").strip().lower()
        if action not in ("reboot", "none", "restart"):
            raise ValueError("final_action must be reboot, none or restart.")
        for vid in ids:
            if not is_uuid(vid):
                raise ValueError(f"version_ids must be UUIDs; got {vid!r}.")
        spec: dict[str, Any] = {"final_action": action}
        if ids:
            spec["version_ids"] = ids
        else:
            spec["use_current_versions"] = True
        return {"system_update_spec": spec}
    return {}


def _effective_strategy(strategy: dict[str, Any] | None) -> dict[str, Any]:
    return {**_STRATEGY_DEFAULTS, **(_normalise_strategy(strategy) or {})}


@mcp.tool(
    description=(
        "List rollouts (GET /rollouts), newest first, as compact rows: id, name, type (config, reboot, restart_workload, "
        "system_update), status (pending, scheduled, in_progress, paused, completed, failed, cancelled, rolled_back), "
        "progress, fleets, configuration + version, counts, paused reason. Filters: fleet (name/UUID or list), status "
        "(one or comma-separated), type, search, active_only. Read-only. Next: get_rollout / list_rollout_devices / "
        "watch_rollout for one rollout."
    ),
    annotations=_READ_ONLY,
)
def list_rollouts(
    fleet: str | list[str] | None = None,
    status: str | None = None,
    type: str | None = None,
    search: str | None = None,
    active_only: bool = False,
    limit: int = 20,
    page: int = 1,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        kind = _rollout_type(type) if type else None
        fleet_ids = [f["id"] for f in _resolve_fleets(client, org_id, fleet)] if fleet else []
        statuses = _as_list(status)
        if active_only:
            statuses = _ACTIVE_ROLLOUT_STATUSES.split(",")
        payload = client.get(
            "rollouts",
            org_id=org_id,
            params={
                "status": ",".join(s.lower() for s in statuses),
                "fleet_id": ",".join(fleet_ids),
                "search": search,
                "limit": max(1, min(int(limit), 200)),
                "page": max(1, int(page)),
            },
        )
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        rows = [_rollout_row(r) for r in _items(payload) if not kind or r.get("type") == kind]
        total = (pagination or {}).get("total", len(rows))
        return _dumps(
            {
                "summary": f"{len(rows)} rollout(s) shown ({total} match the server-side filters)" + (f", type {kind}" if kind else "") + ".",
                "rollouts": rows,
                "pagination": pagination,
                "next": [{"tool": "get_rollout", "why": "Detail, strategy and counts for one rollout."}],
            }
        )
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "READ-ONLY rollout planner: shows what create_rollout would do without creating anything. Resolves the fleet(s) "
        "(name/UUID or list) and, for type=config, the configuration and version ('latest' or an integer); shows per fleet "
        "the current vs target configuration/version, the spec diff, device counts (total/online/offline) and the "
        "POST /rollouts/impact result, the effective strategy (your overrides over the server defaults canary 1, "
        "maxInFlight 50, maxUnavailable 2, failureThreshold 0.1, progressDeadline 30m), and warnings: target equals what "
        "the fleet already runs (such a rollout never converges), offline devices (only admitted while online), unfinished "
        "rollouts on the same fleets, a deprecated/archived configuration, a signature policy in the spec, disruptive types. "
        "Ends with the exact create_rollout arguments. type: config (default), reboot, restart_workload, system_update "
        "(use_current_versions or version_ids)."
    ),
    annotations=_READ_ONLY,
)
def preview_rollout(
    fleets: str | list[str],
    configuration: str | None = None,
    version: int | str = "latest",
    type: str = "config",
    strategy: dict[str, Any] | None = None,
    force: bool = False,
    container_target: str | None = None,
    use_current_versions: bool = False,
    version_ids: list[str] | None = None,
    final_action: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        client, org_id = _ctx(organization_id)
        kind = _rollout_type(type)
        strat = _effective_strategy(strategy)
        fleet_rows = _resolve_fleets(client, org_id, fleets)
        fleet_ids = [str(f["id"]) for f in fleet_rows]
        spec_fragment = _type_spec(kind, force, container_target, use_current_versions, version_ids, final_action)
        warnings: list[str] = []
        cfg: dict[str, Any] | None = None
        target_version: int | None = None
        target_spec: dict[str, Any] = {}
        if kind == "config":
            if not configuration:
                return _dumps({"error": "configuration is required for type=config."})
            cfg = _find_config(client, org_id, configuration)
            latest = _latest_of(cfg)
            target_version = latest if str(version).strip().lower() == "latest" else int(version)
            if not target_version or target_version < 1 or (latest and target_version > latest):
                return _dumps({"error": f"version must be between 1 and {latest} (or 'latest').", "configuration": _ref(cfg)})
            got = client.get(f"configurations/{cfg['id']}/versions/{target_version}", org_id=org_id)
            target_spec = ops.norm_spec((got or {}).get("spec") if isinstance(got, dict) else {})
            if str(cfg.get("status") or "").lower() in ("deprecated", "archived"):
                warnings.append(f"The configuration is {cfg.get('status')}.")
            if target_spec.get("signaturePolicy"):
                warnings.append(
                    "The spec sets an image signature policy; creating a version with it needs the organisation's signature-policy entitlement (a 402 means it is not enabled)."
                )

        def per_fleet(row: dict[str, Any]) -> dict[str, Any]:
            fid = str(row["id"])
            detail = _safe(lambda: client.get(f"fleets/{fid}", org_id=org_id), warnings, f"fleet {row.get('name') or fid}") or row
            item: dict[str, Any] = {
                "fleet": {"id": fid, "name": detail.get("name") or row.get("name")},
                "devices": {"total": detail.get("devices"), "online": detail.get("online"), "offline": detail.get("offline")},
            }
            if kind == "config":
                current, current_spec = _fleet_assignment(client, org_id, fid)
                cur_cfg = current["configuration"]
                item["current"] = {"configuration": cur_cfg and {"id": cur_cfg["id"], "name": cur_cfg["name"]}, "assignment": current["assignment"], "version": current["resolved_version"]}
                item["target"] = {"configuration": _ref(cfg or {}), "version": target_version}
                item["same_version_noop"] = bool(
                    cur_cfg and str(cur_cfg["id"]).lower() == str((cfg or {})["id"]).lower() and current["resolved_version"] == target_version
                )
                item["diff"] = ops.spec_diff(current_spec, target_spec)
            return item

        plans = _parallel_map(per_fleet, fleet_rows)
        impact = _safe(lambda: client.post("rollouts/impact", org_id=org_id, json={"fleet_ids": fleet_ids, "type": kind}), warnings, "impact")
        active = _safe(lambda: _active_rollouts(client, org_id, fleet_ids), warnings, "rollouts") or []
        total = sum(int(p["devices"]["total"] or 0) for p in plans)
        online = sum(int(p["devices"]["online"] or 0) for p in plans)
        if isinstance(impact, dict):
            total, online = impact.get("total_devices", total), impact.get("online_count", online)
        for p in plans:
            name = p["fleet"]["name"] or p["fleet"]["id"]
            if p.get("same_version_noop"):
                warnings.append(
                    f"{name} already runs {(cfg or {}).get('name')} v{target_version}: devices will not change, and a config rollout to the version they already run never converges (it sits 'assigned' until the progress deadline). Pick a different version or skip this fleet."
                )
            if p["devices"]["total"] == 0:
                warnings.append(f"{name} has no devices.")
            elif p["devices"]["offline"]:
                warnings.append(f"{name}: {p['devices']['offline']} device(s) offline; they are only admitted when online, so the rollout stays in_progress until they return.")
        if total and online < strat["canary"]:
            warnings.append(f"Only {online} device(s) online but canary is {strat['canary']}.")
        if active:
            warnings.append(f"{len(active)} unfinished rollout(s) already target these fleets (see active_rollouts); a newer rollout takes over device pins.")
        if isinstance(impact, dict) and impact.get("is_disruptive"):
            warnings.append(f"A {kind} rollout is disruptive: devices restart or reboot.")
        if kind == "system_update" and isinstance(impact, dict) and impact.get("incompatible_count"):
            warnings.append(f"{impact['incompatible_count']} device(s) have no compatible version.")
        args: dict[str, Any] = {"fleet": fleet_ids if len(fleet_ids) > 1 else fleet_ids[0], "type": kind}
        if kind == "config":
            args.update({"configuration": (cfg or {})["id"], "version": target_version})
        elif kind == "reboot" and force:
            args["force"] = True
        elif kind == "restart_workload" and container_target:
            args["container_target"] = container_target
        elif kind == "system_update":
            args.update({"final_action": spec_fragment["system_update_spec"]["final_action"]})
            args.update({k: v for k, v in spec_fragment["system_update_spec"].items() if k in ("version_ids", "use_current_versions")})
        if strategy:
            args["strategy"] = strategy
        blocking = [p for p in plans if p.get("same_version_noop")]
        label = f"{(cfg or {}).get('name')} v{target_version}" if kind == "config" else kind
        return _dumps(
            {
                "summary": f"Preview {label} → {len(plans)} fleet(s), {total} device(s) ({online} online)"
                + (f"; {len(warnings)} warning(s)" if warnings else "; no warnings")
                + ". Nothing was created.",
                "read_only": True,
                "type": kind,
                "configuration": ({**_ref(cfg or {}), "status": (cfg or {}).get("status"), "latest_version": _latest_of(cfg or {})} if cfg else None),
                "fleets": plans,
                "impact": impact,
                "strategy": strat,
                "active_rollouts": active,
                "warnings": warnings,
                "ready": not blocking and bool(total),
                "create_rollout": {"tool": "create_rollout", "arguments": args},
            }
        )
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Migration gate: protocol / agent-version census (read-only)
# ---------------------------------------------------------------------------


def _all_devices(client: AdmiralClient, org_id: str, fleet_id: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    page = 1
    while page <= 50:
        payload = client.list_devices(org_id=org_id, fleet_id=fleet_id, page=page, limit=100)
        rows = _items(payload) or (payload.get("devices") if isinstance(payload, dict) else None) or []
        out.extend(r for r in rows if isinstance(r, dict))
        pagination = payload.get("pagination") if isinstance(payload, dict) else None
        total_pages = (pagination or {}).get("totalPages") or 1
        if page >= total_pages or not rows:
            break
        page += 1
    return out


@mcp.tool(
    description=(
        "Migration gate census (read-only): counts of an organisation's devices by wire protocol (0 = legacy "
        "container push, 1 = desired-state documents, no_state = never reported state) and by agent version "
        "(admiral-init version from state.system.versions.initVersion, else the device's systemSpec.versions), from "
        "the device list plus each device's stored GET /devices/{id}/state. Also lists each device's protocol/version. "
        "Optional fleet_id filter."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
def list_fleet_protocols(
    fleet_id: str | None = None,
    organization_id: str | None = None,
) -> str:
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        devices = _all_devices(client, org_id, fleet_id)

        def census(raw: dict[str, Any]) -> dict[str, Any]:
            summary = summarise_device(raw)
            online = raw.get("isOnline") if "isOnline" in raw else raw.get("onlineStatus")
            if isinstance(online, dict):  # live list rows nest {isOnline, lastSeen, latencyMs}
                online = online.get("isOnline")
            if online is None:
                online = summary.get("status") == "online"
            row: dict[str, Any] = {
                "id": summary.get("id"),
                "name": summary.get("name"),
                "fleet": (raw.get("fleet") or {}).get("name") if isinstance(raw.get("fleet"), dict) else None,
                "online": bool(online),
                "protocol": raw.get("protocol"),
                "agentVersion": raw.get("agent_version") or raw.get("agentVersion"),
            }
            try:
                st = client.get_device_state(str(row["id"]), org_id=org_id)
                st = st if isinstance(st, dict) else {}
                if st.get("protocol") is not None:
                    row["protocol"] = st.get("protocol")
                versions = ((st.get("state") or {}).get("system") or {}).get("versions") or {}
                if not row["agentVersion"] and versions.get("initVersion"):
                    row["agentVersion"] = versions.get("initVersion")
                    row["agentVersionSource"] = "state.system.versions"
                row["rootfsVersion"] = versions.get("rootfsVersion")
                row["stateReceivedAt"] = st.get("receivedAt")
            except AdmiralAPIError as exc:
                if exc.status == 404:
                    row["protocol"] = row["protocol"] if row["protocol"] is not None else "no_state"
                else:
                    row["state_error"] = str(exc)
            if not row["agentVersion"]:
                # Stored state can carry empty system.versions; the device's
                # system spec (GET /devices/{id}) keeps the last reported versions.
                try:
                    detail = client.get_device(str(row["id"]), org_id=org_id)
                    spec_versions = ((detail or {}).get("systemSpec") or {}).get("versions") or {}
                    if spec_versions.get("initVersion"):
                        row["agentVersion"] = spec_versions.get("initVersion")
                        row["agentVersionSource"] = "systemSpec.versions"
                        row["rootfsVersion"] = row.get("rootfsVersion") or spec_versions.get("rootfsVersion")
                except AdmiralAPIError as exc:
                    row["detail_error"] = str(exc)
            return row

        rows = _parallel_map(census, devices, max_workers=8)

        by_protocol = Counter(str(r.get("protocol") if r.get("protocol") is not None else "unknown") for r in rows)
        by_agent = Counter(str(r.get("agentVersion") or "unknown") for r in rows)
        by_both = Counter(
            f"{r.get('protocol') if r.get('protocol') is not None else 'unknown'}|{r.get('agentVersion') or 'unknown'}"
            for r in rows
        )
        out: dict[str, Any] = {
            "organization_id": org_id,
            "fleet_id": fleet_id,
            "total": len(rows),
            "byProtocol": dict(by_protocol),
            "byAgentVersion": dict(by_agent),
            "byProtocolAndAgentVersion": dict(by_both),
            "legacyDevices": [
                {"id": r["id"], "name": r["name"], "agentVersion": r.get("agentVersion"), "online": r.get("online")}
                for r in rows
                if str(r.get("protocol")) == "0"
            ],
            "devices": rows[:300],
            "source": "device list + stored state per device",
        }
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Customer self-service troubleshooting (see troubleshoot.py for the routes used)
# ---------------------------------------------------------------------------

_TROUBLESHOOT_ANNOTATIONS = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

_OFFLINE_READABLE = [
    "last seen, check-in schedule and the backend verdict (GET /devices/{id}/diagnose)",
    "the last stored state: conditions, storage, time, transport, boot reasons (GET /devices/{id}/state)",
    "historical logs and lifecycle events (get_device_logs, get_device_events)",
    "the assigned configuration (get_device_document)",
]


@mcp.tool(
    description=(
        "Answer 'what is wrong with this device right now?' in one call. Gathers, with the caller's own permissions: "
        "observed state (live when the device is online), the backend diagnosis, a fresh on-device probe (one per 5 s per "
        "device; on 429 the previous result is reused and the report says so), recent error logs (distilled to signatures), "
        "notable events, workload state and progress operations, storage fullness, clock sync, transport/connectivity and "
        "desired-vs-observed drift (assigned generation vs applied). Returns {summary, overall, findings[], support, checked}; "
        "each finding has severity, area, finding, short quoted evidence, likely_cause, what_you_can_do (concrete steps), a "
        "docs.admrl.co link and whether Admiral support is needed. symptom (optional free text, e.g. 'screen is black', "
        "'keeps rebooting') promotes the related findings. Offline devices: explains last seen, what can still be read and "
        "the common causes. Read-only; the probe is the same read-only call as probe_device. Never returns tokens, PSKs, "
        "SSIDs or the kernel command line."
    ),
    annotations=_TROUBLESHOOT_ANNOTATIONS,
)
def troubleshoot_device(
    device: str,
    symptom: str | None = None,
    lookback_hours: float = 24,
    run_probe: bool = True,
    organization_id: str | None = None,
) -> str:
    # Routes: routes_diagnostics.go:24 (state), :26 (diagnose), :27-28 (probe), :36 (system-services); routes.go:536
    # (events), :559 (workload), :457 (device-stats), :932/:929 (logs); routes_documents.go:143,146 (document, render).
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client = get_client()
        org_id = found["organization_id"]
        match = found["match"]
        ev = ts.gather(
            client,
            org_id,
            match,
            include={"diagnose", "state", "workload", "events", "logs", "document", "live", "probe", "services", "stats", "memtest"},
            lookback_hours=lookback_hours,
            run_probe_flag=run_probe,
            pmap=_parallel_map,
        )
        findings = ts.analyse(ev, symptom=symptom)
        docs_note = ts.attach_docs(findings)
        online = ev["online"]
        verdict = ts.overall(findings, online)
        name = match.get("name") or match.get("id")
        if findings:
            top = findings[0]
            summary = f"{name} is {verdict}: {len(findings)} finding(s); top: {top['finding']}."
        else:
            summary = f"{name} looks healthy: no problems found in state, diagnosis, logs, events, storage, time or connectivity."
        out: dict[str, Any] = {
            "summary": summary,
            "device": ts.device_header(ev),
            "overall": verdict,
            "online": online,
            "symptom": symptom,
            "findings": ts.public(findings),
            "next_steps": [f["what_you_can_do"][0] for f in findings[:3] if f["what_you_can_do"]],
            "support": ts.support_verdict(findings),
            "state": {
                "source": ev.get("state_source"),
                "received_at": ev.get("state_received_at"),
                "protocol": ev.get("protocol"),
            },
            "checked": ts.checked_sources(ev),
        }
        if docs_note:
            out["docs_note"] = docs_note
        if not online:
            out["offline"] = {
                "last_seen": match.get("last_seen"),
                "can_still_read": _OFFLINE_READABLE,
                "cannot_read": ["live state, the probe, system services, screenshots (all need the device online)"],
                "common_causes": ts.KB["device_offline"]["cause"],
            }
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Check whether a device can talk to Admiral: link (interfaces/IPs), DNS, TCP, NTP/clock, NATS/transport (kind, latency, "
        "reconnects, last backend contact, telemetry stream) and probe results, as a pass/warn/error table plus ranked findings "
        "and the classic causes (captive portal, DNS, firewall to the backend, TLS inspection/proxy, clock skew, weak link). Runs a "
        "network+time probe when the device is online (rate limit one per 5 s per device; on 429 the previous result is reused and "
        "stated). Offline devices: shows last seen, last known transport/clock and what to check on site. Read-only."
    ),
    annotations=_TROUBLESHOOT_ANNOTATIONS,
)
def check_device_connectivity(
    device: str,
    run_probe: bool = True,
    lookback_hours: float = 24,
    organization_id: str | None = None,
) -> str:
    # Routes: routes.go:432 (device/system spec IPs), routes_diagnostics.go:24,26,27-28 (state, diagnose, probe),
    # routes.go:932/:929 (logs for tls/dns errors).
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client = get_client()
        org_id = found["organization_id"]
        match = found["match"]
        ev = ts.gather(
            client,
            org_id,
            match,
            include={"detail", "diagnose", "state", "live", "probe", "logs"},
            lookback_hours=lookback_hours,
            run_probe_flag=run_probe,
            probe_sections=["network", "time"],
            pmap=_parallel_map,
        )
        findings = ts.analyse(ev, areas={"connectivity", "time"})
        docs_note = ts.attach_docs(findings)
        checks = ts.connectivity_checks(ev)
        detail = ev.get("detail") if isinstance(ev.get("detail"), dict) else {}
        ips = collect_ips(detail) if detail else []
        bad = [c for c in checks if c["status"] in ("warn", "error")]
        online = ev["online"]
        name = match.get("name") or match.get("id")
        if not online:
            summary = f"{name} is offline (last seen {match.get('last_seen') or 'never'}); showing last known link, transport and clock."
        elif bad:
            summary = f"{name} is online but {len(bad)} connectivity check(s) are not clean: " + ", ".join(c["check"] for c in bad) + "."
        else:
            summary = f"{name} is online and connectivity checks are clean."
        codes = {f["code"] for f in findings}
        suspected = []
        if "dns_failure" in codes:
            suspected.append(ts._CONNECTIVITY_CAUSES[1])
        if codes & {"backend_unreachable", "wss_fallback"}:
            suspected.append(ts._CONNECTIVITY_CAUSES[2])
        if "tls_error" in codes:
            suspected.append(ts._CONNECTIVITY_CAUSES[3])
        if codes & {"clock_skew", "time_not_synced", "rtc_drift"}:
            suspected.append(ts._CONNECTIVITY_CAUSES[4])
        if codes & {"degraded_link", "high_latency"}:
            suspected.append(ts._CONNECTIVITY_CAUSES[5])
        if "backend_unreachable" in codes or not online:
            suspected.append(ts._CONNECTIVITY_CAUSES[0])
        out: dict[str, Any] = {
            "summary": summary,
            "device": ts.device_header(ev),
            "online": online,
            "last_seen": match.get("last_seen"),
            "checks": checks,
            "interfaces": ips[:8],
            "findings": ts.public(findings),
            "suspected_causes": list(dict.fromkeys(suspected)),
            "classic_causes": ts._CONNECTIVITY_CAUSES,
            "support": ts.support_verdict(findings),
            "checked": ts.checked_sources(ev),
        }
        if docs_note:
            out["docs_note"] = docs_note
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


@mcp.tool(
    description=(
        "Explain why a device's workload is failing: crash/restart loops (failure count, last exit code with its meaning, last "
        "exit time), OOM kills, image pull/extract errors, signature-policy and USB-policy denials, configuration rollbacks and "
        "stuck updates (progress operations), with short scrubbed log excerpts from the workload/runtime and recent restart "
        "events. Returns {summary, workload, findings, log_excerpts, restart_events}. Reads stored state (live when online); "
        "needs the Telemetry add-on for log excerpts (says so when missing). Read-only."
    ),
    annotations=_TROUBLESHOOT_ANNOTATIONS,
)
def explain_workload_failure(
    device: str,
    lookback_hours: float = 24,
    organization_id: str | None = None,
) -> str:
    # Routes: routes.go:559 (workload), routes_diagnostics.go:24,26,36 (state, diagnose, system-services),
    # routes.go:536 (events), routes.go:932/:929 (logs).
    try:
        found = _resolve_device(device, organization_id)
        if not found.get("match"):
            return _dumps(found)
        client = get_client()
        org_id = found["organization_id"]
        match = found["match"]
        ev = ts.gather(
            client,
            org_id,
            match,
            include={"diagnose", "state", "workload", "events", "logs", "live", "services"},
            lookback_hours=lookback_hours,
            run_probe_flag=False,
            pmap=_parallel_map,
        )
        findings = ts.analyse(ev, areas={"workload", "update", "resources"})
        docs_note = ts.attach_docs(findings)
        st = ev.get("state_doc") or {}
        wl, rt = st.get("workload") or {}, st.get("runtime") or {}
        stored = ev.get("workload") if isinstance(ev.get("workload"), dict) else {}
        running = stored.get("running") or (stored.get("data") or {}).get("running") or {}
        exit_code = rt.get("exitCode")
        workload = {
            "state": wl.get("state") or stored.get("state") or (ev.get("diagnose") or {}).get("verdict", {}).get("workloadState"),
            "configuration": wl.get("configurationName") or running.get("configurationName"),
            "image": wl.get("image") or running.get("image"),
            "version": wl.get("version") or running.get("version"),
            "started_at": wl.get("startedAt") or stored.get("startedAt"),
            "failure_count": rt.get("failureCount"),
            "last_exit_code": exit_code,
            "last_exit_meaning": ts._EXIT_MEANING.get(exit_code, "application-defined") if exit_code is not None else None,
            "last_exit_at": rt.get("lastExitAt") if not str(rt.get("lastExitAt") or "").startswith("0001") else None,
            "oom_killed": rt.get("oomKilled"),
            "oom_events": rt.get("oomEvents"),
            "assigned_config_applied": rt.get("assignedConfigApplied"),
            "transition": wl.get("transition"),
            "rollback": wl.get("rollback"),
            "signature": {k: (wl.get("signature") or {}).get(k) for k in ("required", "verified", "format", "error")} if wl.get("signature") else None,
            "operations": [
                compact_op
                for compact_op in (
                    {k: op.get(k) for k in ("kind", "target", "phase", "message", "error", "attempt") if op.get(k) not in (None, "")}
                    for op in ((st.get("progress") or {}).get("operations") or [])
                    if isinstance(op, dict)
                )
            ][:8],
            "state_source": ev.get("state_source"),
            "state_received_at": ev.get("state_received_at"),
        }
        entries = ts.log_entries(ev)
        wanted = [
            e
            for e in entries
            if ts._level(e) in ("error", "fatal", "critical", "panic", "warn")
            or any(k in ts._source(e) for k in ("workload", "container", "crun", "runtime"))
        ]
        excerpts = [
            {"time": e.get("timestamp"), "level": ts._level(e), "source": e.get("source"), "message": ts.scrub(ts._message(e), 200)}
            for e in wanted[:10]
        ]
        restarts = [
            {"time": e.get("timestamp") or e.get("time") or e.get("created_at"), "event": e.get("event")}
            for e in ((ev.get("events") or {}).get("events") or (ev.get("events") or {}).get("items") or (ev.get("events") if isinstance(ev.get("events"), list) else []) or [])
            if isinstance(e, dict) and any(k in ts._event_key(str(e.get("event"))) for k in ("crash", "exit", "restart", "workload_start", "workload_stop", "oom"))
        ][:8]
        name = match.get("name") or match.get("id")
        if findings:
            summary = f"{name}: {findings[0]['finding']}" + (f" (+{len(findings) - 1} more)" if len(findings) > 1 else "") + "."
        else:
            summary = f"{name}: workload is {str(workload['state'] or 'unknown').lower()}; no failure signatures found."
        out: dict[str, Any] = {
            "summary": summary,
            "device": ts.device_header(ev),
            "online": ev["online"],
            "workload": workload,
            "findings": ts.public(findings),
            "log_excerpts": excerpts,
            "restart_events": restarts,
            "support": ts.support_verdict(findings),
            "checked": ts.checked_sources(ev),
        }
        if "logs" in ev["unavailable"]:
            out["log_excerpts_note"] = f"logs unavailable: {ev['unavailable']['logs']}"
        if docs_note:
            out["docs_note"] = docs_note
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


_FLEET_MAX_PAGES = 10


@mcp.tool(
    description=(
        "Fleet (or whole-organisation) health at a glance: devices by status, offline devices by how long since last seen, and "
        "devices grouped by their top problem (backend diagnosis, stored data only) with counts, a few example names, the first "
        "step and a docs link, plus the 10 worst offenders. Bounded: lists at most 1000 devices and diagnoses up to sample_limit "
        "(default 40, max 100) devices, not-online ones first; it states how many were not diagnosed. Never one line per device. "
        "fleet is a fleet name or UUID (omit for the whole organisation). Read-only."
    ),
    annotations=_TROUBLESHOOT_ANNOTATIONS,
)
def fleet_health_report(
    fleet: str | None = None,
    sample_limit: int = 40,
    organization_id: str | None = None,
) -> str:
    # Routes: routes.go:597 (fleets), routes.go:426 (devices list, 100 per page), routes_diagnostics.go:26 (diagnose).
    try:
        client = get_client()
        org_id = DeviceResolver(client).resolve_org(organization_id)
        fleet_row: dict[str, Any] | None = None
        if fleet:
            if is_uuid(fleet):
                fleet_row = {"id": fleet, "name": None}
            else:
                fleet_row = _pick_by_name(_items(client.list_fleets(org_id=org_id, search=fleet, limit=50)), fleet, "Fleet")
        fleet_id = fleet_row["id"] if fleet_row else None

        rows: list[dict[str, Any]] = []
        total = None
        truncated = False
        for page in range(1, _FLEET_MAX_PAGES + 1):
            payload = client.list_devices(org_id=org_id, fleet_id=fleet_id, page=page, limit=100)
            batch = _items(payload) or (payload.get("devices") if isinstance(payload, dict) else None) or []
            rows.extend(r for r in batch if isinstance(r, dict))
            pagination = (payload.get("pagination") if isinstance(payload, dict) else None) or {}
            total = pagination.get("total", total)
            if page >= (pagination.get("totalPages") or 1) or not batch:
                break
        else:
            truncated = True
        summaries = [summarise_device(r) for r in rows]
        now = ts._now()
        by_status = Counter(s["status"] for s in summaries)
        buckets: Counter[str] = Counter()
        for s in summaries:
            if s["status"] == "online":
                continue
            seen = ts._parse(s.get("last_seen"))
            if seen is None:
                buckets["never seen"] += 1
                continue
            hours = (now - seen).total_seconds() / 3600
            buckets["< 1 h" if hours < 1 else "< 24 h" if hours < 24 else "< 7 d" if hours < 168 else "> 7 d"] += 1

        limit = min(max(int(sample_limit), 1), 100)
        ordered = sorted(summaries, key=lambda s: (s["status"] == "online", str(s.get("last_seen") or "")))
        sample = ordered[:limit]

        def diag(target: dict[str, Any]) -> dict[str, Any]:
            try:
                return {"device": target, "diagnosis": client.diagnose(target["id"], org_id=org_id)}
            except AdmiralAPIError as exc:
                return {"device": target, "error": str(exc)}

        results = _parallel_map(diag, sample, max_workers=8)
        groups: dict[str, dict[str, Any]] = {}
        offenders: list[dict[str, Any]] = []
        healthy = failed = 0
        for res in results:
            d = res.get("diagnosis")
            if not isinstance(d, dict):
                failed += 1
                continue
            issues = [i for i in d.get("issues") or [] if isinstance(i, dict)]
            top = (d.get("verdict") or {}).get("topIssue")
            if not top:
                healthy += 1
                continue
            sev = next((i.get("severity") for i in issues if i.get("code") == top), "warning")
            g = groups.setdefault(top, {"code": top, "severity": sev, "devices": 0, "examples": []})
            g["devices"] += 1
            if len(g["examples"]) < 5:
                g["examples"].append(res["device"].get("name"))
            offenders.append(
                {
                    "name": res["device"].get("name"),
                    "id": res["device"].get("id"),
                    "top_issue": top,
                    "severity": sev,
                    "health_score": (d.get("verdict") or {}).get("healthScore"),
                    "last_seen": res["device"].get("last_seen"),
                }
            )
        problems = sorted(groups.values(), key=lambda g: (ts._SEV_RANK.get(g["severity"], 9), -g["devices"]))
        diagnosed = len(results) - failed
        for g in problems:
            kb = ts.KB.get(g["code"], {})
            g["area"] = kb.get("area", "other")
            g["share_of_diagnosed"] = f"{round(100 * g['devices'] / diagnosed)}%" if diagnosed else None
            g["likely_cause"] = kb.get("cause", "")
            g["first_step"] = (kb.get("steps") or [None])[0]
            g["_docs_query"] = kb.get("docs") or g["code"].replace("_", " ")
        docs_note = ts.attach_docs(problems)
        offenders.sort(key=lambda o: (ts._SEV_RANK.get(o["severity"], 9), o["health_score"] if isinstance(o["health_score"], (int, float)) else 101))
        scope = f"fleet {fleet_row.get('name') or fleet_row['id']}" if fleet_row else "the organisation"
        listed = len(summaries)
        offline_n = listed - by_status.get("online", 0)
        out: dict[str, Any] = {
            "summary": (
                f"{scope}: {listed} device(s), {by_status.get('online', 0)} online, {offline_n} not online; "
                f"{sum(g['devices'] for g in problems)} of {diagnosed} diagnosed have a problem"
                + (f" (top: {problems[0]['code']} on {problems[0]['devices']})" if problems else "")
                + "."
            ),
            "organization_id": org_id,
            "fleet": fleet_row,
            "devices": {"total": total if total is not None else listed, "listed": listed, "list_truncated": truncated},
            "status_counts": dict(by_status),
            "not_online_by_last_seen": dict(buckets),
            "diagnosed": diagnosed,
            "not_diagnosed": max(listed - diagnosed, 0),
            "healthy_among_diagnosed": healthy,
            "problems": ts.public(problems),
            "worst_offenders": offenders[:10],
            "note": (
                f"Diagnosed {diagnosed} of {listed} devices (not-online first); raise sample_limit (max 100) or pass a fleet to cover more."
                if listed > diagnosed
                else None
            ),
        }
        if failed:
            out["diagnose_errors"] = failed
        if docs_note:
            out["docs_note"] = docs_note
        return _dumps(out)
    except Exception as exc:
        return _err(exc)


def main() -> None:
    # Stay up without a PAT so MCP hosts can still discover tools.
    # Tool calls fail with ConfigError until ADMRL_API_TOKEN_ID / ADMRL_API_SECRET_KEY are set.
    # Optional extension packages (ADMRL_MCP_EXTENSIONS=mod1,mod2) add tools before serving.
    load_env_extensions(mcp)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
