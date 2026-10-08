"""Customer self-service troubleshooting: gather evidence for one device, rank findings.

Everything here uses the ordinary routes any organisation member can call, so it acts with the
caller's own permissions, in the stdio server and in the browser (Pyodide) alike. Calls that
could run in parallel go through an injected ``pmap`` (the server passes its ``_parallel_map``,
which is sequential under emscripten); this module itself never starts a thread.

Customer routes used (``~/Dev/qdyn/admiral/src/routers``):

* ``GET  /devices/{id}``                         routes.go:432   presence, fleet, system spec
* ``GET  /devices/{id}/state[?live=1]``          routes_diagnostics.go:24  observed state (live: 5 s/device limit)
* ``GET  /devices/{id}/diagnose``                routes_diagnostics.go:26  stored-data verdict + issues
* ``POST /devices/{id}/diagnostics/probe``       routes_diagnostics.go:27-28  on-device probe (429 = 1 per 5 s/device)
* ``GET  /devices/{id}/workload``                routes.go:559
* ``GET  /devices/{id}/events``                  routes.go:536
* ``GET  /devices/{id}/logs`` / ``POST /metrics/logs/query``  routes.go:930,932 (Telemetry add-on: 402)
* ``GET  /devices/{id}/device-stats``            routes.go:457
* ``GET  /devices/{id}/system-services``         routes_diagnostics.go:36  (live, device must be online)
* ``GET  /devices/{id}/document``, ``POST …/document:render?dryRun=1``  routes_documents.go:143,146
* ``GET  /devices`` / ``GET /fleets``            routes.go:426,597

Never surfaced: tokens, PSKs, SSIDs, kernel cmdline, transport endpoints, proxies, probe items
the backend marked sensitive. Log excerpts pass through :func:`scrub`.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .client import AdmiralAPIError, parse_rfc3339
from .distill import _event_key, _level, _message, _source, distill_events, distill_logs, health_from_stats

SEVERITIES = ("critical", "warning", "info")
_SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}

#: The probe is limited to one call per 5 s per device (device_rate_limit.go:64-77).
PROBE_RATE_LIMIT_S = 5

_UNEXPECTED_BOOT_KINDS = {
    "watchdog_no_backend": "The device rebooted itself because it lost the Admiral backend for an hour (network problem).",
    "deadman": "The device rebooted itself after 3 days without reaching the backend (network or outage).",
    "emergency": "The agent forced an emergency reboot.",
    "kernel_fatal": "A fatal kernel error was detected and the device rebooted.",
    "agent_fatal": "The Admiral agent crashed fatally and the device rebooted.",
    "unclean": "Nothing was recorded before the reboot: power loss or a kernel panic.",
    "rollback": "The bootloader rolled back to the previous system slot (the new one failed to boot).",
}

_EXIT_MEANING = {
    0: "exited cleanly (the process finished; a workload that should keep running must not return)",
    1: "generic application error",
    2: "misuse / bad arguments",
    125: "container runtime could not start the command",
    126: "command found but not executable",
    127: "command not found in the image",
    134: "aborted (SIGABRT)",
    137: "killed (SIGKILL): usually the out-of-memory killer or a forced stop",
    139: "segmentation fault (SIGSEGV)",
    143: "terminated (SIGTERM)",
}

# ---------------------------------------------------------------- knowledge ---
# code -> area, likely cause, concrete steps, docs search query, whether support is typically needed.
KB: dict[str, dict[str, Any]] = {
    "device_offline": dict(
        area="connectivity",
        cause="No check-in. Usual causes: no power or link, Wi-Fi network or password changed, outbound access to Admiral "
        "blocked (firewall, proxy, captive portal), DNS failing, or a badly wrong clock.",
        steps=[
            "Check the device has power and a live network link (cable lit, or Wi-Fi joined) on its own screen.",
            "On the device screen press F2 (or tap Wi-Fi) to run its diagnostics and open the Wi-Fi screen.",
            "Allow outbound HTTPS (443) to Admiral: on restrictive networks devices fall back from TCP to UDP to WebSocket over 443.",
            "Try a phone hotspot to tell a device fault from a site-network fault.",
            "When it reconnects, run check_device_connectivity.",
        ],
        docs="device stalled or offline",
        support=False,
        support_note="Contact Admiral if it is powered, linked, port 443 is open and it still stays offline.",
    ),
    "workload_crashing": dict(
        area="workload",
        cause="The workload container keeps exiting and being restarted.",
        steps=[
            "Run explain_workload_failure for exit codes and the last log lines.",
            "If it started after a configuration change, assign the previous configuration version to the fleet.",
            "Check the workload's own logs for the first error after start-up (missing file, bad env var, port in use).",
        ],
        docs="workload container crash restart troubleshooting",
        support=False,
    ),
    "workload_oom": dict(
        area="workload",
        cause="The workload (or the device) ran out of memory and the kernel killed it.",
        steps=[
            "Look at the 24 h memory trend: get_device_metrics metric=memory.",
            "Reduce the workload's memory use, or raise/remove its memory limit in the configuration.",
            "Check for a leak: memory climbing steadily until the restart points to the application.",
        ],
        docs="workload out of memory limits",
        support=False,
    ),
    "workload_error": dict(
        area="workload",
        cause="The container runtime could not start the workload.",
        steps=[
            "Read the error text in the evidence: it names the failing step (image, mount, device, command).",
            "Check the image reference and tag exist, mounts/devices in the configuration exist on this hardware, and the command is correct.",
        ],
        docs="workload configuration image mounts devices",
        support=False,
    ),
    "workload_rollback": dict(
        area="workload",
        cause="The device automatically fell back to the previous configuration because the new one kept failing.",
        steps=[
            "Read the rollback reason in the evidence, fix that problem in the new configuration version.",
            "Assign the corrected version; the device refuses the failed one until a different version is applied.",
        ],
        docs="configuration rollback failed update",
        support=False,
    ),
    "image_signature": dict(
        area="workload",
        cause="The image-signature policy refused the image (unsigned, wrong signer, or a signature that does not verify).",
        steps=[
            "Check the effective signature policy (organisation, fleet and configuration levels are merged).",
            "Sign the image with a key the policy trusts, or correct the policy; then re-assign the configuration.",
            "Confirm the image digest in the evidence is the one you signed (a re-pushed tag changes the digest).",
        ],
        docs="image signature policy verification",
        support=False,
    ),
    "image_pull_failed": dict(
        area="workload",
        cause="The device could not download or unpack the workload image.",
        steps=[
            "Check the image name and tag exist in the registry and the registry credentials in the configuration are valid.",
            "Check the device network can reach the registry (DNS, proxy, firewall) and that the device has free disk space.",
            "Run check_device_connectivity to rule out DNS or clock problems (TLS fails on a wrong clock).",
        ],
        docs="workload image registry credentials pull",
        support=False,
    ),
    "usb_denied": dict(
        area="workload",
        cause="A USB device was blocked by the USB access policy.",
        steps=[
            "Review the USB allow-list for this fleet/organisation and add the device's vendor/product ID if it is expected.",
            "Re-plug the device after the policy reaches the device (check Converged in get_device_state).",
        ],
        docs="USB device policy allow list",
        support=False,
    ),
    "stuck_transition": dict(
        area="update",
        cause="An update has been in flight for a long time: a slow or failing download, or it is waiting for the update window.",
        steps=[
            "Look at the progress operations in the evidence (phase, bytes, ETA, error).",
            "If it is waiting for a maintenance window, that is expected; otherwise check connectivity and disk space.",
        ],
        docs="update progress maintenance window staged",
        support=False,
    ),
    "crash_logs": dict(
        area="logs",
        cause="Recent logs contain crash-like lines.",
        steps=["Read the quoted lines; the first occurrence usually names the cause. get_device_logs level=error has more."],
        docs="device logs troubleshooting",
        support=False,
    ),
    "log_errors": dict(
        area="logs",
        cause="Recent logs contain error-level lines.",
        steps=["Read the quoted signatures (counts collapse repeats); search the first one with get_device_logs."],
        docs="device logs troubleshooting",
        support=False,
    ),
    "notable_events": dict(
        area="events",
        cause="Lifecycle events worth a look (offline transitions, crashes, failed updates).",
        steps=["get_device_events lists them with timestamps."],
        docs="device operations monitoring events",
        support=False,
    ),
    "unexpected_reboot": dict(
        area="boot",
        cause="The device restarted without an operator asking for it.",
        steps=[
            "Check power supply and cabling; brown-outs show up as 'unclean' reboots.",
            "If it repeats, note the times (get_device_events) and open a support ticket with the findings.",
        ],
        docs="device reboot reasons boot",
        support=False,
    ),
    "wss_fallback": dict(
        area="connectivity",
        cause="Direct TCP/UDP to Admiral is blocked, so the device is connected over WebSocket on 443. It works but is slower and less stable.",
        steps=[
            "Ask the network owner to allow the device's outbound TCP/UDP traffic to Admiral, or accept the fallback.",
            "Avoid TLS-inspecting proxies on this path.",
        ],
        docs="Admiral mesh WebSocket fallback network requirements",
        support=False,
    ),
    "degraded_link": dict(
        area="connectivity",
        cause="The device's connection to Admiral keeps dropping (reconnects or instability).",
        steps=[
            "Check Wi-Fi signal or the cable; try the other interface.",
            "Look for the pattern: reconnects at the same time each day point to a router or ISP job.",
        ],
        docs="device stalled or offline network",
        support=False,
    ),
    "high_latency": dict(
        area="connectivity",
        cause="Round trips to Admiral are slow (weak Wi-Fi, congested uplink, long-distance link).",
        steps=["Move closer to the access point or use Ethernet; check uplink saturation."],
        docs="device stalled or offline network",
        support=False,
    ),
    "dns_failure": dict(
        area="connectivity",
        cause="The device cannot resolve names. Typical: the DHCP-supplied DNS server is down or blocks external names, or a captive portal intercepts DNS.",
        steps=[
            "Check the DNS server your DHCP hands out answers external names (from another host on the same VLAN).",
            "If the network has a login (captive) portal, exempt the device or use a static DNS such as the site resolver.",
            "Set a static DNS server in the device's network settings if DHCP DNS is unreliable.",
        ],
        docs="network configuration DNS static IP",
        support=False,
    ),
    "backend_unreachable": dict(
        area="connectivity",
        cause="The device cannot reach Admiral over TCP or NATS: a firewall, proxy or captive portal is blocking the outbound connection.",
        steps=[
            "Allow outbound traffic from the device to Admiral (at minimum HTTPS 443 for the WebSocket fallback).",
            "If an HTTP proxy is required, set it in the device's network settings.",
            "Check for a captive portal: open a browser on the same network and see whether a login page appears.",
        ],
        docs="device stalled or offline network requirements",
        support=False,
    ),
    "tls_error": dict(
        area="connectivity",
        cause="TLS/certificate errors: a wrong device clock, or a proxy that re-signs HTTPS traffic.",
        steps=[
            "Check the device clock (check_device_connectivity shows the offset).",
            "Exempt the device from TLS inspection.",
        ],
        docs="network proxy certificate time sync",
        support=False,
    ),
    "jetstream_disconnected": dict(
        area="connectivity",
        cause="The device is online but its telemetry stream is not publishing, so metrics and logs will lag.",
        steps=["Usually clears with the next reconnect; if it persists check the network for UDP/long-lived TCP drops."],
        docs="telemetry metrics logs device",
        support=False,
    ),
    "clock_skew": dict(
        area="time",
        cause="The device clock is too far from real time; Admiral authentication and TLS reject clocks outside the allowed skew.",
        steps=[
            "Allow outbound NTP (UDP 123) from the device, or point it at your site NTP server.",
            "Devices without a battery-backed clock start wrong after power loss and fix themselves once NTP is reachable.",
        ],
        docs="time synchronisation NTP clock",
        support=False,
    ),
    "time_not_synced": dict(
        area="time",
        cause="The device has not synchronised its clock (NTP blocked or unreachable).",
        steps=["Allow outbound NTP (UDP 123) or set a reachable NTP server; check again in a few minutes."],
        docs="time synchronisation NTP clock",
        support=False,
    ),
    "rtc_drift": dict(
        area="time",
        cause="The hardware clock differs from the system clock (flat RTC battery or never set).",
        steps=["Let NTP sync; if the drift returns after every power cycle, replace the RTC battery."],
        docs="time synchronisation NTP clock",
        support=False,
    ),
    "disk_level": dict(
        area="storage",
        cause="A filesystem the device depends on is nearly full (logs, workload data, image layers).",
        steps=[
            "Find what is writing: get_device_stats shows per-partition use; check workload logging and data volumes.",
            "Rotate or delete workload data, or pick a hardware model with a larger disk.",
            "Remove old configuration versions' images by applying a new configuration (unused layers are pruned).",
        ],
        docs="storage disk volumes full",
        support=False,
    ),
    "mount_mode": dict(
        area="storage",
        cause="A filesystem is mounted read-only (or read-write) against expectation; usually after storage errors or a failing disk/SD card.",
        steps=[
            "Reboot once with reboot_device (only if the user agrees); if it returns read-only the media is failing.",
            "Replace the SD card/disk if it repeats.",
        ],
        docs="storage disk health",
        support=True,
        support_note="Repeated read-only remounts mean failing media or filesystem damage.",
    ),
    "memory_pressure": dict(
        area="resources",
        cause="Memory use is very high; the OOM killer may start killing the workload.",
        steps=["Check the memory trend (get_device_metrics metric=memory) and the workload's memory limit."],
        docs="workload out of memory limits",
        support=False,
    ),
    "metric_health": dict(
        area="resources",
        cause="The device's own health metrics are flagging a problem (CPU, memory, disk or temperature).",
        steps=["get_device_stats shows which gauge is over its threshold."],
        docs="device operations monitoring health",
        support=False,
    ),
    "pending_trial": dict(
        area="update",
        cause="A system update is on trial: it will be confirmed after a clean boot or rolled back.",
        steps=["Wait for the trial to finish; if the device keeps rebooting it will roll back by itself."],
        docs="system updates trial rollback",
        support=False,
    ),
    "failed_updates": dict(
        area="update",
        cause="One or more updates failed on this device and are blacklisted.",
        steps=["Read the version and reason in the evidence; publish a fixed version instead of retrying the same one."],
        docs="system updates failed rollback",
        support=True,
        support_note="Ask Admiral if the same system update fails on several devices.",
    ),
    "version_floor_breach": dict(
        area="update",
        cause="The device runs software older than the minimum version your policy requires.",
        steps=["Roll out the required version (create_rollout) or lower the floor in the policy."],
        docs="system updates version policy",
        support=False,
    ),
    "config_drift": dict(
        area="drift",
        cause="The device is not running the configuration assigned to it (not yet applied, still downloading, staged for a window, or refused).",
        steps=[
            "If it is offline it cannot pick the change up: fix connectivity first.",
            "If a transition is shown, wait for it; otherwise render the pending change with render_device_document (dry run) and read the diff.",
            "Check the progress operations and the 'workload_rollback' finding if the device refused the version.",
        ],
        docs="rollouts convergence desired state device document",
        support=False,
    ),
    "local_override": dict(
        area="drift",
        cause="Someone changed settings on the device itself (screen, Bluetooth or console) and those override what the dashboard assigns.",
        steps=[
            "If the change is wanted, adopt it into the device document (adopt_local_override).",
            "If not, discard it (discard_local_override); beware: discarding network changes can cut the device off.",
        ],
        docs="device document local override",
        support=False,
    ),
    "service_flapping": dict(
        area="system",
        cause="A system service on the device is crash-looping or unstable.",
        steps=["get_device_system_services shows exit codes/signals for the last exits of each service."],
        docs="device operations monitoring system services",
        support=True,
        support_note="A crash-looping system service (not your workload) is an Admiral OS problem.",
    ),
    "no_observed_state": dict(
        area="drift",
        cause="The device has never reported state: it has not connected since pairing, or it runs old firmware (protocol 0).",
        steps=["Make sure it is online and up to date; old firmware reports through the workload report only."],
        docs="pairing your device",
        support=False,
    ),
    "telemetry_gated": dict(
        area="logs",
        cause="Logs and metrics need the Telemetry add-on on this organisation, so log evidence is unavailable.",
        steps=["Enable the Telemetry add-on to see log evidence; everything else in this report still applies."],
        docs="telemetry add-on logs metrics",
        support=False,
    ),
}

_AREA_ORDER = ["connectivity", "workload", "time", "storage", "update", "drift", "boot", "system", "resources", "logs", "events", "probe"]

# symptom keyword -> areas that become "relevant to your symptom"
_SYMPTOM_AREAS = [
    (("offline", "connect", "network", "wifi", "wi-fi", "internet", "dns", "unreachable", "cloud"), {"connectivity", "time"}),
    (("crash", "restart", "reboot", "loop", "exit", "oom", "memory", "app", "container", "workload", "start"), {"workload", "boot", "resources"}),
    (("screen", "black", "display", "blank", "video", "hdmi"), {"workload", "system"}),
    (("disk", "storage", "full", "space", "read-only", "readonly"), {"storage"}),
    (("update", "upgrade", "stuck", "version", "rollout", "deploy", "config", "pull", "download", "image"), {"update", "drift", "workload"}),
    (("clock", "time", "date", "ntp"), {"time"}),
    (("slow", "cpu", "hot", "temperature", "lag"), {"resources", "workload"}),
]

# ------------------------------------------------------------------ helpers ---

_AUTH_RE = re.compile(r"(?i)\bauthorization\s*[:=]\s*(?:bearer|basic|token)?\s*\S+")
_SECRET_RE = re.compile(
    r"(?i)\b(token|secret|password|passwd|psk|passphrase|api[-_]?key|cmdline)\b(\s*[=:]\s*|\s+)\S+"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")


def scrub(text: Any, limit: int = 160) -> str:
    """One quoted evidence line: secrets masked, whitespace collapsed, length bounded."""
    s = re.sub(r"\s+", " ", str(text if text is not None else "")).strip()
    s = _AUTH_RE.sub("Authorization=[redacted]", s)
    s = _BEARER_RE.sub("Bearer [redacted]", s)
    s = _SECRET_RE.sub(lambda m: f"{m.group(1)}=[redacted]", s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: Any) -> datetime | None:
    if not ts or not isinstance(ts, str) or ts.startswith("0001-"):
        return None
    try:
        return parse_rfc3339(ts)
    except Exception:  # noqa: BLE001 - tolerate odd timestamps from edges
        return None


def ago(ts: Any, now: datetime | None = None) -> str | None:
    when = _parse(ts)
    if when is None:
        return None
    secs = int(((now or _now()) - when).total_seconds())
    if secs < 0:
        return "in the future"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit} ago"
    return f"{secs}s ago"


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _ok_status(exc: AdmiralAPIError) -> str:
    if exc.status == 402:
        return "needs the Telemetry add-on (402)"
    if exc.status in (503, 504):
        return "device did not answer (offline or timed out)"
    if exc.status == 429:
        return "rate limited (one live call per 5 s per device)"
    if exc.status == 404:
        return "not available (404)"
    if exc.status in (401, 403):
        return f"not permitted for this user/token ({exc.status})"
    return f"{exc.status} {exc.message}"[:120]


# --------------------------------------------------------------- findings ---


def finding(
    code: str,
    severity: str,
    text: str,
    evidence: Iterable[Any] = (),
    *,
    area: str | None = None,
    cause: str | None = None,
    steps: list[str] | None = None,
    support: bool | None = None,
    support_note: str | None = None,
) -> dict[str, Any]:
    kb = KB.get(code, {})
    return {
        "code": code,
        "severity": severity if severity in _SEV_RANK else "info",
        "area": area or kb.get("area", "other"),
        "finding": text,
        "evidence": [scrub(e) for e in evidence if e not in (None, "")][:5],
        "likely_cause": cause or kb.get("cause", ""),
        "what_you_can_do": list(steps if steps is not None else kb.get("steps", [])),
        "docs": None,
        "needs_admiral_support": bool(kb.get("support", False) if support is None else support),
        "support_note": support_note if support_note is not None else kb.get("support_note"),
        "_docs_query": kb.get("docs"),
    }


def merge_findings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse findings with the same code: keep the first text, union evidence, take the worst severity."""
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        have = merged.get(item["code"])
        if have is None:
            merged[item["code"]] = item
            continue
        for ev in item["evidence"]:
            if ev not in have["evidence"] and len(have["evidence"]) < 5:
                have["evidence"].append(ev)
        if _SEV_RANK[item["severity"]] < _SEV_RANK[have["severity"]]:
            have["severity"] = item["severity"]
        have["needs_admiral_support"] = have["needs_admiral_support"] or item["needs_admiral_support"]
        if item["what_you_can_do"] and item["what_you_can_do"][0] not in have["what_you_can_do"]:
            have["what_you_can_do"].insert(0, item["what_you_can_do"][0])
    return list(merged.values())


def symptom_areas(symptom: str | None) -> set[str]:
    text = (symptom or "").lower()
    out: set[str] = set()
    for words, areas in _SYMPTOM_AREAS:
        if any(w in text for w in words):
            out |= areas
    return out


def rank(findings: list[dict[str, Any]], symptom: str | None = None) -> list[dict[str, Any]]:
    wanted = symptom_areas(symptom)

    def key(f: dict[str, Any]) -> tuple[int, int, int, str]:
        area = f["area"]
        pos = _AREA_ORDER.index(area) if area in _AREA_ORDER else len(_AREA_ORDER)
        return (_SEV_RANK[f["severity"]], 0 if area in wanted else 1, pos, f["code"])

    for f in findings:
        if wanted:
            f["relevant_to_symptom"] = f["area"] in wanted
    return sorted(findings, key=key)


def attach_docs(
    findings: list[dict[str, Any]],
    search: Callable[[str], list[dict[str, Any]]] | None = None,
) -> str | None:
    """Give every finding its best docs.admrl.co link via ``search_docs`` (never fails the report).

    Returns a note when the docs index could not be read. Queries are cached per call.
    """
    if search is None:
        from . import docs

        def search(q: str) -> list[dict[str, Any]]:  # noqa: F811 - default searcher
            return docs.search(q, limit=1)

    cache: dict[str, dict[str, str] | None] = {}
    note: str | None = None
    for f in findings:
        query = f.pop("_docs_query", None) or f["finding"]
        if query not in cache:
            try:
                hits = search(query) if note is None else []
            except Exception as exc:  # noqa: BLE001 - docs are optional
                note = f"docs lookup unavailable: {scrub(exc, 100)}"
                hits = []
            cache[query] = {"title": hits[0]["title"], "url": hits[0]["url"]} if hits else None
        f["docs"] = cache[query]
    for f in findings:
        f.pop("_docs_query", None)
    return note


# ----------------------------------------------------------------- probe ---

_probe_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_probe_cache() -> None:
    _probe_cache.clear()


def run_probe(
    client: Any, device_id: str, org_id: str, sections: list[str] | None = None
) -> tuple[dict[str, Any] | None, str]:
    """Run the probe; on 429 reuse the last result for this device and say so.

    Returns ``(report or None, note)``.
    """
    try:
        report = client.probe_device(device_id, org_id=org_id, sections=sections)
    except AdmiralAPIError as exc:
        if exc.status == 429:
            cached = _probe_cache.get(device_id)
            if cached:
                age = int(time.time() - cached[0])
                return cached[1], (
                    f"probe rate-limited (one per {PROBE_RATE_LIMIT_S} s per device, shared by every caller); "
                    f"using the previous result from {age}s ago"
                )
            return None, f"probe rate-limited (one per {PROBE_RATE_LIMIT_S} s per device) and no earlier result is available; retry in a few seconds"
        return None, f"probe unavailable: {_ok_status(exc)}"
    if isinstance(report, dict):
        _probe_cache[device_id] = (time.time(), report)
        return report, "fresh probe"
    return None, "probe returned no report"


def probe_items(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten probe sections to items, dropping anything the backend marked sensitive/redacted."""
    out: list[dict[str, Any]] = []
    for section in (report or {}).get("sections") or []:
        if not isinstance(section, dict):
            continue
        for item in section.get("items") or []:
            if not isinstance(item, dict) or item.get("sensitive") or item.get("value") == "[redacted]":
                continue
            out.append(
                {
                    "section": section.get("key") or section.get("title"),
                    "key": item.get("key") or "",
                    "label": item.get("label") or "",
                    "value": item.get("value"),
                    "level": (item.get("level") or "info").lower(),
                    "hint": item.get("hint"),
                }
            )
    return out


def classify_item(item: dict[str, Any]) -> str:
    """dns | nats | ntp | tcp | gateway | link | other. Item keys are device-defined, so match loosely."""
    text = f"{item.get('key', '')} {item.get('label', '')}".lower()
    for name, needles in (
        ("dns", ("dns", "resolv")),
        ("nats", ("nats", "backend", "mesh", "transport", "websocket", "wss")),
        ("ntp", ("ntp", "chrony", "clock", "time sync", "skew", "drift")),
        ("tcp", ("tcp", "latency", "ping", "rtt")),
        ("gateway", ("gateway", "route")),
        ("link", ("wifi", "wlan", "link", "carrier", "ethernet", "dhcp", "interface", "address")),
    ):
        if any(n in text for n in needles):
            return name
    return "other"


def _item_line(item: dict[str, Any]) -> str:
    return f"{item['label'] or item['key']}: {item['value']}" + (f" [{item['level']}]" if item["level"] not in ("ok", "info") else "")


# -------------------------------------------------------------- gathering ---


def gather(
    client: Any,
    org_id: str,
    device: dict[str, Any],
    *,
    include: set[str],
    lookback_hours: float = 24,
    run_probe_flag: bool = True,
    probe_sections: list[str] | None = None,
    pmap: Callable[[Any, list[Any]], list[Any]] | None = None,
) -> dict[str, Any]:
    """Fetch the evidence named in ``include``. Failures never raise: they land in ``unavailable``.

    ``include`` ⊆ {detail, diagnose, state, workload, events, logs, document, stats, live, probe, services}.
    The cheap stored reads run first (parallel); live reads (live state, probe, system services, stats)
    only if the device is online.
    """
    device_id = device["id"]
    unavailable: dict[str, str] = {}
    ev: dict[str, Any] = {"device": device, "unavailable": unavailable, "checked": {}}
    now = _now()
    start = (now - _td(lookback_hours)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def attempt(name: str, fn: Callable[[], Any], *, soft_statuses: tuple[int, ...] = ()) -> Any:
        try:
            return fn()
        except AdmiralAPIError as exc:
            if exc.status in soft_statuses:
                return None
            unavailable[name] = _ok_status(exc)
            return None

    def t_detail() -> Any:
        return attempt("detail", lambda: client.get_device(device_id, org_id=org_id))

    def t_diagnose() -> Any:
        return attempt("diagnose", lambda: client.diagnose(device_id, org_id=org_id))

    def t_state() -> Any:
        return attempt("state", lambda: client.get_device_state(device_id, org_id=org_id))

    def t_workload() -> Any:
        return attempt("workload", lambda: client.get_device_workload(device_id, org_id=org_id))

    def t_events() -> Any:
        return attempt("events", lambda: client.get_device_events(device_id, org_id=org_id, start=start, end=end, limit=100))

    def t_logs() -> Any:
        return attempt(
            "logs",
            lambda: client.fetch_logs(device_id=device_id, org_id=org_id, start=start, end=end, limit=500),
        )

    def t_document() -> Any:
        def go() -> Any:
            response = client.get_device_document(device_id, org_id=org_id, fmt="json")
            try:
                return response.json()
            except ValueError:
                return None

        return attempt("document", go)

    phase1 = [n for n in ("detail", "diagnose", "state", "workload", "events", "logs", "document") if n in include]
    runners = {
        "detail": t_detail,
        "diagnose": t_diagnose,
        "state": t_state,
        "workload": t_workload,
        "events": t_events,
        "logs": t_logs,
        "document": t_document,
    }
    run = pmap or (lambda fn, items: [fn(i) for i in items])
    results = run(lambda name: runners[name](), phase1)
    for name, value in zip(phase1, results):
        ev[name] = value
        ev["checked"][name] = "unavailable" if name in unavailable else "ok"

    diag = ev.get("diagnose") if isinstance(ev.get("diagnose"), dict) else None
    status = str(device.get("status") or "").lower()
    if diag is not None and isinstance(_dig(diag, "verdict"), dict) and "online" in diag["verdict"]:
        online = bool(diag["verdict"]["online"])
    else:
        online = status == "online"
    ev["online"] = online
    state_payload = ev.get("state") if isinstance(ev.get("state"), dict) else {}
    ev["state_doc"] = state_payload.get("state") if isinstance(state_payload.get("state"), dict) else None
    ev["state_source"] = "stored" if ev["state_doc"] else None
    ev["state_received_at"] = state_payload.get("receivedAt")
    ev["protocol"] = state_payload.get("protocol")

    live_tasks: list[tuple[str, Callable[[], Any]]] = []
    if online:
        if "live" in include:
            live_tasks.append(
                ("live", lambda: attempt("live_state", lambda: client.get_device_state(device_id, org_id=org_id, live=True)))
            )
        if "probe" in include and run_probe_flag:
            live_tasks.append(("probe", lambda: run_probe(client, device_id, org_id, probe_sections)))
        if "services" in include:
            live_tasks.append(
                ("services", lambda: attempt("system_services", lambda: client.get_device_system_services(device_id, org_id=org_id)))
            )
        if "stats" in include:
            live_tasks.append(("stats", lambda: attempt("stats", lambda: client.get_device_stats(device_id, org_id=org_id))))
    elif "stats" in include:
        live_tasks.append(("stats", lambda: attempt("stats", lambda: client.get_device_stats(device_id, org_id=org_id))))
    if not online and "probe" in include:
        ev["probe_note"] = "device is offline: the live probe, live state and system services need an online device"

    outs = run(lambda pair: pair[1](), live_tasks)
    for (name, _), value in zip(live_tasks, outs):
        if name == "live":
            if isinstance(value, dict) and isinstance(value.get("state"), dict):
                ev["state_doc"] = value["state"]
                ev["state_source"] = "live"
                ev["state_received_at"] = value.get("receivedAt")
                ev["protocol"] = value.get("protocol", ev["protocol"])
                unavailable.pop("live_state", None)
            elif "live_state" in unavailable:
                ev["live_note"] = f"live state unavailable ({unavailable.pop('live_state')}); using the stored copy"
        elif name == "probe":
            report, note = value
            ev["probe"], ev["probe_note"] = report, note
        else:
            ev[name] = value
            ev["checked"][name] = "unavailable" if name in unavailable or (name == "system_services" and name in unavailable) else "ok"
    if ev.get("state_doc") is not None:
        ev["checked"]["state"] = ev["state_source"] or "stored"

    # Desired-vs-observed: only render (dry run, nothing is pushed) when drift is already suspected.
    if "document" in include and _drift_suspected(ev) and "render" not in unavailable:
        ev["render"] = attempt(
            "render", lambda: client.render_device_document(device_id, org_id=org_id, dry_run=True)
        )
    return ev


def _td(hours: float) -> Any:
    from datetime import timedelta

    return timedelta(hours=hours)


def _document_generation(doc: Any) -> int | None:
    gen = _dig(doc, "metadata", "generation")
    return gen if isinstance(gen, int) else None


def _drift_suspected(ev: dict[str, Any]) -> bool:
    st = ev.get("state_doc") or {}
    cond = _condition(st, "Converged")
    if cond and cond.get("status") == "False":
        return True
    gen = _document_generation(ev.get("document"))
    obs = st.get("observedGeneration")
    return bool(gen and isinstance(obs, int) and obs and gen > obs)


def _condition(state: dict[str, Any] | None, ctype: str) -> dict[str, Any] | None:
    for c in (state or {}).get("conditions") or []:
        if isinstance(c, dict) and c.get("type") == ctype:
            return c
    return None


# --------------------------------------------------------------- analysis ---


def _from_diagnosis(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    diag = ev.get("diagnose")
    if not isinstance(diag, dict):
        return out
    for issue in diag.get("issues") or []:
        if not isinstance(issue, dict) or not issue.get("code"):
            continue
        code = issue["code"]
        extra = [issue.get("detail")] + list(issue.get("evidence") or [])[:3]
        out.append(finding(code, issue.get("severity") or "info", issue.get("title") or code, extra))
    return out


def _offline(ev: dict[str, Any]) -> list[dict[str, Any]]:
    if ev["online"]:
        return []
    dev, diag = ev["device"], ev.get("diagnose") or {}
    now = _now()
    evid = []
    seen = dev.get("last_seen") or _dig(diag, "signals", "lastSeenAt") or _dig(diag, "schedule", "lastCheckInAt")
    evid.append(f"last seen {seen} ({ago(seen, now)})" if seen else "never seen online")
    sched = diag.get("schedule") if isinstance(diag.get("schedule"), dict) else {}
    if sched.get("restart"):
        r = sched["restart"]
        evid.append(f"device will restart itself at {r.get('at')} ({r.get('reason')}), then every {int(r.get('thenEveryMs', 0) / 60000)} min until it reconnects")
    st = ev.get("state_doc") or {}
    tr = st.get("transport") or _dig(diag, "signals", "transport") or {}
    cause_bits = []
    if tr.get("kind") == "wss":
        evid.append("last known transport: WebSocket fallback (direct TCP/UDP was blocked)")
        cause_bits.append("it was already on the 443 fallback, so a firewall/proxy was blocking direct traffic")
    for br in (st.get("bootReasons") or [])[:1]:
        if br.get("kind") == "watchdog_no_backend":
            cause_bits.append("its last reboot was the 1 h no-backend watchdog, so it had been unable to reach Admiral for a while")
    f = finding("device_offline", "critical", f"{ev['device'].get('name') or 'Device'} is offline", evid)
    if cause_bits:
        f["likely_cause"] = "; ".join(cause_bits) + ". " + f["likely_cause"]
    if not seen:
        f["likely_cause"] = "The device has never connected: pairing was not completed or it has never had a working network. " + f["likely_cause"]
    return [f]


def _workload(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    st = ev.get("state_doc") or {}
    wl = st.get("workload") or {}
    rt = st.get("runtime") or {}
    stored = _dig(ev.get("workload"), "latestReport", "status") or {}
    state = str(wl.get("state") or stored.get("state") or _dig(ev.get("diagnose"), "verdict", "workloadState") or "").upper()
    err = wl.get("error") or stored.get("error")
    evid: list[Any] = []
    if state:
        evid.append(f"workload state {state}")
    if rt.get("failureCount"):
        evid.append(f"{rt['failureCount']} failures since start")
    if rt.get("exitCode") is not None:
        evid.append(f"last exit code {rt['exitCode']}: {_EXIT_MEANING.get(rt['exitCode'], 'application-defined')}")
    if rt.get("lastExitAt") and not str(rt["lastExitAt"]).startswith("0001"):
        evid.append(f"last exit {ago(rt['lastExitAt'])}")
    if err:
        evid.append(f"error: {err}")
    if state == "CRASHING":
        out.append(finding("workload_crashing", "critical", "The workload is crash-looping", evid))
    elif state == "ERROR" or (err and state not in ("RUNNING", "")):
        out.append(finding("workload_error", "critical", "The workload is in an error state", evid))
    elif state in ("STOPPED", "EXITED"):
        out.append(
            finding(
                "workload_error",
                "warning",
                f"The workload is {state.lower()}",
                evid + ["Stopped on purpose? start it with change_workload_status"],
                cause="The workload is not running. Either it was stopped (dashboard, command) or it exited and was not restarted.",
                steps=["If it should be running, start it with change_workload_status action=start (only with the user's go-ahead).", "If it exited by itself, run explain_workload_failure."],
            )
        )
    if rt.get("oomKilled") or (rt.get("oomEvents") or 0) > 0:
        out.append(
            finding(
                "workload_oom",
                "critical",
                "The workload was killed for running out of memory",
                [f"oomKilled={rt.get('oomKilled')}, oomEvents={rt.get('oomEvents')}"] + ([f"exit code {rt['exitCode']}"] if rt.get("exitCode") == 137 else []),
            )
        )
    rb = wl.get("rollback")
    if isinstance(rb, dict):
        out.append(
            finding(
                "workload_rollback",
                "warning",
                "The device rolled back to the previous configuration",
                [f"reason: {rb.get('reason')}", f"{rb.get('failures')} failures of v{rb.get('fromVersion')} ({rb.get('fromConfigurationName')}); fell back to v{rb.get('toVersion')}"],
            )
        )
    sig = wl.get("signature")
    if isinstance(sig, dict) and sig.get("required") and (not sig.get("verified") or sig.get("error")):
        out.append(
            finding(
                "image_signature",
                "critical",
                "The image failed signature verification",
                [f"signature error: {sig.get('error') or 'not verified'}", f"digest {sig.get('digest')}" if sig.get("digest") else None],
            )
        )
    for op in ((st.get("progress") or {}).get("operations") or []):
        if not isinstance(op, dict):
            continue
        if op.get("phase") == "failed" or op.get("error"):
            kind = str(op.get("kind") or "")
            code = "image_pull_failed" if kind in ("image_pull", "image_extract", "rootfs_download") else "stuck_transition"
            out.append(
                finding(
                    code,
                    "critical" if code == "image_pull_failed" else "warning",
                    f"{kind or 'An operation'} failed" + (f" for {scrub(op.get('target'), 80)}" if op.get("target") else ""),
                    [f"{kind}: {op.get('error') or op.get('message')}", f"attempt {op['attempt']}" if op.get("attempt") else None],
                )
            )
        elif op.get("phase") == "paused" and op.get("kind") == "staged_waiting_window":
            out.append(
                finding(
                    "stuck_transition",
                    "info",
                    "An update is staged and waiting for the update window",
                    [op.get("message") or "waiting for the maintenance window"],
                    cause="The new configuration is downloaded and prepared; it applies when the maintenance window opens.",
                    steps=["Nothing is wrong. To apply sooner, change the maintenance window or start a rollout."],
                    support=False,
                )
            )
    return out


def _storage(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    st = ev.get("state_doc") or {}
    fss = st.get("storage") or _dig(ev.get("diagnose"), "signals", "storage") or []
    for fs in fss:
        if not isinstance(fs, dict):
            continue
        total, used = fs.get("totalBytes") or 0, fs.get("usedBytes") or 0
        pct = round(100.0 * used / total, 1) if total else None
        mount = fs.get("mount")
        line = f"{mount}: {pct}% used ({_gb(used)} of {_gb(total)})" if pct is not None else f"{mount}: level {fs.get('level')}"
        if fs.get("level") == "error" or (pct is not None and pct >= 97):
            out.append(finding("disk_level", "critical", f"{mount} is almost full" if pct and pct >= 90 else f"{mount} reports a storage error", [line]))
        elif fs.get("level") == "warn" or (pct is not None and pct >= 90):
            out.append(finding("disk_level", "warning", f"{mount} is filling up", [line]))
        exp = fs.get("expected")
        if exp and ((exp == "rw" and fs.get("readOnly")) or (exp == "ro" and not fs.get("readOnly"))):
            out.append(finding("mount_mode", "critical", f"{mount} is mounted {'read-only' if fs.get('readOnly') else 'read-write'}, expected {exp}", [line]))
    stats = health_from_stats(ev.get("stats"))
    if stats:
        if isinstance(stats.get("memory"), (int, float)) and stats["memory"] >= 92:
            out.append(finding("memory_pressure", "warning", "Memory use is very high", [f"memory {stats['memory']}% used"]))
        if isinstance(stats.get("disk"), (int, float)) and stats["disk"] >= 90 and not any(f["code"] == "disk_level" for f in out):
            out.append(finding("disk_level", "warning", "Disk use is high", [f"disk {stats['disk']}% used"]))
        for issue in (stats.get("issues") or [])[:3]:
            out.append(finding("metric_health", "warning", "Device health metrics report a problem", [issue if isinstance(issue, str) else str(issue)]))
    return out


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _time(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t = (ev.get("state_doc") or {}).get("time") or _dig(ev.get("diagnose"), "signals", "time")
    if not isinstance(t, dict):
        return out
    budget = t.get("authSkewBudgetMs") or 0
    offset = t.get("offsetMs") or 0
    synced = t.get("synced")
    if budget and abs(offset) > budget:
        out.append(finding("clock_skew", "critical", "The device clock is too far off for Admiral to accept it", [f"clock offset {int(offset)} ms; authentication tolerates {budget} ms"]))
    elif synced is False:
        out.append(finding("time_not_synced", "warning", "The device clock is not synchronised", [f"synced=false, stratum {t.get('stratum')}", f"offset {int(offset)} ms" if offset else None]))
    if abs(t.get("rtcDriftMs") or 0) > 2000:
        out.append(finding("rtc_drift", "warning", "The hardware clock disagrees with the system clock", [f"RTC drift {t['rtcDriftMs']} ms"]))
    return out


def _transport(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tr = (ev.get("state_doc") or {}).get("transport") or _dig(ev.get("diagnose"), "signals", "transport")
    if not isinstance(tr, dict) or not ev["online"]:
        return out
    if tr.get("kind") == "wss":
        out.append(finding("wss_fallback", "warning", "Connected over the WebSocket fallback", [f"transport {tr['kind']}"]))
    if (tr.get("instability") or 0) > 0 or (tr.get("reconnects") or 0) >= 5:
        out.append(finding("degraded_link", "warning", "The connection to Admiral keeps dropping", [f"{tr.get('reconnects')} reconnects, instability {tr.get('instability')}"]))
    if (tr.get("latencyMs") or 0) >= 500:
        out.append(finding("high_latency", "warning", "Round trips to Admiral are slow", [f"latency {tr['latencyMs']} ms"]))
    if tr.get("jetStreamConnected") is False:
        out.append(finding("jetstream_disconnected", "info", "Telemetry is not publishing", ["jetStreamConnected=false"]))
    return out


def _drift(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    st = ev.get("state_doc")
    if st is None:
        if str(ev.get("unavailable", {}).get("state", "")).startswith("not available"):
            out.append(finding("no_observed_state", "info", "The device has not reported its state yet", []))
        return out
    evid: list[Any] = []
    cond = _condition(st, "Converged")
    if cond and cond.get("status") == "False":
        evid.append(f"Converged=False ({cond.get('reason') or cond.get('message') or 'no reason given'})")
    gen = _document_generation(ev.get("document"))
    obs = st.get("observedGeneration")
    if gen and isinstance(obs, int) and obs and gen > obs:
        evid.append(f"assigned (desired) generation {gen}, device has applied {obs}")
    rev = _dig(ev.get("document"), "status", "renderedRevision")
    if (st.get("renderedRevision") and rev and st["renderedRevision"] != rev) and not evid:
        evid.append("assigned bundle revision differs from the one the device applied")
    if (st.get("workload") or {}).get("transition"):
        t = st["workload"]["transition"]
        evid.append(f"transition {t.get('state')}: {t.get('message')}")
    rt = st.get("runtime") or {}
    if rt and rt.get("assignedConfigApplied") is False:
        evid.append("the assigned workload configuration is not the one running")
    diff = (ev.get("render") or {}).get("diff") if isinstance(ev.get("render"), dict) else None
    if diff:
        evid.append("pending changes: " + ", ".join(str(d.get("path")) for d in diff[:6] if isinstance(d, dict)))
    if evid:
        f = finding("config_drift", "warning", "The device is not running what is assigned to it", evid)
        if not ev["online"]:
            f["likely_cause"] = "The device is offline, so it cannot receive or apply the assigned change. " + f["likely_cause"]
        out.append(f)
    lo = st.get("localOverride")
    if isinstance(lo, dict):
        out.append(finding("local_override", "info", "Settings were changed on the device itself", [f"source {lo.get('source')}, paths {', '.join(lo.get('paths') or [])}", f"edited {ago(lo.get('editedAt'))}" if lo.get("editedAt") else None]))
    br = (st.get("bootReasons") or [None])[0]
    if isinstance(br, dict) and br.get("kind") in _UNEXPECTED_BOOT_KINDS and ago(br.get("at")) and _within(br.get("at"), 24 * 3600):
        kind = br["kind"]
        f = finding("unexpected_reboot", "warning", f"The device rebooted unexpectedly ({kind})", [f"{br.get('at')} ({ago(br.get('at'))}), previous boot lasted {br.get('uptimeSeconds')}s", br.get("detail")])
        f["likely_cause"] = _UNEXPECTED_BOOT_KINDS[kind]
        if kind in ("kernel_fatal", "agent_fatal", "unclean"):
            f["needs_admiral_support"] = kind != "unclean"
        out.append(f)
    return out


def _within(ts: Any, seconds: float) -> bool:
    when = _parse(ts)
    return when is not None and (_now() - when).total_seconds() <= seconds


def _services(ev: dict[str, Any]) -> list[dict[str, Any]]:
    report = ev.get("services")
    out: list[dict[str, Any]] = []
    for svc in (report or {}).get("services") or []:
        if not isinstance(svc, dict):
            continue
        health = svc.get("health")
        if health in ("flapping", "unstable", "restarting") or (health == "stopped" and svc.get("normallyUp", True)):
            out.append(finding("service_flapping", "warning" if health != "flapping" else "critical", f"System service {svc.get('name')} is {health}", [f"{svc.get('name')}: {health} — {svc.get('reason')}"]))
    return out


_LOG_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("usb_denied", "warning", re.compile(r"usb.*(denied|blocked|not allowed|rejected|policy)|(denied|blocked|rejected).*usb", re.I)),
    ("image_signature", "critical", re.compile(r"(signature|cosign|notation).*(fail|invalid|denied|unverified|not verified|mismatch)", re.I)),
    ("image_pull_failed", "critical", re.compile(r"(pull|manifest|registry|layer).*(fail|unauthori[sz]ed|denied|not found|timeout|refused)", re.I)),
    ("tls_error", "warning", re.compile(r"x509|certificate (has expired|is not yet valid|signed by unknown)|tls: (bad|failed)", re.I)),
    ("dns_failure", "warning", re.compile(r"no such host|dns.*(fail|timeout|refused|error)|temporary failure in name resolution|lookup .* on ", re.I)),
    ("workload_oom", "critical", re.compile(r"out of memory|oom[- ]?kill|killed process", re.I)),
    ("disk_level", "critical", re.compile(r"no space left on device", re.I)),
]


def log_entries(ev: dict[str, Any]) -> list[dict[str, Any]]:
    logs = ev.get("logs")
    return [e for e in (logs or {}).get("logs") or [] if isinstance(e, dict)]


def _logs(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    entries = log_entries(ev)
    if "logs" in ev.get("unavailable", {}):
        if "Telemetry" in ev["unavailable"]["logs"]:
            out.append(finding("telemetry_gated", "info", "Logs are not available on this plan", [ev["unavailable"]["logs"]]))
        return out
    errs = [e for e in entries if _level(e) in ("error", "fatal", "critical", "panic")]
    hits: dict[str, tuple[str, list[str]]] = {}
    for e in entries:
        msg = _message(e)
        for code, sev, pat in _LOG_PATTERNS:
            if pat.search(msg):
                have = hits.setdefault(code, (sev, []))
                if len(have[1]) < 3:
                    have[1].append(f"{_source(e) or 'log'}: {msg}")
                break
    for code, (sev, lines) in hits.items():
        out.append(finding(code, sev, KB[code]["cause"].split(".")[0] if code in KB else code, lines))
    if errs:
        dist = distill_logs({"logs": errs})
        top = [f"{s['count']}× {s['message']}" for s in dist["top_signatures"][:4]]
        out.append(finding("log_errors", "warning" if len(errs) >= 5 else "info", f"{len(errs)} error-level log lines in the window", top))
    return out


def _events(ev: dict[str, Any]) -> list[dict[str, Any]]:
    if not ev.get("events"):
        return []
    dist = distill_events(ev["events"])
    names = Counter(str(e.get("event")) for e in dist["notable"])
    if not names:
        return []
    return [finding("notable_events", "info", "Notable lifecycle events in the window", [f"{n}× {name}" for name, n in names.most_common(4)])]


def _probe(ev: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    generic: dict[str, list[dict[str, Any]]] = {}
    for item in probe_items(ev.get("probe")):
        if item["level"] not in ("warn", "error"):
            continue
        cls = classify_item(item)
        line = _item_line(item)
        steps = [item["hint"]] if item.get("hint") else None
        err = item["level"] == "error"
        if cls == "dns":
            f = finding("dns_failure", "critical" if err else "warning", "DNS lookups from the device are failing", [line])
        elif cls in ("nats", "tcp"):
            if cls == "tcp" and not err and re.search(r"latency|rtt|ping", f"{item['key']} {item['label']}", re.I):
                f = finding("high_latency", "warning", "Round trips to Admiral are slow", [line])
            else:
                f = finding("backend_unreachable", "critical" if err else "warning", "The device cannot reliably reach Admiral", [line])
        elif cls == "ntp":
            f = finding("clock_skew" if err else "time_not_synced", "critical" if err else "warning", "The device clock is not in sync", [line])
        else:
            generic.setdefault(str(item["section"] or "probe"), []).append(item)
            continue
        if steps:
            f["what_you_can_do"] = steps + [s for s in f["what_you_can_do"] if s not in steps]
        out.append(f)
    for section, items in generic.items():
        worst = "critical" if any(i["level"] == "error" for i in items) else "warning"
        hints = [i["hint"] for i in items if i.get("hint")]
        f = finding(
            f"probe_{section}",
            "warning" if worst == "critical" and section not in ("storage", "boot") else worst,
            f"On-device check '{section}' reports {'errors' if worst == 'critical' else 'warnings'}",
            [_item_line(i) for i in items[:4]],
            area="probe" if section not in ("storage", "network", "time") else {"storage": "storage", "network": "connectivity", "time": "time"}[section],
            cause="The device's own diagnostic flagged this item.",
            steps=hints[:3] or ["Read the quoted items; the device's hint (if any) is the first step."],
            support=False,
        )
        f["_docs_query"] = f"{section} device diagnostics troubleshooting"
        out.append(f)
    return out


ANALYSERS: list[Callable[[dict[str, Any]], list[dict[str, Any]]]] = [
    _offline,  # before the backend issues so its richer likely_cause wins the merge
    _from_diagnosis,
    _workload,
    _storage,
    _time,
    _transport,
    _drift,
    _services,
    _probe,
    _logs,
    _events,
]


def analyse(ev: dict[str, Any], *, areas: set[str] | None = None, symptom: str | None = None) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for fn in ANALYSERS:
        raw.extend(fn(ev))
    merged = merge_findings(raw)
    if areas is not None:
        merged = [f for f in merged if f["area"] in areas]
    return rank(merged, symptom)


def public(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop internal keys before returning findings."""
    return [{k: v for k, v in f.items() if not k.startswith("_")} for f in findings]


def overall(findings: list[dict[str, Any]], online: bool) -> str:
    if not online:
        return "offline"
    if any(f["severity"] == "critical" for f in findings):
        return "critical"
    if any(f["severity"] == "warning" for f in findings):
        return "degraded"
    return "healthy"


def support_verdict(findings: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = [f for f in findings if f["needs_admiral_support"]]
    if flagged:
        return {"needed": True, "reason": "; ".join(f"{f['code']}: {f['support_note'] or f['finding']}" for f in flagged[:3])}
    notes = [f["support_note"] for f in findings[:3] if f.get("support_note")]
    return {"needed": False, "if_unresolved": notes[0] if notes else "Nothing found that needs Admiral; if the steps above do not fix it, open a support ticket with this report."}


def device_header(ev: dict[str, Any]) -> dict[str, Any]:
    d = ev["device"]
    return {"id": d.get("id"), "name": d.get("name"), "status": d.get("status"), "fleet": (d.get("fleet") or {}).get("name"), "last_seen": d.get("last_seen")}


def checked_sources(ev: dict[str, Any]) -> dict[str, Any]:
    checked = dict(ev.get("checked", {}))
    if ev.get("probe_note"):
        checked["probe"] = ev["probe_note"]
    if ev.get("live_note"):
        checked["live_state"] = ev["live_note"]
    for name, why in ev.get("unavailable", {}).items():
        checked[name] = f"unavailable: {why}"
    return checked


# ------------------------------------------------------- connectivity tool ---

_CONNECTIVITY_CAUSES = [
    "Captive portal: a network that needs a web login intercepts DNS/HTTPS until someone signs in.",
    "DNS: the DHCP-provided DNS server is down or blocks external names.",
    "Firewall: outbound TCP/UDP to Admiral is blocked; only HTTPS 443 (WebSocket fallback) may be open.",
    "TLS inspection or an HTTP proxy that is not configured on the device.",
    "Clock skew: a wrong clock makes authentication and TLS fail (no NTP reachable).",
    "Weak Wi-Fi or a flaky cable/switch port: frequent reconnects.",
]


def connectivity_checks(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """The fixed check table: link, dns, tcp, ntp, nats/transport, clock."""
    st = ev.get("state_doc") or {}
    tr = st.get("transport") or _dig(ev.get("diagnose"), "signals", "transport") or {}
    t = st.get("time") or _dig(ev.get("diagnose"), "signals", "time") or {}
    items = probe_items(ev.get("probe"))
    by: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        by.setdefault(classify_item(it), []).append(it)

    def from_probe(cls: str) -> tuple[str, str]:
        rows = by.get(cls) or []
        if not rows:
            return "unknown", "not probed" if not ev.get("probe") else "no matching probe item"
        worst = "error" if any(r["level"] == "error" for r in rows) else "warn" if any(r["level"] == "warn" for r in rows) else "ok"
        return worst, "; ".join(_item_line(r) for r in rows[:3])

    checks = []
    ifaces = [i for i in (st.get("network") or {}).get("interfaces") or [] if isinstance(i, dict)]
    up = [i for i in ifaces if str(i.get("state")).upper() == "UP"]
    if ifaces:
        checks.append({"check": "link", "status": "ok" if up else "error", "detail": ", ".join(f"{i.get('name')} {i.get('state')} {','.join(a.get('address') for a in i.get('ips') or [] if isinstance(a, dict))}".strip() for i in ifaces[:5])})
    else:
        s, d = from_probe("link")
        checks.append({"check": "link", "status": s, "detail": d})
    for cls, name in (("dns", "dns"), ("tcp", "tcp"), ("ntp", "ntp"), ("nats", "backend_probe"), ("gateway", "gateway")):
        s, d = from_probe(cls)
        checks.append({"check": name, "status": s, "detail": d})
    if tr:
        bad = tr.get("kind") == "wss" or (tr.get("instability") or 0) > 0
        checks.append({"check": "transport", "status": "warn" if bad else "ok", "detail": f"{tr.get('kind')}, latency {tr.get('latencyMs')} ms, {tr.get('reconnects')} reconnects, instability {tr.get('instability')}, last backend contact {ago(tr.get('lastBackendContact')) or 'unknown'}, telemetry {'on' if tr.get('jetStreamConnected') else 'off'}"})
    else:
        checks.append({"check": "transport", "status": "unknown", "detail": "no transport data (device offline or older firmware)"})
    if t:
        budget = t.get("authSkewBudgetMs") or 0
        off = t.get("offsetMs") or 0
        status = "error" if budget and abs(off) > budget else "warn" if t.get("synced") is False else "ok"
        checks.append({"check": "clock", "status": status, "detail": f"synced={t.get('synced')}, offset {int(off)} ms (limit {budget} ms), stratum {t.get('stratum')}"})
    else:
        checks.append({"check": "clock", "status": "unknown", "detail": "no time data"})
    return checks
