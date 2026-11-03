"""Per-service view of a device workload (a set of services).

Pure helpers, no network. Agents that run a multi-service set report ``workload.services[]`` (name,
state, image, imageDigest, exitCode, restarts, health, error, startedAt) and ``runtime.services[]``
(name, containerId, pid, exitCode, oomKilled, imageDigest, crunState, failureCount, ip) in the state
document; the workload report carries the same per-service status. Older agents report neither, and
every helper here then returns ``None`` / an empty list so callers keep today's single-workload shape.
"""

from __future__ import annotations

import re
from typing import Any


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _dig(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _zero_time(value: Any) -> bool:
    return not value or str(value).startswith("0001-")


def status_entries(workload: Any = None, state_doc: Any = None) -> list[dict[str, Any]]:
    """Per-service status entries, from the workload response or (failing that) the state document."""
    candidates = (
        _dig(workload, "services"),
        _dig(workload, "set", "services"),
        _dig(workload, "latestReport", "status", "services"),
        _dig(workload, "data", "services"),
        _dig(state_doc, "workload", "services"),
    )
    for found in candidates:
        rows = _dicts(found)
        if rows:
            return rows
    return []


def runtime_entries(state_doc: Any = None) -> list[dict[str, Any]]:
    return _dicts(_dig(state_doc, "runtime", "services"))


def service_rows(workload: Any = None, state_doc: Any = None) -> list[dict[str, Any]]:
    """One merged row per service, in report order (dependency order); empty for single-workload agents.

    Row keys (only those the agent reported): name, state, image, digest, restarts, health
    {status, failing_streak, last}, ip, pid, exit_code, oom_killed, container_id, error, started_at.
    """
    status = status_entries(workload, state_doc)
    runtime = {str(r.get("name")): r for r in runtime_entries(state_doc)}
    names = [str(s.get("name")) for s in status if s.get("name")]
    names += [n for n in runtime if n not in names]
    by_status = {str(s.get("name")): s for s in status}
    rows: list[dict[str, Any]] = []
    for name in names:
        s, r = by_status.get(name, {}), runtime.get(name, {})
        health = s.get("health") if isinstance(s.get("health"), dict) else None
        row: dict[str, Any] = {
            "name": name,
            "state": s.get("state") or (str(r.get("crunState")).upper() if r.get("crunState") else None),
            "image": s.get("image"),
            "digest": s.get("imageDigest") or r.get("imageDigest"),
            "restarts": s.get("restarts") if s.get("restarts") is not None else r.get("failureCount"),
            "health": (
                {
                    "status": health.get("status"),
                    "failing_streak": health.get("failingStreak"),
                    "last": health.get("last"),
                }
                if health
                else None
            ),
            "ip": r.get("ip"),
            "pid": r.get("pid"),
            "exit_code": s.get("exitCode") if s.get("exitCode") is not None else r.get("exitCode"),
            "oom_killed": r.get("oomKilled"),
            "container_id": r.get("containerId"),
            "error": s.get("error"),
            "started_at": None if _zero_time(s.get("startedAt")) else s.get("startedAt"),
        }
        if row["health"]:
            row["health"] = {k: v for k, v in row["health"].items() if v not in (None, "")}
        rows.append({k: v for k, v in row.items() if v not in (None, "", {})})
    return rows


def set_view(workload: Any = None, state_doc: Any = None) -> dict[str, Any] | None:
    """``{set: {...}, services: [...]}`` for a multi-service report, ``None`` for a single-workload agent."""
    rows = service_rows(workload, state_doc)
    if not rows:
        return None
    wl = workload if isinstance(workload, dict) else {}
    state = (
        _dig(state_doc, "workload", "state")
        or wl.get("state")
        or _dig(wl, "latestReport", "status", "state")
        or _dig(wl, "set", "state")
    )
    running: dict[str, Any] = _dig(wl, "running") or {}
    down = [r["name"] for r in rows if str(r.get("state") or "").upper() != "RUNNING"]
    unhealthy = [r["name"] for r in rows if str((r.get("health") or {}).get("status") or "").lower() == "unhealthy"]
    set_info: dict[str, Any] = {
        "state": state,
        "configuration": running.get("configurationName") or _dig(state_doc, "workload", "configurationName"),
        "version": running.get("version") or _dig(state_doc, "workload", "version"),
        "service_count": len(rows),
        "not_running": down,
        "unhealthy": unhealthy,
    }
    return {"set": {k: v for k, v in set_info.items() if v not in (None, "", [])}, "services": rows}


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def check_service_name(service: str | None) -> str | None:
    """Normalised service name or ``None`` for the whole set; raises ValueError on a malformed name."""
    name = (service or "").strip()
    if not name:
        return None
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid service name {service!r}: lower-case letters, digits, '-' and '_', starting with a letter (at most 32 characters).")
    return name


def log_matches_service(entry: dict[str, Any], service: str) -> bool:
    """True when a log record belongs to ``service``: its ``service`` label, else the container id in
    ``workload.<instanceID>-<service>.<stream>`` (instance ids carry no ``-``; the default is ``default``)."""
    raw_labels = entry.get("labels")
    labels: dict[str, Any] = raw_labels if isinstance(raw_labels, dict) else {}
    labelled = entry.get("service") or labels.get("service")
    if labelled:
        return str(labelled) == service
    parts = str(entry.get("source") or "").split(".")
    if len(parts) >= 3 and parts[0] == "workload":
        container = ".".join(parts[1:-1])
        return container == service or container.partition("-")[2] == service
    return False
