from __future__ import annotations

import re
from collections import Counter
from typing import Any

CRASH_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bpanic\b",
        r"\bfatal\b",
        r"\bsegfault\b",
        r"\bsig(segv|abrt|kill|term)\b",
        r"\boomed\b",
        r"\bout of memory\b",
        r"killed process",
        r"workload.*(crash|exit|restart)",
        r"container.*(crash|exit|oom)",
        r"watchdog",
        r"kernel panic",
        r"unable to",
        r"failed to",
        r"error:",
        r"traceback",
        r"exception",
    )
]


def _entries(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if isinstance(payload, dict):
        for key in ("logs", "events", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [e for e in value if isinstance(e, dict)]
            if isinstance(value, dict) and isinstance(value.get("logs"), list):
                return [e for e in value["logs"] if isinstance(e, dict)]
    return []


def _level(entry: dict[str, Any]) -> str:
    return str(entry.get("level") or entry.get("Level") or "info").lower()


def _message(entry: dict[str, Any]) -> str:
    return str(entry.get("message") or entry.get("msg") or entry.get("event") or "").strip()


def _source(entry: dict[str, Any]) -> str:
    return str(entry.get("source") or "").lower()


def _event_key(name: str) -> str:
    """Normalize PascalCase / kebab event names to snake_case."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", (name or "").strip())
    return s.replace("-", "_").lower()


def _looks_crashy(entry: dict[str, Any]) -> bool:
    level = _level(entry)
    if level in {"error", "fatal", "panic", "critical"}:
        return True
    text = _message(entry)
    return any(p.search(text) for p in CRASH_PATTERNS)


def distill_logs(payload: Any, *, max_samples: int = 12) -> dict[str, Any]:
    entries = _entries(payload)
    levels = Counter(_level(e) for e in entries)
    sources = Counter(_source(e) or "unknown" for e in entries)
    crashy = [e for e in entries if _looks_crashy(e)]
    signatures: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for entry in crashy:
        msg = _message(entry)
        sig = re.sub(r"0x[0-9a-f]+", "<id>", msg, flags=re.IGNORECASE)
        sig = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", sig, flags=re.IGNORECASE)
        sig = re.sub(r"\b\d+\b", "N", sig)
        sig = re.sub(r"\s+", " ", sig)[:240]
        signatures[sig or msg[:240]] += 1
        if len(samples) < max_samples:
            samples.append(
                {
                    "timestamp": entry.get("timestamp"),
                    "level": _level(entry),
                    "source": entry.get("source"),
                    "message": msg[:500],
                }
            )

    timeline = []
    if entries:
        first = entries[0].get("timestamp")
        last = entries[-1].get("timestamp")
        timeline = {"first": first, "last": last, "count": len(entries)}

    return {
        "count": len(entries),
        "has_more": bool(isinstance(payload, dict) and (payload.get("has_more") or payload.get("hasMore"))),
        "levels": dict(levels),
        "sources": dict(sources),
        "crash_like": len(crashy),
        "top_signatures": [{"message": m, "count": c} for m, c in signatures.most_common(8)],
        "samples": samples,
        "window": timeline,
    }


def distill_events(payload: Any, *, max_samples: int = 15) -> dict[str, Any]:
    events = []
    if isinstance(payload, dict):
        raw = payload.get("events") or payload.get("data") or payload
        if isinstance(raw, dict):
            raw = raw.get("events") or []
        if isinstance(raw, list):
            events = [e for e in raw if isinstance(e, dict)]
    elif isinstance(payload, list):
        events = [e for e in payload if isinstance(e, dict)]

    kinds = Counter(str(e.get("event") or "unknown") for e in events)
    sources = Counter(str(e.get("source") or "unknown") for e in events)
    notable_exact = {
        "device_offline",
        "workload_crash",
        "workload_crashed",
        "workload_exited",
        "container_exit",
        "update_failed",
        "reboot",
    }
    interesting = []
    for e in events:
        raw = str(e.get("event") or "")
        name = _event_key(raw)
        if (
            name in notable_exact
            or "crash" in name
            or "fail" in name
            or name.endswith("_offline")
        ):
            interesting.append(e)
    return {
        "count": len(events),
        "kinds": dict(kinds),
        "sources": dict(sources),
        "notable": interesting[:max_samples],
    }


def health_from_stats(stats: Any) -> dict[str, Any] | None:
    if not isinstance(stats, dict):
        return None
    health = stats.get("health") if isinstance(stats.get("health"), dict) else stats
    if not isinstance(health, dict):
        return None
    return {
        "overall": health.get("overall_status") or health.get("status"),
        "score": health.get("health_score"),
        "issues": health.get("issues") or [],
        "warnings": health.get("warnings") or [],
        "last_metric_time": health.get("last_metric_time"),
        "cpu": (stats.get("cpu") or {}).get("usage_percent") if isinstance(stats.get("cpu"), dict) else None,
        "memory": (stats.get("memory") or {}).get("usage_percent") if isinstance(stats.get("memory"), dict) else None,
        "disk": (stats.get("disk") or {}).get("usage_percent") if isinstance(stats.get("disk"), dict) else None,
    }
