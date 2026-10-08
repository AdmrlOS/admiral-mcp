"""Fleet- and organisation-level metrics.

Pure helpers (no MCP, no threads, Pyodide-safe) behind ``get_fleet_metrics``,
``get_fleet_health``, ``get_fleet_uptime``, ``get_org_metrics``,
``query_telemetry_metrics`` and ``get_telemetry_scope``.

Backend surfaces (customer routes):

* ``GET /fleets/{id}/metrics|health|uptime|uptime/percentage`` — per-fleet.
* ``GET|POST /telemetry/metrics/{query|query_range}`` and ``GET /telemetry/scope``
  — scoped PromQL proxy. The organisation and the caller's grants are enforced
  server-side from ``X-Organization-ID``; ``scope_fleet_id`` only narrows.

Stored series are named ``edge_cpu_usagepercent`` etc. and carry ``device_id``
and ``fleet_id`` labels (they are bare names, no namespace prefix).
"""

from __future__ import annotations

import math
from typing import Any

from .client import AdmiralAPIError, AdmiralClient
from .resolve import _as_list, _norm, is_uuid

# metric id -> (stored name, unit, counter?, how several series on ONE device
# collapse: disks report one series per mountpoint, NICs one per interface)
METRICS: dict[str, tuple[str, str, bool, str]] = {
    "cpu": ("edge_cpu_usagepercent", "%", False, "avg"),
    "memory": ("edge_memory_usagepercent", "%", False, "avg"),
    "disk": ("edge_disk_usagepercent", "%", False, "max"),
    "network_rx": ("edge_network_interfaces_rxbytes", "B/s", True, "sum"),
    "network_tx": ("edge_network_interfaces_txbytes", "B/s", True, "sum"),
}
GROUPS = ("fleet", "device")
STATS = ("avg", "max")

MAX_QUERY_CHARS = 2000
MAX_RANGE_HOURS = 24 * 7
MAX_RANGE_POINTS = 500
MAX_SERIES = 25
MAX_POINTS_PER_SERIES = 60
# Hard cap on device rows pulled into one org-wide ranking.
MAX_DEVICE_ROWS = 5000

DOCS_URL = "https://docs.admrl.co"


# ------------------------------------------------------------------ gates ---


def gate_result(exc: AdmiralAPIError, what: str) -> dict[str, Any] | None:
    """Non-error payload for the billing (402) and permission (403) gates."""
    if exc.status == 402:
        return {
            "summary": (
                f"{what} needs the Telemetry add-on, which this organisation does not have. "
                "This is a billing gate, not an outage."
            ),
            "available": False,
            "status": 402,
            "reason": "telemetry_api",
            "next_step": (
                "Enable the Telemetry add-on in the organisation's billing settings, or use get_device_stats "
                "for the live snapshot of a device. Free-tier history is limited to the last 24 hours."
            ),
            "docs": DOCS_URL,
        }
    if exc.status == 403:
        return {
            "summary": f"{what} is not permitted for this user/token in this organisation.",
            "available": False,
            "status": 403,
            "next_step": (
                "Check the token's organisation (list_organisations) and that the user has telemetry access to "
                "the fleet; get_telemetry_scope shows what this caller can see."
            ),
        }
    return None


# ------------------------------------------------------------------ stats ---


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _r(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def series_stats(values: Any) -> dict[str, Any]:
    """avg/max/min/latest over ``[[ts, value], ...]`` points."""
    pts = [(p[0], _num(p[1])) for p in (values or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
    nums = [v for _, v in pts if v is not None]
    if not nums:
        return {"points": 0, "avg": None, "max": None, "min": None, "latest": None}
    return {
        "points": len(nums),
        "avg": _r(sum(nums) / len(nums)),
        "max": _r(max(nums)),
        "min": _r(min(nums)),
        "latest": _r(nums[-1]),
    }


def _mean(vals: list[float | None]) -> float | None:
    got = [v for v in vals if v is not None]
    return _r(sum(got) / len(got)) if got else None


def _max(vals: list[float | None]) -> float | None:
    got = [v for v in vals if v is not None]
    return _r(max(got)) if got else None


def rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-device rows (avg/max/latest) into one group row."""
    return {
        "devices": len(rows),
        "avg": _mean([r.get("avg") for r in rows]),
        "max": _max([r.get("max") for r in rows]),
        "latest": _mean([r.get("latest") for r in rows]),
    }


# --------------------------------------------------------------- resolve ---


def resolve_fleet(client: AdmiralClient, query: str, org_id: str) -> dict[str, Any]:
    """Fleet by id or name. Returns ``{"match": {id, name}}`` or ``{"match": None, "candidates": [...]}``."""
    q = (query or "").strip()
    if not q:
        return {"match": None, "candidates": [], "error": "fleet is required (name or id)"}
    fleets = [f for f in _as_list(client.list_fleets(org_id=org_id, limit=100)) if isinstance(f, dict)]
    if is_uuid(q):
        for f in fleets:
            if _norm(f.get("id")) == _norm(q):
                return {"match": {"id": f["id"], "name": f.get("name")}, "candidates": []}
        return {"match": {"id": q, "name": None}, "candidates": []}
    nq = _norm(q)
    exact = [f for f in fleets if _norm(f.get("name")) == nq]
    if len(exact) == 1:
        pool = exact
    else:
        pool = exact or [f for f in fleets if nq in _norm(f.get("name"))]
    cands = [{"id": f.get("id"), "name": f.get("name")} for f in pool]
    if len(cands) == 1:
        return {"match": cands[0], "candidates": []}
    return {
        "match": None,
        "candidates": cands,
        "note": (
            "Several fleets match; pass a fleet id (UUID)." if cands else "No fleet matches that name; try list_fleets."
        ),
    }


def fleet_names(client: AdmiralClient, org_id: str) -> dict[str, str]:
    try:
        fleets = _as_list(client.list_fleets(org_id=org_id, limit=100))
    except AdmiralAPIError:
        return {}
    return {str(f["id"]): str(f.get("name") or "") for f in fleets if isinstance(f, dict) and f.get("id")}


def device_names(client: AdmiralClient, org_id: str, fleet_id: str | None = None) -> dict[str, dict[str, Any]]:
    """id -> {name, fleet_id, fleet}. First page of up to 100; best effort."""
    out: dict[str, dict[str, Any]] = {}
    try:
        payload = client.list_devices(org_id=org_id, fleet_id=fleet_id, limit=100)
    except AdmiralAPIError:
        return out
    for d in _as_list(payload):
        if isinstance(d, dict) and d.get("id"):
            fl = d.get("fleet") or {}
            out[str(d["id"])] = {"name": d.get("name"), "fleet_id": fl.get("id") or d.get("fleet_id")}
    return out


# ------------------------------------------------------------------ fleet ---


def fleet_metrics(
    client: AdmiralClient,
    *,
    fleet: dict[str, Any],
    org_id: str,
    metric: str,
    start: str,
    end: str,
    per_device: bool,
    limit: int,
    device_id: str | None = None,
) -> dict[str, Any]:
    """Window stats for a fleet from ``GET /fleets/{id}/metrics?aggregate=false``.

    Always fetches per-device series so one call yields fleet avg/max/latest and
    the device count; ``per_device`` only controls whether rows are returned.
    """
    _, unit, _, _ = METRICS[metric]
    payload = client.get_fleet_metrics(
        fleet["id"], org_id=org_id, metric_name=metric, start=start, end=end, aggregate=False
    )
    series = (payload or {}).get("series") if isinstance(payload, dict) else None
    window = {"metric": metric, "unit": unit, "start": start, "end": end}
    base = {"fleet": fleet, "query": window}
    if not series:
        return {
            "summary": (
                f"No {metric} data for fleet {fleet.get('name') or fleet['id']} in this window "
                "(no devices reporting, or the window is too short). Not an error."
            ),
            **base,
            "fleet_stats": {"devices": 0, "avg": None, "max": None, "latest": None},
            "devices": [],
        }
    by_dev: dict[str, list[dict[str, Any]]] = {}
    for s in series:
        labels = s.get("labels") or {}
        by_dev.setdefault(str(labels.get("device_id") or "unknown"), []).append(series_stats(s.get("values")))
    rows = []
    for did, parts in by_dev.items():
        st = parts[0] if len(parts) == 1 else {  # several series on one device (mountpoints/NICs)
            "avg": _max([p["avg"] for p in parts]) if METRICS[metric][3] == "max" else _mean([p["avg"] for p in parts]),
            "max": _max([p["max"] for p in parts]),
            "latest": _max([p["latest"] for p in parts]) if METRICS[metric][3] == "max" else _mean([p["latest"] for p in parts]),
        }
        rows.append({"device_id": did, "avg": st["avg"], "max": st["max"], "latest": st["latest"]})
    fleet_stats = rollup(rows)
    names = device_names(client, org_id, fleet["id"])
    for r in rows:
        r["name"] = (names.get(r["device_id"]) or {}).get("name")
    rows.sort(key=lambda r: (r["avg"] is None, -(r["avg"] or 0)))
    out: dict[str, Any] = {
        "summary": (
            f"Fleet {fleet.get('name') or fleet['id']} {metric}: avg {fleet_stats['avg']}{unit}, "
            f"max {fleet_stats['max']}{unit}, latest {fleet_stats['latest']}{unit} across "
            f"{fleet_stats['devices']} device(s)."
        ),
        **base,
        "fleet_stats": fleet_stats,
    }
    if device_id:
        mine = next((r for r in rows if r["device_id"] == device_id), None)
        if mine is None:
            out["device"] = {"device_id": device_id, "note": "No data for this device in the window."}
        else:
            rank = rows.index(mine) + 1
            delta = None if mine["avg"] is None or fleet_stats["avg"] is None else _r(mine["avg"] - fleet_stats["avg"])
            out["device"] = {**mine, "rank_by_avg": rank, "of": len(rows), "avg_vs_fleet": delta}
            out["summary"] += (
                f" Device {mine.get('name') or device_id}: avg {mine['avg']}{unit} "
                f"({'+' if (delta or 0) >= 0 else ''}{delta} vs fleet), rank {rank}/{len(rows)} (1 = highest)."
            )
    if per_device:
        out["devices"] = rows[: max(1, limit)]
        if len(rows) > limit:
            out["truncated"] = f"Showing top {limit} of {len(rows)} devices by avg."
    return out


def fleet_uptime_view(buckets: Any, current: Any, fleet: dict[str, Any], period_type: str) -> dict[str, Any]:
    periods = []
    avg = None
    if isinstance(buckets, dict):
        avg = buckets.get("avg_uptime_pct")
        for p in buckets.get("periods") or []:
            if isinstance(p, dict):
                periods.append(
                    {
                        "start": p.get("period_start"),
                        "end": p.get("period_end"),
                        "uptime_pct": _r(_num(p.get("uptime_pct"))),
                    }
                )
    cur = current.get("uptime_pct") if isinstance(current, dict) else None
    name = fleet.get("name") or fleet.get("id")
    return {
        "summary": (
            f"Fleet {name} uptime: {_r(_num(cur))}% over the current window, "
            f"{_r(_num(avg))}% average over {len(periods)} {period_type} bucket(s)."
        ),
        "fleet": fleet,
        "current_uptime_pct": _r(_num(cur)),
        "average_uptime_pct": _r(_num(avg)),
        "period_type": period_type,
        "periods": periods,
        **(
            {
                "note": (
                    "Uptime is 0% in every bucket; the uptime aggregate can lag or be unpopulated. Cross-check "
                    "with list_devices(fleet_id=...) counts before reporting an outage."
                )
            }
            if periods and not (_num(cur) or _num(avg))
            else {}
        ),
    }


# -------------------------------------------------------------------- org ---


def device_expr(metric: str) -> str:
    """PromQL collapsing a metric to ONE series per device (labels device_id, fleet_id)."""
    name, _, counter, collapse = METRICS[metric]
    inner = f"rate({name}[5m])" if counter else name
    return f"{collapse} by (device_id, fleet_id) ({inner})"


def _step(hours: float) -> str:
    return "5m" if hours <= 24 else "30m" if hours <= 24 * 7 else "1h"


def window_queries(metric: str, hours: float) -> dict[str, str]:
    """Instant queries (evaluated at ``now``) for per-device window avg / max / latest.

    Only enum-validated pieces are interpolated; fleet narrowing goes through the
    server-enforced ``scope_fleet_id`` parameter, never into the query text.
    """
    h = max(1, int(math.ceil(hours)))
    sub = f"[{h}h:{_step(hours)}]"
    expr = device_expr(metric)
    return {
        "avg": f"avg_over_time(({expr}){sub})",
        "max": f"max_over_time(({expr}){sub})",
        "latest": f"last_over_time(({expr}){sub})",
    }


def _vector(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    result = data.get("result") if isinstance(data, dict) else None
    return [r for r in (result or []) if isinstance(r, dict)]


def _instant_value(row: dict[str, Any]) -> float | None:
    v = row.get("value")
    return _num(v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else None


def org_metrics(
    client: AdmiralClient,
    *,
    org_id: str,
    metric: str,
    group_by: str,
    stat: str,
    hours: float,
    limit: int,
    fleet: dict[str, Any] | None = None,
    sort: str = "desc",
) -> dict[str, Any]:
    _, unit, _, _ = METRICS[metric]
    scope = {"scope_fleet_id": fleet["id"]} if fleet else {}
    per_stat: dict[str, dict[tuple[str, str], float | None]] = {}
    for key, q in window_queries(metric, hours).items():
        payload = client.telemetry_query(q, org_id=org_id, extra=scope)
        got: dict[tuple[str, str], float | None] = {}
        for r in _vector(payload):
            m = r.get("metric") or {}
            if m.get("device_id"):
                got[(str(m["device_id"]), str(m.get("fleet_id") or ""))] = _instant_value(r)
        per_stat[key] = got
    keys = set().union(*(set(v) for v in per_stat.values())) if per_stat else set()
    window = {"metric": metric, "unit": unit, "lookback_hours": hours, "group_by": group_by, "ranked_by": stat}
    if len(keys) > MAX_DEVICE_ROWS:
        return {
            "summary": f"{len(keys)} devices matched; narrow with a fleet filter.",
            "query": window,
            "rows": [],
        }
    dev_rows = [
        {
            "device_id": did,
            "fleet_id": fid,
            "avg": _r(per_stat["avg"].get((did, fid))),
            "max": _r(per_stat["max"].get((did, fid))),
            "latest": _r(per_stat["latest"].get((did, fid))),
        }
        for did, fid in keys
    ]
    if not dev_rows:
        return {
            "summary": (
                f"No {metric} data in the last {hours:g}h"
                + (f" for fleet {fleet.get('name') or fleet['id']}" if fleet else "")
                + " (no devices reporting in the window). Not an error."
            ),
            "query": window,
            "org_stats": {"devices": 0, "avg": None, "max": None, "latest": None},
            "rows": [],
        }
    org_stats = rollup(dev_rows)
    fnames = fleet_names(client, org_id)
    if group_by == "fleet":
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in dev_rows:
            groups.setdefault(r["fleet_id"], []).append(r)
        rows = [
            {"fleet_id": fid, "fleet": fnames.get(fid) or None, **rollup(rs)} for fid, rs in groups.items()
        ]
    else:
        rows = [dict(r, fleet=fnames.get(r["fleet_id"]) or None) for r in dev_rows]
    rows.sort(key=lambda r: (r.get(stat) is None, (r.get(stat) or 0) * (1 if sort == "asc" else -1)))
    top = rows[: max(1, limit)]
    if group_by == "device":
        # Resolve names only for the rows we return.
        by_fleet: dict[str | None, dict[str, dict[str, Any]]] = {}
        for r in top:
            fid = r["fleet_id"] or None
            if fid not in by_fleet:
                by_fleet[fid] = device_names(client, org_id, fid)
            r["name"] = (by_fleet[fid].get(r["device_id"]) or {}).get("name")
    scope_txt = f" in fleet {fleet.get('name') or fleet['id']}" if fleet else " across the organisation"
    lead = top[0]
    lead_name = lead.get("name") or lead.get("fleet") or lead.get("device_id") or lead.get("fleet_id")
    out: dict[str, Any] = {
        "summary": (
            f"{metric}{scope_txt} (last {hours:g}h): avg {org_stats['avg']}{unit}, max {org_stats['max']}{unit}, "
            f"latest {org_stats['latest']}{unit} over {org_stats['devices']} device(s). "
            f"Highest {stat} by {group_by}: {lead_name} ({lead.get(stat)}{unit})."
            if sort == "desc"
            else f"{metric}{scope_txt} (last {hours:g}h): avg {org_stats['avg']}{unit} over {org_stats['devices']} device(s)."
        ),
        "query": window,
        "org_stats": org_stats,
        "rows": top,
    }
    if fleet:
        out["fleet"] = fleet
    if len(rows) > len(top):
        out["truncated"] = f"Showing {len(top)} of {len(rows)} {group_by} rows."
    return out


# -------------------------------------------------------- raw passthrough ---


def downsample(points: list[Any], n: int = MAX_POINTS_PER_SERIES) -> list[Any]:
    if len(points) <= n:
        return points
    stride = (len(points) - 1) / (n - 1)
    return [points[round(i * stride)] for i in range(n)]


def compact_prom(payload: Any, mode: str) -> dict[str, Any]:
    """Trim a native Prometheus-API JSON result to bounded size."""
    data = payload.get("data") if isinstance(payload, dict) else None
    result = [r for r in ((data or {}).get("result") or []) if isinstance(r, dict)]
    total = len(result)
    shown = []
    for r in result[:MAX_SERIES]:
        row: dict[str, Any] = {"metric": r.get("metric") or {}}
        if mode == "range":
            vals = r.get("values") or []
            row["stats"] = series_stats([[p[0], p[1]] for p in vals if isinstance(p, (list, tuple)) and len(p) >= 2])
            row["values"] = downsample(vals)
        else:
            row["value"] = r.get("value")
        shown.append(row)
    out: dict[str, Any] = {"result_type": (data or {}).get("resultType"), "series_total": total, "result": shown}
    if total > MAX_SERIES:
        out["truncated"] = f"Showing {MAX_SERIES} of {total} series; add a label matcher or an aggregation."
    return out
