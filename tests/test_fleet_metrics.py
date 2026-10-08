"""Fleet- and organisation-level metrics tools (mock transport, no network)."""

import asyncio
import json

import httpx
import pytest

from admrl_mcp import fleetmetrics, server
from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings

FLEET_A = "11111111-1111-4111-8111-111111111111"
FLEET_B = "22222222-2222-4222-8222-222222222222"
DEV_1 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1"
DEV_2 = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2"
DEV_3 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb3"

FLEETS = [{"id": FLEET_A, "name": "shop-fleet"}, {"id": FLEET_B, "name": "shop-fleet-test"}]
DEVICES = [
    {"id": DEV_1, "name": "till-01", "fleet": {"id": FLEET_A, "name": "shop-fleet"}},
    {"id": DEV_2, "name": "till-02", "fleet": {"id": FLEET_A, "name": "shop-fleet"}},
    {"id": DEV_3, "name": "kiosk-01", "fleet": {"id": FLEET_B, "name": "shop-fleet-test"}},
]


def _env(data, **extra):
    return {"code": 200, "msg": "ok", "data": data, **extra}


def _vec(rows):
    return {"status": "success", "data": {"resultType": "vector", "result": rows}}


def _row(dev, fleet, value):
    return {"metric": {"device_id": dev, "fleet_id": fleet}, "value": [1700000000, str(value)]}


def _install(monkeypatch, handler):
    paths: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        paths.append(request)
        path = request.url.path
        if path.endswith("/fleets") and request.method == "GET":
            return httpx.Response(200, json=_env(FLEETS))
        if path.endswith("/devices") and request.method == "GET":
            fid = request.url.params.get("fleet_id")
            devs = [d for d in DEVICES if not fid or d["fleet"]["id"] == fid]
            return httpx.Response(200, json=_env(devs, pagination={"page": 1}))
        return handler(request)

    settings = Settings(api_base="https://api.test/v1", token_id="t", secret_key="s", org_id="org-1")
    client = AdmiralClient(settings=settings, transport=httpx.MockTransport(wrapped))
    monkeypatch.setattr(server, "get_client", lambda: client)
    return paths


def _gate(status):
    return lambda request: httpx.Response(status, json={"code": status, "msg": "gate", "success": False})


# ------------------------------------------------------------------ client ---


def test_telemetry_query_instant_and_range_paths(monkeypatch):
    seen = _install(monkeypatch, lambda r: httpx.Response(200, json=_vec([])))
    client = server.get_client()
    client.telemetry_query("up", extra={"scope_fleet_id": FLEET_A})
    client.telemetry_query("up", start="2026-01-01T00:00:00Z", end="2026-01-01T01:00:00Z", step="60s")
    inst, rng = seen
    assert inst.url.path.endswith("/telemetry/metrics/query")
    assert inst.url.params["scope_fleet_id"] == FLEET_A and inst.headers["x-organization-id"] == "org-1"
    assert rng.url.path.endswith("/telemetry/metrics/query_range") and rng.url.params["step"] == "60s"


# ------------------------------------------------------------- fleet tools ---


def _fleet_series(handler_calls):
    def handler(request):
        handler_calls.append(request)
        assert request.url.path.endswith(f"/fleets/{FLEET_A}/metrics")
        assert request.url.params["aggregate"] == "false" and request.url.params["metric_name"] == "cpu"
        return httpx.Response(
            200,
            json=_env(
                {
                    "meta": {"metric": "cpu", "unit": "%"},
                    "series": [
                        {"labels": {"device_id": DEV_1}, "values": [[1, 10.0], [2, 30.0]]},
                        {"labels": {"device_id": DEV_2}, "values": [[1, 50.0], [2, 70.0]]},
                    ],
                }
            ),
        )

    return handler


def test_get_fleet_metrics_by_name_with_stats_and_devices(monkeypatch):
    calls: list = []
    _install(monkeypatch, _fleet_series(calls))
    out = json.loads(server.get_fleet_metrics("shop-fleet", per_device=True))
    assert list(out)[0] == "summary"
    assert out["fleet"] == {"id": FLEET_A, "name": "shop-fleet"}  # exact name wins over shop-fleet-test
    assert out["fleet_stats"] == {"devices": 2, "avg": 40.0, "max": 70.0, "latest": 50.0}
    assert [d["name"] for d in out["devices"]] == ["till-02", "till-01"]
    assert out["devices"][0]["avg"] == 60.0
    assert len(calls) == 1


def test_get_fleet_metrics_compares_device_with_fleet(monkeypatch):
    _install(monkeypatch, _fleet_series([]))
    monkeypatch.setattr(
        server,
        "_resolve_device",
        lambda q, organization_id=None, fleet_id=None: {
            "match": {"id": DEV_1, "name": "till-01", "fleet": {"id": FLEET_A, "name": "shop-fleet"}},
            "candidates": [],
            "organization_id": "org-1",
        },
    )
    out = json.loads(server.get_fleet_metrics(device="till-01"))
    assert out["device"]["avg"] == 20.0 and out["device"]["avg_vs_fleet"] == -20.0
    assert out["device"]["rank_by_avg"] == 2 and out["device"]["of"] == 2
    assert "devices" not in out or out["devices"] == []


def test_get_fleet_metrics_empty_series_is_not_an_error(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(200, json=_env({"meta": {}, "series": None})))
    out = json.loads(server.get_fleet_metrics(FLEET_A))
    assert "error" not in out and out["fleet_stats"]["devices"] == 0
    assert "No cpu data" in out["summary"]


def test_get_fleet_metrics_402_is_a_billing_gate(monkeypatch):
    _install(monkeypatch, _gate(402))
    out = json.loads(server.get_fleet_metrics("shop-fleet"))
    assert "error" not in out
    assert out["available"] is False and out["reason"] == "telemetry_api" and "next_step" in out
    assert "not an outage" in out["summary"]


def test_get_fleet_metrics_403_message(monkeypatch):
    _install(monkeypatch, _gate(403))
    out = json.loads(server.get_fleet_metrics("shop-fleet"))
    assert out["status"] == 403 and out["available"] is False and "get_telemetry_scope" in out["next_step"]


def test_fleet_name_resolution_ambiguous_and_missing(monkeypatch):
    _install(monkeypatch, _gate(500))
    amb = json.loads(server.get_fleet_metrics("shop"))
    assert amb["match"] is None and len(amb["candidates"]) == 2 and "fleet id" in amb["note"]
    none = json.loads(server.get_fleet_metrics("nope"))
    assert none["candidates"] == [] and "No fleet" in none["note"]
    bad = json.loads(server.get_fleet_metrics("shop-fleet", metric="gpu"))
    assert bad["choices"] == list(fleetmetrics.METRICS)


def test_get_fleet_health(monkeypatch):
    def handler(request):
        assert request.url.path.endswith(f"/fleets/{FLEET_A}/health")
        return httpx.Response(
            200, json=_env({"total_devices": 4, "online": 3, "avg_cpu": 12.5, "avg_memory": 40.1, "avg_disk": 20})
        )

    _install(monkeypatch, handler)
    out = json.loads(server.get_fleet_health("shop-fleet"))
    assert list(out)[0] == "summary" and out["health"]["offline"] == 1 and "3/4 online" in out["summary"]


def test_get_fleet_health_zero_devices_adds_cross_check_note(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(200, json=_env({"total_devices": 0, "online": 0})))
    assert "Cross-check" in json.loads(server.get_fleet_health("shop-fleet"))["note"]


def test_get_fleet_health_402(monkeypatch):
    _install(monkeypatch, _gate(402))
    out = json.loads(server.get_fleet_health("shop-fleet"))
    assert out["reason"] == "telemetry_api"


def test_get_fleet_uptime_buckets_and_current(monkeypatch):
    def handler(request):
        if request.url.path.endswith("/uptime/percentage"):
            return httpx.Response(200, json=_env({"fleet_id": FLEET_A, "uptime_pct": 99.5}))
        assert request.url.params["period_type"] == "daily" and request.url.params["periods_back"] == "2"
        return httpx.Response(
            200,
            json=_env(
                {
                    "periods": [
                        {"period_start": "2026-01-01T00:00:00Z", "period_end": "2026-01-02T00:00:00Z", "uptime_pct": 100},
                        {"period_start": "2026-01-02T00:00:00Z", "period_end": "2026-01-03T00:00:00Z", "uptime_pct": 90.5},
                    ],
                    "avg_uptime_pct": 95.25,
                }
            ),
        )

    _install(monkeypatch, handler)
    out = json.loads(server.get_fleet_uptime("shop-fleet", periods_back=2))
    assert out["current_uptime_pct"] == 99.5 and out["average_uptime_pct"] == 95.25
    assert [p["uptime_pct"] for p in out["periods"]] == [100.0, 90.5]
    assert json.loads(server.get_fleet_uptime("shop-fleet", period_type="yearly"))["choices"]


# --------------------------------------------------------------- org tools ---


def _org_handler(queries):
    def handler(request):
        assert request.url.path.endswith("/telemetry/metrics/query")
        q = request.url.params["query"]
        queries.append(request)
        val = {"avg_over_time": (10, 30, 50), "max_over_time": (20, 60, 90), "last_over_time": (12, 33, 55)}
        kind = q.split("(")[0]
        a, b, c = val[kind]
        return httpx.Response(200, json=_vec([_row(DEV_1, FLEET_A, a), _row(DEV_2, FLEET_A, b), _row(DEV_3, FLEET_B, c)]))

    return handler


def test_get_org_metrics_by_fleet_ranked_with_names(monkeypatch):
    queries: list = []
    _install(monkeypatch, _org_handler(queries))
    out = json.loads(server.get_org_metrics("cpu", "fleet", "avg"))
    assert list(out)[0] == "summary"
    assert out["org_stats"] == {"devices": 3, "avg": 30.0, "max": 90.0, "latest": 33.33}
    assert [r["fleet"] for r in out["rows"]] == ["shop-fleet-test", "shop-fleet"]
    assert out["rows"][1] == {"fleet_id": FLEET_A, "fleet": "shop-fleet", "devices": 2, "avg": 20.0, "max": 60.0, "latest": 22.5}
    assert len(queries) == 3
    q = queries[0].url.params["query"]
    assert "edge_cpu_usagepercent" in q and "by (device_id, fleet_id)" in q and "[24h:5m]" in q


def test_get_org_metrics_top_devices_by_max_with_names_and_limit(monkeypatch):
    _install(monkeypatch, _org_handler([]))
    out = json.loads(server.get_org_metrics("cpu", "device", "max", limit=2))
    assert [r["name"] for r in out["rows"]] == ["kiosk-01", "till-02"]
    assert out["rows"][0]["fleet"] == "shop-fleet-test" and out["truncated"]
    asc = json.loads(server.get_org_metrics("cpu", "device", "avg", ascending=True, limit=1))
    assert asc["rows"][0]["name"] == "till-01"


def test_get_org_metrics_fleet_filter_uses_server_scope_not_query_text(monkeypatch):
    queries: list = []
    _install(monkeypatch, _org_handler(queries))
    json.loads(server.get_org_metrics("memory", "device", "avg", fleet="shop-fleet"))
    for req in queries:
        assert req.url.params["scope_fleet_id"] == FLEET_A
        assert FLEET_A not in req.url.params["query"]


def test_get_org_metrics_counter_and_disk_expressions():
    assert "rate(edge_network_interfaces_rxbytes[5m])" in fleetmetrics.device_expr("network_rx")
    assert fleetmetrics.device_expr("network_rx").startswith("sum by")
    assert fleetmetrics.device_expr("disk").startswith("max by")
    assert fleetmetrics.window_queries("cpu", 72)["avg"].endswith("[72h:30m])")


def test_get_org_metrics_empty_is_not_an_error(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(200, json=_vec([])))
    out = json.loads(server.get_org_metrics())
    assert "error" not in out and out["rows"] == [] and "No cpu data" in out["summary"]


@pytest.mark.parametrize("status", [402, 403])
def test_get_org_metrics_gates_are_not_errors(monkeypatch, status):
    _install(monkeypatch, _gate(status))
    out = json.loads(server.get_org_metrics())
    assert "error" not in out and out["available"] is False and out["status"] == status
    assert out["next_step"]


def test_get_org_metrics_rejects_bad_choices(monkeypatch):
    _install(monkeypatch, _gate(500))
    assert "choices" in json.loads(server.get_org_metrics(metric="x"))
    assert "choices" in json.loads(server.get_org_metrics(group_by="x"))
    assert "choices" in json.loads(server.get_org_metrics(stat="p99"))
    assert "error" in json.loads(server.get_org_metrics(lookback_hours=100000))


# ---------------------------------------------------------- raw + scope ---


def test_query_telemetry_metrics_instant_bounded(monkeypatch):
    rows = [_row(f"d{i}", FLEET_A, i) for i in range(40)]
    _install(monkeypatch, lambda r: httpx.Response(200, json=_vec(rows)))
    out = json.loads(server.query_telemetry_metrics("edge_cpu_usagepercent"))
    assert out["series_total"] == 40 and len(out["result"]) == fleetmetrics.MAX_SERIES and out["truncated"]


def test_query_telemetry_metrics_range_downsamples_and_floors_step(monkeypatch):
    seen: list = []
    pts = [[1700000000 + i * 60, str(i)] for i in range(300)]

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json={"status": "success", "data": {"resultType": "matrix", "result": [{"metric": {"device_id": "d"}, "values": pts}]}}
        )

    _install(monkeypatch, handler)
    out = json.loads(server.query_telemetry_metrics("up", mode="range", lookback_hours=168, step="1s"))
    req = seen[0]
    assert req.url.path.endswith("/query_range") and req.url.params["step"] == "1209s"
    series = out["result"][0]
    assert len(series["values"]) == fleetmetrics.MAX_POINTS_PER_SERIES and series["stats"]["points"] == 300


def test_query_telemetry_metrics_validation_and_402(monkeypatch):
    _install(monkeypatch, _gate(402))
    assert "error" in json.loads(server.query_telemetry_metrics(""))
    assert "error" in json.loads(server.query_telemetry_metrics("x" * 2001))
    assert "choices" in json.loads(server.query_telemetry_metrics("up", mode="logs"))
    assert json.loads(server.query_telemetry_metrics("up"))["reason"] == "telemetry_api"


def test_get_telemetry_scope(monkeypatch):
    payload = {"org_id": "org-1", "org_wide": False, "fleet_ids": [FLEET_A], "device_ids": [DEV_3]}
    _install(monkeypatch, lambda r: httpx.Response(200, json=_env(payload)))
    out = json.loads(server.get_telemetry_scope())
    assert out["org_wide"] is False and out["fleets"] == [{"id": FLEET_A, "name": "shop-fleet"}]
    assert "1 fleet(s) and 1 device(s)" in out["summary"]
    _install(monkeypatch, lambda r: httpx.Response(200, json=_env({"org_id": "org-1", "org_wide": True})))
    assert "org-wide" in json.loads(server.get_telemetry_scope())["summary"]
    _install(monkeypatch, _gate(403))
    assert json.loads(server.get_telemetry_scope())["status"] == 403


# ------------------------------------------------------------- registration ---

NEW_TOOLS = {
    "get_fleet_metrics",
    "get_fleet_health",
    "get_fleet_uptime",
    "get_org_metrics",
    "query_telemetry_metrics",
    "get_telemetry_scope",
}


def test_new_tools_registered_read_only():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert NEW_TOOLS <= set(tools)
    for name in NEW_TOOLS:
        assert tools[name].annotations is not None and tools[name].annotations.readOnlyHint is True
