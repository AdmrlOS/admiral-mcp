"""Memory-test tools: online modes only, error mapping, summaries, and the troubleshoot findings."""

from __future__ import annotations

import json

import httpx

from admrl_mcp import server
from admrl_mcp import troubleshoot as ts
from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings

DEV = "5f0c2a1e-7b3d-4c8e-9a6f-1d2e3f4a5b6c"
RESOLVED = {"match": {"id": DEV, "name": "kiosk"}, "candidates": [], "organization_id": "org-1"}
BASE = f"/v1/devices/{DEV}/memory-test"

STATUS = {
    "runId": "run-1",
    "plan": {"impact": "running", "placement": "online"},
    "phase": "testing",
    "pass": 1,
    "passes": 2,
    "pattern": "walking_ones",
    "patternIndex": 3,
    "patternCount": 12,
    "testedBytes": 50,
    "targetBytes": 200,
    "coverageBytes": 512 * 1024 * 1024,
    "totalBytes": 2048 * 1024 * 1024,
    "errors": 0,
    "tempMilliC": 61000,
}


def _setup(monkeypatch, routes):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        route = routes.get((request.method, request.url.path))
        if route is None:
            return httpx.Response(599, json={"msg": f"unexpected {request.method} {request.url.path}"})
        return route(request) if callable(route) else route

    client = AdmiralClient(
        settings=Settings(api_base="https://api.test/v1", token_id="t", secret_key="s", org_id=None),
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr(server, "get_client", lambda: client)
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: dict(RESOLVED))
    return seen


def _env(data):
    return httpx.Response(200, json={"success": True, "msg": "ok", "code": 200, "data": data})


def test_start_live_sends_online_running_and_never_offline(monkeypatch):
    seen = _setup(monkeypatch, {("POST", BASE): httpx.Response(202, json={"success": True, "code": 202, "data": {"deviceId": DEV, "runId": "run-1", "status": STATUS}})})
    out = json.loads(server.start_memory_test("kiosk", mode="live", quick=True, passes=1))
    body = json.loads(seen[0].content)
    assert body == {"impact": "running", "placement": "online", "quick": True, "passes": 1}
    assert out["started"] and out["run_id"] == "run-1"
    assert out["status"]["coverage_percent"] == 25.0 and out["status"]["temperature_c"] == 61.0


def test_full_online_requires_confirm(monkeypatch):
    seen = _setup(monkeypatch, {("POST", BASE): httpx.Response(202, json={"data": {"runId": "r2", "status": {}}})})
    out = json.loads(server.start_memory_test("kiosk", mode="full_online"))
    assert out["error"] == "confirmation_required" and seen == []
    out = json.loads(server.start_memory_test("kiosk", mode="full_online", confirm=True))
    assert json.loads(seen[0].content) == {"impact": "stopped", "placement": "online"}
    assert out["started"] and "workload is stopped" in out["note"]


def test_quick_is_ignored_for_full_online(monkeypatch):
    seen = _setup(monkeypatch, {("POST", BASE): httpx.Response(202, json={"data": {"runId": "r"}})})
    server.start_memory_test("kiosk", mode="full_online", quick=True, confirm=True)
    assert "quick" not in json.loads(seen[0].content)


def test_offline_is_rejected_client_side(monkeypatch):
    seen = _setup(monkeypatch, {})
    for mode in ("offline", "test_boot", "Test Boot"):
        out = json.loads(server.start_memory_test("kiosk", mode=mode, confirm=True))
        assert out["code"] == "operator_required" and "dashboard" in out["error"] and "console" in out["error"]
    assert seen == []


def test_no_placement_parameter_exposed():
    import asyncio

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in ("start_memory_test", "cancel_memory_test", "get_memory_test", "list_memory_test_results"):
        assert "placement" not in tools[name].inputSchema["properties"]
    assert tools["start_memory_test"].annotations.destructiveHint is True
    assert tools["get_memory_test"].annotations.readOnlyHint is True
    assert tools["list_memory_test_results"].annotations.readOnlyHint is True


def test_bad_args_do_not_call_api(monkeypatch):
    seen = _setup(monkeypatch, {})
    assert "passes" in json.loads(server.start_memory_test("kiosk", passes=99))["error"]
    assert "Unknown mode" in json.loads(server.start_memory_test("kiosk", mode="weird"))["error"]
    assert "limit" in json.loads(server.list_memory_test_results("kiosk", limit=0))["error"]
    assert seen == []


def test_error_codes_map_to_clear_messages(monkeypatch):
    cases = [
        (400, {"error": "invalid", "message": "passes: out of range"}, "invalid", "passes: out of range"),
        (403, {"error": "operator_required"}, "operator_required", None),
        (409, {"error": "busy", "status": STATUS}, "busy", None),
        (422, {"error": "unsupported"}, "unsupported", None),
        (429, {"error": "rate_limited"}, "rate_limited", None),
        (502, {"error": "workload_stop_failed", "message": "container wedged"}, "workload_stop_failed", "container wedged"),
        (502, {"error": "internal"}, "internal", None),
        (503, {"error": "device_offline", "device_id": DEV}, "device_offline", None),
        (504, {"error": "device_timeout"}, "device_timeout", None),
    ]
    for status, body, code, detail in cases:
        _setup(monkeypatch, {("POST", BASE): httpx.Response(status, json=body)})
        out = json.loads(server.start_memory_test("kiosk", mode="live"))
        assert out["code"] == code and out["http_status"] == status
        assert out["error"] and not out["error"].startswith("Admiral API")
        assert out.get("detail") == detail
        if code == "busy":
            assert out["active_run"]["phase"] == "testing"


def test_cancel_not_running_and_plain_403(monkeypatch):
    _setup(monkeypatch, {("POST", BASE + "/cancel"): httpx.Response(409, json={"error": "not_running"})})
    out = json.loads(server.cancel_memory_test("kiosk"))
    assert out["code"] == "not_running" and "nothing to cancel" in out["error"]
    # A 403 for no device access is not operator_required.
    _setup(monkeypatch, {("POST", BASE + "/cancel"): httpx.Response(403, json={"msg": "forbidden"})})
    out = json.loads(server.cancel_memory_test("kiosk"))
    assert "code" not in out and "403" in out["error"]


def test_cancel_sends_run_id(monkeypatch):
    seen = _setup(monkeypatch, {("POST", BASE + "/cancel"): _env({})})
    out = json.loads(server.cancel_memory_test("kiosk", run_id="run-1"))
    assert json.loads(seen[0].content) == {"runId": "run-1"} and out["cancelled"] is True


HEALTH_FAULT = {
    "capabilities": {"online": True, "offline": True, "softOffline": True},
    "fault": True,
    "faultReason": "data_line",
    "retiredPages": [4096, 8192],
    "edacCorrected": -1,
    "edacUncorrected": -1,
    "lastResult": {
        "runId": "run-0",
        "plan": {"impact": "stopped", "placement": "online"},
        "outcome": "fail",
        "verdict": "data_line",
        "errorCount": 40,
        "retired": [4096, 8192],
        "fault": True,
        "coverageBytes": 100,
        "totalBytes": 200,
        "errors": [{"physAddr": 4096, "mask": 16, "pattern": "walking_ones", "confirmed": True}],
    },
}


def test_get_memory_test_surfaces_fault_and_summary(monkeypatch):
    seen = _setup(monkeypatch, {("GET", BASE): _env({"deviceId": DEV, "source": "live", "supported": True, "status": None, "health": HEALTH_FAULT, "history": []})})
    out = json.loads(server.get_memory_test("kiosk"))
    assert out["needs_attention"][0]["severity"] == "critical" and "replace the board/RAM" in out["needs_attention"][0]["text"]
    h = out["health"]
    assert h["fault"] is True and h["retired_pages"] == 2 and "ecc_corrected" not in h
    assert h["capabilities"]["test_boot_available"] is True
    last = h["last_result"]
    assert last["verdict"] == "data_line" and last["coverage_percent"] == 50.0
    assert last["sample_errors"][0]["physical_address"] == "0x1000"
    assert seen[0].method == "GET"


def test_get_memory_test_interrupted_is_warning(monkeypatch):
    interrupted = {"runId": "r", "outcome": "interrupted", "interrupted": {"phase": "testing"}}
    _setup(monkeypatch, {("GET", BASE): _env({"source": "stored", "supported": True, "status": None, "health": {"lastResult": interrupted}, "history": [interrupted]})})
    out = json.loads(server.get_memory_test("kiosk"))
    assert [a["severity"] for a in out["needs_attention"]] == ["warning"]
    assert out["history"][0]["interrupted_at_phase"] == "testing"


def test_get_memory_test_stopped_run_is_maintenance_info(monkeypatch):
    st = {**STATUS, "plan": {"impact": "stopped", "placement": "online"}}
    _setup(monkeypatch, {("GET", BASE): _env({"source": "live", "supported": True, "status": st, "health": {}, "history": []})})
    out = json.loads(server.get_memory_test("kiosk"))
    assert out["needs_attention"][0]["severity"] == "info"
    assert out["status"]["mode"] == "full online (workload stopped)" and out["status"]["active"] is True


def test_get_memory_test_unsupported(monkeypatch):
    _setup(monkeypatch, {("GET", BASE): _env({"source": "stored", "supported": False, "status": None, "health": None, "history": []})})
    out = json.loads(server.get_memory_test("kiosk"))
    assert out["supported"] is False and "does not report" in out["note"]


def test_list_results_limit(monkeypatch):
    seen = _setup(monkeypatch, {("GET", BASE + "/results"): _env({"deviceId": DEV, "results": [HEALTH_FAULT["lastResult"]]})})
    out = json.loads(server.list_memory_test_results("kiosk", limit=5))
    assert dict(seen[0].url.params) == {"limit": "5"} and out["count"] == 1
    assert out["results"][0]["retired_pages"] == 2


# ---------------------------------------------------------- troubleshoot ---


def _ev(state=None, memtest=None, **extra):
    return {"device": {"id": DEV}, "online": True, "unavailable": {}, "checked": {}, "state_doc": state, "memtest": memtest, **extra}


def _cond(t, status="True", reason=""):
    return {"type": t, "status": status, "reason": reason}


def test_troubleshoot_memory_fault_is_critical():
    f = ts.analyse(_ev(state={"conditions": [_cond("MemoryFault", reason="data_line")]}))
    mf = next(x for x in f if x["code"] == "memory_fault")
    assert mf["severity"] == "critical" and "replace the board/RAM" in mf["finding"]
    assert ts.overall(f, True) == "critical"


def test_troubleshoot_interrupted_is_warning():
    mt = {"health": {"lastResult": {"outcome": "interrupted", "interrupted": {"phase": "testing"}}}, "history": []}
    f = ts.analyse(_ev(memtest=mt))
    assert [(x["code"], x["severity"]) for x in f if x["code"].startswith("memory")] == [("memory_test_interrupted", "warning")]


def test_troubleshoot_memory_test_hold_is_not_a_failure():
    state = {
        "conditions": [_cond("MemoryTest", reason="online_stopped"), _cond("WorkloadReady", "False", "MaintenanceHold")],
        "workload": {"state": "STOPPED"},
    }
    f = ts.analyse(_ev(state=state))
    codes = {x["code"]: x["severity"] for x in f}
    assert codes == {"memory_test_running": "info"}
    assert ts.overall(f, True) == "healthy"
    # Without the hold, a stopped workload still warns.
    f = ts.analyse(_ev(state={"workload": {"state": "STOPPED"}}))
    assert any(x["code"] == "workload_error" for x in f)


def test_gather_memtest_soft_on_unsupported():
    class C:
        def get_memory_test(self, *a, **k):
            from admrl_mcp.client import AdmiralAPIError

            raise AdmiralAPIError(422, "unsupported", {"error": "unsupported"})

    ev = ts.gather(C(), "org", {"id": DEV, "status": "offline"}, include={"memtest"})
    assert "memtest" not in ev["unavailable"] and ev["memtest"] is None
