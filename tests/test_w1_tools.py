"""W1-A tools: observed state, documents, diagnose, probe,
rollouts and the protocol census. Each test fakes the HTTP API with
httpx.MockTransport and asserts method, path, headers and body."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx
import pytest

from admrl_mcp import server
from admrl_mcp.client import AdmiralClient, parse_sse
from admrl_mcp.config import Settings

DEV = "0d9e8f7a-6b5c-4d3e-8f2a-1b0c9d8e7f6a"
FLEET = "11111111-2222-3333-4444-555555555555"
CONFIG = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
ROLLOUT = "99999999-8888-7777-6666-555555555555"
RESOLVED = {"match": {"id": DEV, "name": "dan-qemu-3"}, "candidates": [], "organization_id": "org-1"}


def envelope(data: Any, **extra: Any) -> dict[str, Any]:
    return {"code": 200, "msg": "ok", "data": data, **extra}


class FakeAPI:
    """Route table keyed on (METHOD, path-after-/v1). Records every request."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response] | httpx.Response] = {}
        self.seen: list[httpx.Request] = []

    def on(self, method: str, path: str, response: Any) -> None:
        self.routes[(method, path)] = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        path = request.url.path.removeprefix("/v1")
        route = self.routes.get((request.method, path))
        if route is None:
            return httpx.Response(599, json={"msg": f"unexpected {request.method} {path}"})
        return route(request) if callable(route) else route

    def only(self) -> httpx.Request:
        assert len(self.seen) == 1, [f"{r.method} {r.url.path}" for r in self.seen]
        return self.seen[0]


def install(monkeypatch, api: FakeAPI, *, org_id: str | None = "org-1") -> AdmiralClient:
    settings = Settings(api_base="https://api.test/v1", token_id="tok", secret_key="sec", org_id=org_id)
    client = AdmiralClient(settings=settings, transport=httpx.MockTransport(api))
    monkeypatch.setattr(server, "get_client", lambda: client)
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: dict(RESOLVED))
    return client


def assert_pat(request: httpx.Request, org: str | None = "org-1") -> None:
    assert request.headers["X-API-Token-ID"] == "tok"
    assert request.headers["X-API-Secret-Key"] == "sec"
    if org is None:
        assert "X-Organization-ID" not in request.headers
    else:
        assert request.headers["X-Organization-ID"] == org


def sse(*events: tuple[str, Any], keepalive: bool = True) -> bytes:
    out = []
    if keepalive:
        out.append(": keepalive\n\n")
    for name, payload in events:
        out.append(f"event: {name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(out).encode()


def state(gen: int, converged: str, ops: list[dict[str, Any]] | None = None, wl: str = "RUNNING") -> dict[str, Any]:
    st: dict[str, Any] = {
        "observedGeneration": gen,
        "conditions": [
            {"type": "Converged", "status": converged, "reason": "Applied" if converged == "True" else "Pending"},
            {"type": "WorkloadReady", "status": "True"},
        ],
        "workload": {"state": wl, "configurationName": "kiosk", "version": gen},
        "system": {"versions": {"initVersion": "v00051.4-dev", "rootfsVersion": "1.2.0"}},
    }
    if ops is not None:
        st["progress"] = {"targetGeneration": 5, "operations": ops}
    return st


# ------------------------------------------------------------------ parse ---


def test_parse_sse_handles_comments_multiline_and_default_name():
    lines = [": keepalive", "", "event: state", "data: {\"a\":", "data: 1}", "", "data: plain", ""]
    assert list(parse_sse(lines)) == [("state", '{"a":\n1}'), ("message", "plain")]


# ------------------------------------------------------------------ state ---


def test_get_device_state_stored(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state", httpx.Response(200, json=envelope(
        {"deviceId": DEV, "state": state(4, "True"), "source": "stored", "protocol": 1, "receivedAt": "t"})))
    install(monkeypatch, api)

    out = json.loads(server.get_device_state("dan-qemu-3"))

    req = api.only()
    assert req.method == "GET" and dict(req.url.params) == {}
    assert_pat(req)
    assert out["protocol"] == 1 and out["source"] == "stored"
    assert out["summary"]["conditions"]["Converged"]["status"] == "True"
    assert out["summary"]["observedGeneration"] == 4


def test_get_device_state_live_sends_live_1(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state", httpx.Response(200, json=envelope({"state": state(4, "True"), "source": "live"})))
    install(monkeypatch, api)

    out = json.loads(server.get_device_state(DEV, live=True))

    assert dict(api.only().url.params) == {"live": "1"}
    assert out["source"] == "live"


def test_get_device_state_offline_is_error(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state", httpx.Response(503, json={"error": "device_offline", "device_id": DEV}))
    install(monkeypatch, api)
    out = json.loads(server.get_device_state(DEV, live=True))
    assert "503" in out["error"] and "device_offline" in out["error"]


def test_watch_device_state_collects_transitions_and_progress(monkeypatch):
    op_running = {"id": "op1", "kind": "image_pull", "target": "kiosk:5", "phase": "running", "bytesDone": 50, "bytesTotal": 100}
    op_done = dict(op_running, phase="done", bytesDone=100)
    body = sse(
        ("state", {"deviceId": DEV, "kind": "state", "at": "t0", "state": state(4, "True")}),
        ("state", {"deviceId": DEV, "kind": "progress", "at": "t1", "state": state(4, "False", [op_running])}),
        ("state", {"deviceId": DEV, "kind": "progress", "at": "t2", "state": state(4, "False", [op_done])}),
        ("state", {"deviceId": DEV, "kind": "state", "at": "t3", "state": state(5, "True", [op_done])}),
    )
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state/stream", httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    install(monkeypatch, api)

    out = json.loads(server.watch_device_state("dan-qemu-3", timeout_s=5))

    req = api.only()
    assert req.headers["Accept"] == "text/event-stream"
    assert_pat(req)
    assert out["stream"]["ended"] == "closed" and out["stream"]["events"] == 4
    conds = [(t["condition"], t["from"], t["to"]) for t in out["transitions"] if t["type"] == "condition"]
    assert conds == [("Converged", "True", "False"), ("Converged", "False", "True")]
    gens = [(t["from"], t["to"]) for t in out["transitions"] if t["type"] == "observedGeneration"]
    assert gens == [(4, 5)]
    [op] = out["progress_operations"]
    assert op["kind"] == "image_pull" and op["phases"] == ["running", "done"] and op["last"]["pct"] == 100.0
    assert out["updates_by_kind"] == {"state": 2, "progress": 2}
    assert out["converged"] is True


def test_watch_device_state_stop_when_converged_with_min_generation(monkeypatch):
    body = sse(
        ("state", {"kind": "state", "state": state(4, "True")}),
        ("state", {"kind": "state", "state": state(5, "True")}),
        ("state", {"kind": "state", "state": state(6, "True")}),
    )
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state/stream", httpx.Response(200, content=body))
    install(monkeypatch, api)

    out = json.loads(server.watch_device_state(DEV, timeout_s=5, stop_when_converged=True, min_generation=5))

    assert out["stream"]["ended"] == "stop" and out["stream"]["events"] == 2
    assert out["final"]["observedGeneration"] == 5


def test_watch_device_state_times_out_on_keepalive_only_stream(monkeypatch):
    def keepalives():
        while True:
            time.sleep(0.02)
            yield b": keepalive\n\n"

    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state/stream", lambda r: httpx.Response(200, content=keepalives()))
    install(monkeypatch, api)
    monkeypatch.setattr(server, "_MAX_WATCH_S", 0.3)

    started = time.monotonic()
    out = json.loads(server.watch_device_state(DEV, timeout_s=0.3))

    assert out["stream"]["ended"] == "timeout"
    assert time.monotonic() - started < 3
    assert out["transitions"] == [] and out["initial"] is None


def test_watch_device_state_keeps_events_when_stream_breaks(monkeypatch):
    def broken():
        yield sse(("state", {"kind": "state", "state": state(4, "True")}))
        raise httpx.ReadError("connection reset")

    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state/stream", lambda r: httpx.Response(200, content=broken()))
    install(monkeypatch, api)

    out = json.loads(server.watch_device_state(DEV, timeout_s=5))

    assert out["stream"]["ended"].startswith("error:") and out["stream"]["events"] == 1
    assert out["initial"]["observedGeneration"] == 4


def test_watch_device_state_http_error_before_events(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/state/stream", httpx.Response(404, json={"code": 404, "msg": "Device not found"}))
    install(monkeypatch, api)
    out = json.loads(server.watch_device_state(DEV, timeout_s=5))
    assert "404" in out["error"]


# -------------------------------------------------------------- documents ---


DOC = {"apiVersion": "admrl.co/v1", "kind": "Device", "metadata": {"resourceVersion": "rv7", "generation": 3}, "spec": {}, "status": {}}


def test_get_device_document_json(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json=DOC, headers={"ETag": '"rv7"'}))
    install(monkeypatch, api)

    out = json.loads(server.get_device_document("dan-qemu-3"))

    req = api.only()
    assert req.headers["Accept"] == "application/json"
    assert_pat(req)
    assert out["resource_version"] == "rv7"
    assert out["document"]["kind"] == "Device"


def test_get_device_document_yaml(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/document",
           httpx.Response(200, text="apiVersion: admrl.co/v1\nkind: Device\n", headers={"ETag": '"rv7"', "content-type": "application/yaml"}))
    install(monkeypatch, api)

    out = json.loads(server.get_device_document(DEV, format="yaml"))

    assert api.only().headers["Accept"] == "application/yaml"
    assert out["document"].startswith("apiVersion: admrl.co/v1")
    assert out["content_type"] == "application/yaml"


def test_get_device_document_rejects_unknown_format(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    out = json.loads(server.get_device_document(DEV, format="xml"))
    assert "error" in out and api.seen == []


def test_patch_device_document_sends_merge_patch_with_if_match(monkeypatch):
    api = FakeAPI()
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(
        200, json=DOC, headers={"ETag": '"rv8"', "X-Admrl-Push": "sent", "X-Admrl-Changed": "spec.system.screenshots"}))
    install(monkeypatch, api)
    patch = {"spec": {"system": {"screenshots": "disabled"}}}

    out = json.loads(server.patch_device_document("dan-qemu-3", patch, if_match="rv7", change_reason="test", confirm=True))

    req = api.only()
    assert req.method == "PATCH"
    assert req.headers["Content-Type"] == "application/merge-patch+json"
    assert req.headers["If-Match"] == '"rv7"'
    assert req.headers["X-Change-Reason"] == "test"
    assert json.loads(req.content) == patch
    assert_pat(req)
    assert out["push"] == "sent" and out["changed"] == ["spec.system.screenshots"] and out["resource_version"] == "rv8"


def test_patch_device_document_accepts_json_string_and_rejects_empty(monkeypatch):
    api = FakeAPI()
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(200, json=DOC))
    install(monkeypatch, api)

    assert "error" in json.loads(server.patch_device_document(DEV, {}, confirm=True))
    assert api.seen == []
    json.loads(server.patch_device_document(DEV, '{"spec": {"fleet": "f2"}}', confirm=True))
    req = api.only()
    assert json.loads(req.content) == {"spec": {"fleet": "f2"}}
    assert "If-Match" not in req.headers


def test_patch_device_document_conflict(monkeypatch):
    api = FakeAPI()
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(409, json={"code": 409, "msg": "resourceVersion mismatch"}))
    install(monkeypatch, api)
    out = json.loads(server.patch_device_document(DEV, {"spec": {}}, if_match="old", confirm=True))
    assert "409" in out["error"]


def test_render_device_document_is_dry_run_by_default(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/devices/{DEV}/document:render",
           httpx.Response(200, json={"bundle": {}, "revision": "r1", "diff": [], "converged": True, "pushed": False}))
    install(monkeypatch, api)

    out = json.loads(server.render_device_document("dan-qemu-3"))
    assert dict(api.seen[-1].url.params) == {"dryRun": "1"}
    assert out["dry_run"] is True and out["render"]["revision"] == "r1"

    gated = json.loads(server.render_device_document("dan-qemu-3", push=True))
    assert gated["confirmation_required"] is True and dict(api.seen[-1].url.params) == {"dryRun": "1"}
    assert gated["next"]["arguments"] == {"device": DEV, "push": True, "confirm": True, "organization_id": "org-1"}
    json.loads(server.render_device_document("dan-qemu-3", push=True, confirm=True))
    assert dict(api.seen[-1].url.params) == {}
    assert api.seen[-1].method == "POST"
    assert_pat(api.seen[-1])


def test_adopt_local_override(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/devices/{DEV}/document:adoptLocalOverride", httpx.Response(
        200, json={"adopted": ["network"], "notAdopted": [], "document": DOC}, headers={"ETag": '"rv9"'}))
    install(monkeypatch, api)

    out = json.loads(server.adopt_local_override("dan-qemu-3", if_match="rv8", confirm=True))

    req = api.only()
    assert req.method == "POST" and req.headers["If-Match"] == '"rv8"'
    assert out["resource_version"] == "rv9" and out["result"]["adopted"] == ["network"]


def test_discard_local_override_defaults_to_all_paths(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/devices/{DEV}/document:discardLocalOverride",
           lambda r: httpx.Response(200, json={"deviceId": DEV, "paths": json.loads(r.content)["paths"], "sent": True}))
    install(monkeypatch, api)

    out = json.loads(server.discard_local_override("dan-qemu-3", confirm=True))
    assert json.loads(api.seen[-1].content) == {"paths": ["*"]}
    assert out["result"]["sent"] is True

    json.loads(server.discard_local_override("dan-qemu-3", paths=["network", "system.hostname"], confirm=True))
    assert json.loads(api.seen[-1].content) == {"paths": ["network", "system.hostname"]}


# --------------------------------------------------------------- diagnose ---


DIAGNOSIS = {
    "deviceId": DEV,
    "verdict": {"online": True, "workloadState": "RUNNING", "healthScore": 92, "topIssue": "rtc_drift"},
    "issues": [{"code": "rtc_drift", "severity": "warning", "title": "RTC drift", "evidence": []}],
    "lastReboot": {"kind": "power_on"},
    "signals": {"protocol": 1},
}


def test_diagnose_device_uses_server_route(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(200, json=envelope(DIAGNOSIS)))
    install(monkeypatch, api)
    monkeypatch.setattr(server.DeviceResolver, "find", lambda self, q, org_id=None, fleet_id=None: dict(RESOLVED))

    out = json.loads(server.diagnose_device("dan-qemu-3"))

    req = api.only()
    assert req.method == "GET"
    assert_pat(req)
    [report] = out["reports"]
    assert report["source"] == "server" and report["diagnosis"]["verdict"]["healthScore"] == 92


def test_diagnose_device_falls_back_to_local_distill_on_404(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(404, text="404 page not found\n"))
    api.on("GET", f"/devices/{DEV}/workload", httpx.Response(200, json=envelope({"state": "running"})))
    api.on("GET", f"/devices/{DEV}/device-stats", httpx.Response(200, json=envelope({"cpu": {"usage_percent": 3}})))
    api.on("GET", f"/devices/{DEV}/events", httpx.Response(200, json=envelope([{"event": "WorkloadCrashed", "timestamp": "t"}])))
    api.on("POST", "/metrics/logs/query", httpx.Response(200, json=envelope({"logs": [
        {"timestamp": "2026-10-05T00:00:00Z", "level": "error", "message": "panic: boom"}]})))
    install(monkeypatch, api)
    monkeypatch.setattr(server.DeviceResolver, "find", lambda self, q, org_id=None, fleet_id=None: dict(RESOLVED))

    out = json.loads(server.diagnose_device("dan-qemu-3"))

    [report] = out["reports"]
    assert report["source"] == "local_fallback" and "404" in report["fallback_reason"]
    assert report["workload"] == {"state": "running"}
    assert "logs" in report and "events" in report
    paths = [r.url.path for r in api.seen]
    assert paths[0].endswith("/diagnose")


def test_diagnose_device_real_error_is_reported_not_masked(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(500, json={"msg": "boom"}))
    install(monkeypatch, api)
    monkeypatch.setattr(server.DeviceResolver, "find", lambda self, q, org_id=None, fleet_id=None: dict(RESOLVED))

    out = json.loads(server.diagnose_device("dan-qemu-3"))

    assert "500" in out["reports"][0]["error"]
    assert len(api.seen) == 1


# ------------------------------------------------------------------ probe ---


def test_probe_device_posts_sections_and_timeout(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/devices/{DEV}/diagnostics/probe",
           httpx.Response(200, json=envelope({"sections": [{"name": "network", "items": [{"key": "psk", "value": "[redacted]"}]}]})))
    install(monkeypatch, api)

    out = json.loads(server.probe_device("dan-qemu-3", sections=["network", "time"], timeout_ms=5000))

    req = api.only()
    assert json.loads(req.content) == {"sections": ["network", "time"], "timeoutMs": 5000}
    assert_pat(req)
    assert out["report"]["sections"][0]["name"] == "network"


def test_probe_device_default_body_is_empty(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/devices/{DEV}/diagnostics/probe", httpx.Response(200, json=envelope({"sections": []})))
    install(monkeypatch, api)
    json.loads(server.probe_device(DEV))
    assert json.loads(api.only().content) == {}


# --------------------------------------------------------------- rollouts ---


def test_create_rollout_resolves_names_latest_version_and_strategy(monkeypatch):
    api = FakeAPI()
    api.on("GET", "/fleets", httpx.Response(200, json=envelope([{"id": FLEET, "name": "AlexConfigTest"},
                                                                {"id": "other", "name": "AlexConfigTest-2"}])))
    api.on("GET", "/configurations", httpx.Response(200, json=envelope([{"id": CONFIG, "name": "kiosk", "latest_version": 7}])))
    api.on("POST", "/rollouts", lambda r: httpx.Response(201, json=envelope({"id": ROLLOUT, "status": "pending"})))
    install(monkeypatch, api)

    out = json.loads(server.create_rollout(
        "AlexConfigTest", "kiosk", "latest",
        strategy={"canary": 1, "max_in_flight": 5, "failureThreshold": 0.2, "progress_deadline": "10m"}, confirm=True))

    post = api.seen[-1]
    assert post.method == "POST" and post.url.path == "/v1/rollouts"
    assert_pat(post)
    body = json.loads(post.content)
    assert body == {
        "name": "kiosk v7 → AlexConfigTest",
        "type": "config",
        "fleet_ids": [FLEET],
        "config_spec": {"config_id": CONFIG, "config_version": 7},
        "strategy": {"canary": 1, "maxInFlight": 5, "failureThreshold": 0.2, "progressDeadline": "10m"},
    }
    assert dict(api.seen[0].url.params)["search"] == "AlexConfigTest"
    assert out["rollout"]["id"] == ROLLOUT


def test_create_rollout_with_uuids_and_explicit_version_skips_lookups(monkeypatch):
    api = FakeAPI()
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))
    install(monkeypatch, api)

    json.loads(server.create_rollout(FLEET, CONFIG, 3, name="r1", confirm=True))

    body = json.loads(api.only().content)
    assert body["fleet_ids"] == [FLEET] and body["config_spec"] == {"config_id": CONFIG, "config_version": 3}
    assert body["name"] == "r1" and "strategy" not in body


def test_create_rollout_rejects_unknown_strategy_field(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    out = json.loads(server.create_rollout(FLEET, CONFIG, 3, strategy={"batch": 3}, confirm=True))
    assert "Unknown strategy field" in out["error"] and api.seen == []


def test_get_rollout_and_devices(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/rollouts/{ROLLOUT}", httpx.Response(200, json=envelope({"id": ROLLOUT, "status": "in_progress",
                                                                             "counts": {"targets": 2}})))
    api.on("GET", f"/rollouts/{ROLLOUT}/devices", httpx.Response(200, json={
        "success": True, "message": "ok", "data": [{"device_id": DEV, "phase": "pulling_image"}],
        "pagination": {"page": 1, "limit": 50, "total": 1, "totalPages": 1}}))
    install(monkeypatch, api)

    out = json.loads(server.get_rollout(ROLLOUT))
    assert out["rollout"]["counts"]["targets"] == 2
    assert_pat(api.seen[-1])

    out = json.loads(server.list_rollout_devices(ROLLOUT, status="failed", limit=50))
    assert dict(api.seen[-1].url.params) == {"status": "failed", "page": "1", "limit": "50"}
    assert out["devices"]["items"][0]["phase"] == "pulling_image"
    assert out["devices"]["pagination"]["total"] == 1


@pytest.mark.parametrize("action", ["pause", "resume", "cancel", "rollback"])
def test_rollout_control_actions(monkeypatch, action):
    api = FakeAPI()
    api.on("POST", f"/rollouts/{ROLLOUT}/{action}", httpx.Response(202, json=envelope(
        {"rolloutId": ROLLOUT, "action": action, "status": "signalled"})))
    install(monkeypatch, api)

    out = json.loads(server.rollout_control(ROLLOUT, action.upper(), reason="because", confirm=True))

    req = api.only()
    assert json.loads(req.content) == {"reason": "because"}
    assert_pat(req)
    assert out["result"]["status"] == "signalled"


def test_rollout_control_rejects_unknown_action(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    out = json.loads(server.rollout_control(ROLLOUT, "delete", confirm=True))
    assert "Unsupported" in out["error"] and api.seen == []


def test_watch_rollout_until_terminal(monkeypatch):
    body = sse(
        ("rollout", {"rolloutId": ROLLOUT, "status": "in_progress", "counts": {"targets": 2, "converged": 0}, "at": "t0"}),
        ("device", {"deviceId": DEV, "phase": "pulling_image", "progress": {"kind": "image_pull", "phase": "running",
                                                                           "bytesDone": 1, "bytesTotal": 4}, "at": "t1"}),
        ("device", {"deviceId": DEV, "phase": "converged", "progress": None, "at": "t2"}),
        ("rollout", {"rolloutId": ROLLOUT, "status": "paused", "pausedReason": "operator", "counts": {"targets": 2}, "at": "t3"}),
        ("rollout", {"rolloutId": ROLLOUT, "status": "completed", "counts": {"targets": 2, "converged": 2}, "at": "t4"}),
        ("rollout", {"rolloutId": ROLLOUT, "status": "never-read", "counts": {}, "at": "t5"}),
    )
    api = FakeAPI()
    api.on("GET", f"/rollouts/{ROLLOUT}", httpx.Response(200, json=envelope({"id": ROLLOUT, "status": "in_progress"})))
    api.on("GET", f"/rollouts/{ROLLOUT}/stream", httpx.Response(200, content=body))
    install(monkeypatch, api)

    out = json.loads(server.watch_rollout(ROLLOUT, timeout_s=5))

    stream_req = api.seen[-1]
    assert stream_req.headers["Accept"] == "text/event-stream"
    assert_pat(stream_req)
    assert out["stream"]["ended"] == "stop" and out["terminal"] is True and out["status"] == "completed"
    assert [(t["from"], t["to"]) for t in out["phase_transitions"]] == [("in_progress", "paused"), ("paused", "completed")]
    assert out["phase_transitions"][0]["pausedReason"] == "operator"
    assert [(t["from"], t["to"]) for t in out["device_transitions"]] == [(None, "pulling_image"), ("pulling_image", "converged")]
    assert out["device_progress"][DEV]["pct"] == 25.0
    assert out["counts"]["converged"] == 2


def test_watch_rollout_already_terminal_skips_stream(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/rollouts/{ROLLOUT}", httpx.Response(200, json=envelope({"id": ROLLOUT, "status": "completed",
                                                                             "counts": {"converged": 2}})))
    install(monkeypatch, api)

    out = json.loads(server.watch_rollout(ROLLOUT))

    assert out["stream"]["ended"] == "already_terminal" and out["counts"] == {"converged": 2}
    assert len(api.seen) == 1


# ----------------------------------------------------------------- census ---


def test_list_fleet_protocols_counts_by_protocol_and_agent(monkeypatch):
    devices = [
        {"id": "d1", "name": "new", "status": "online", "isOnline": True, "fleet": {"name": "A"}},
        {"id": "d2", "name": "old", "status": "online", "isOnline": {"isOnline": True, "lastSeen": "t"}, "fleet": {"name": "A"}},
        {"id": "d3", "name": "never", "status": "claimed", "isOnline": False, "fleet": {"name": "B"}},
    ]
    states = {
        "d1": httpx.Response(200, json=envelope({"protocol": 1, "receivedAt": "t",
                                                 "state": {"system": {"versions": {"initVersion": "v51", "rootfsVersion": "1.2.0"}}}})),
        "d2": httpx.Response(200, json=envelope({"protocol": 0, "state": {"system": {"versions": {}}}})),
        "d3": httpx.Response(404, json={"code": 404, "msg": "The device has not reported state yet"}),
    }
    api = FakeAPI()
    # Empty state versions (seen live) and missing state fall back to the system spec.
    api.on("GET", "/devices/d2", httpx.Response(200, json=envelope({"id": "d2", "systemSpec": {"versions": {"initVersion": "v40"}}})))
    api.on("GET", "/devices/d3", httpx.Response(200, json=envelope({"id": "d3", "systemSpec": {"versions": {"initVersion": "v30"}}})))
    api.on("GET", "/devices", httpx.Response(200, json=envelope(devices, pagination={"page": 1, "totalPages": 1})))
    for did, resp in states.items():
        api.on("GET", f"/devices/{did}/state", resp)
    install(monkeypatch, api)

    out = json.loads(server.list_fleet_protocols(fleet_id=FLEET))

    list_req = next(r for r in api.seen if r.url.path == "/v1/devices")
    assert dict(list_req.url.params)["fleet_id"] == FLEET
    assert all(r.method == "GET" for r in api.seen)
    assert out["total"] == 3
    assert out["byProtocol"] == {"1": 1, "0": 1, "no_state": 1}
    assert out["byAgentVersion"] == {"v51": 1, "v40": 1, "v30": 1}
    sources = {r["id"]: r.get("agentVersionSource") for r in out["devices"]}
    assert sources == {"d1": "state.system.versions", "d2": "systemSpec.versions", "d3": "systemSpec.versions"}
    assert not any(r.url.path == "/v1/devices/d1" for r in api.seen)
    assert out["legacyDevices"] == [{"id": "d2", "name": "old", "agentVersion": "v40", "online": True}]
    assert "platform_gate" not in out and "platform_gate_error" not in out
