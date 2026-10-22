"""Developer sessions and the HTTP proxy to device services (mocked transport, synthetic ids)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from admrl_mcp import server

from test_confirmation import FLEET, GuardedAPI, world
from test_ops_tools import ok
from test_w1_tools import DEV, assert_pat, install

SESSION = {
    "id": "sess-1",
    "user": "dev@example.com",
    "since": "2026-10-22T01:00:00Z",
    "expires": "2026-10-22T09:00:00Z",
    "mounts": ["/srv/app"],
    "image": "registry.example.com/app:dev",
}
PROXY = {
    "name": "brisk-otter-1234",
    "url": "https://brisk-otter-1234-proxy.example.com/",
    "device_id": DEV,
    "service": "web",
    "created_at": "2026-10-22T01:00:00Z",
    "expires": "2026-10-22T13:00:00Z",
}


def j(text: str) -> dict[str, Any]:
    return json.loads(text)


@pytest.fixture
def api(monkeypatch) -> GuardedAPI:
    fake = world()
    install(monkeypatch, fake)
    return fake


def writes(api: GuardedAPI, method: str | None = None) -> list[httpx.Request]:
    return [r for r in api.seen if r.method != "GET" and (method is None or r.method == method)]


# ------------------------------------------------------------ devices ---


def test_get_device_includes_the_dev_session(api):
    api.on("GET", f"/devices/{DEV}", ok({"id": DEV, "name": "dan-qemu-3", "status": "online"}))
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"status": {"devSession": SESSION}}))
    out = j(server.get_device("dan-qemu-3"))
    assert out["device"]["dev_session"] == SESSION
    assert out["document"]["status"]["devSession"]["user"] == "dev@example.com"


def test_get_device_without_a_session_has_no_dev_session(api):
    api.on("GET", f"/devices/{DEV}", ok({"id": DEV, "name": "dan-qemu-3", "status": "online"}))
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"status": {"protocol": 1}}))
    out = j(server.get_device("dan-qemu-3"))
    assert "dev_session" not in out["device"] and "document" not in out and "notes" not in out


def test_get_device_survives_a_failed_document_read(api):
    api.on("GET", f"/devices/{DEV}", ok({"id": DEV, "name": "dan-qemu-3", "status": "online"}))
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(403, json={"msg": "no access"}))
    out = j(server.get_device("dan-qemu-3"))
    assert out["device"]["id"] == DEV and "developer session unavailable" in out["notes"][0]


def test_list_devices_dev_session_filter_maps_to_query_param(api):
    api.on("GET", "/devices", ok([{"id": DEV, "name": "dan-qemu-3", "status": "online"}]))
    j(server.list_devices(dev_session=True))
    assert {r.url.params.get("devSession") for r in api.seen if r.url.path.endswith("/devices")} == {"true"}


def test_list_devices_default_sends_no_dev_session_param(api):
    api.on("GET", "/devices", ok([]))
    j(server.list_devices())
    assert all("devSession" not in r.url.params for r in api.seen)


# ------------------------------------------------------------ proxies ---


def test_open_device_proxy_posts_service_and_returns_url(api):
    api.on("POST", f"/devices/{DEV}/proxy", ok(PROXY))
    out = j(server.open_device_proxy("dan-qemu-3", "web"))
    req = writes(api)[0]
    assert_pat(req) and json.loads(req.content) == {"service": "web"}
    assert out["proxy"]["url"] == PROXY["url"] and out["proxy"]["expires"] == PROXY["expires"]
    assert out["next"][0] == {"tool": "close_proxy", "why": "Close it when finished.", "arguments": {"name": PROXY["name"]}}


def test_open_device_proxy_disabled_platform_says_not_enabled(api):
    api.on("POST", f"/devices/{DEV}/proxy", httpx.Response(503, json={"msg": "the HTTP proxy is not enabled"}))
    out = j(server.open_device_proxy("dan-qemu-3", "web"))
    assert "not enabled" in out["error"] and "proxy" not in out


def test_open_device_proxy_other_errors_pass_through_and_service_is_required(api):
    api.on("POST", f"/devices/{DEV}/proxy", httpx.Response(404, json={"msg": "service not found"}))
    assert "404" in j(server.open_device_proxy("dan-qemu-3", "nope"))["error"]
    n = len(api.seen)
    assert "service is required" in j(server.open_device_proxy("dan-qemu-3", " "))["error"] and len(api.seen) == n


def test_list_proxies(api):
    api.on("GET", "/proxy", ok([PROXY]))
    out = j(server.list_proxies())
    assert out["summary"] == "1 open proxy." and out["proxies"][0]["name"] == PROXY["name"]
    api.on("GET", "/proxy", ok([]))
    assert j(server.list_proxies())["proxies"] == []
    api.on("GET", "/proxy", httpx.Response(503, json={"msg": "off"}))
    assert "not enabled" in j(server.list_proxies())["error"]


def test_close_proxy(api):
    api.on("DELETE", f"/proxy/{PROXY['name']}", ok(None))
    out = j(server.close_proxy(PROXY["name"]))
    assert out["closed"] == PROXY["name"] and writes(api, "DELETE")
    api.on("DELETE", "/proxy/gone", httpx.Response(404, json={"msg": "proxy not found"}))
    assert "No open proxy" in j(server.close_proxy("gone"))["error"]


def test_proxy_tool_annotations():
    tools = {t.name: t.annotations for t in __import__("asyncio").run(server.mcp.list_tools())}
    assert tools["list_proxies"].readOnlyHint is True
    for name in ("open_device_proxy", "close_proxy"):
        assert tools[name].readOnlyHint is False and tools[name].destructiveHint is False
    assert tools["set_fleet_dev_sessions"].destructiveHint is True


# ----------------------------------------------------- fleet dev sessions ---


def test_get_fleet_policies_shows_dev_sessions(api):
    api.on("GET", f"/fleets/{FLEET}/dev-sessions", ok({"devSessionsAllowed": True, "devSessionMaxLife": 28800}))
    out = j(server.get_fleet_policies(FLEET))
    assert out["dev_sessions"] == {"allowed": True, "max_life_seconds": 28800}
    assert "dev sessions allowed (max 28800s)" in out["summary"]


def test_set_fleet_dev_sessions_previews_then_writes_and_reads_back(api):
    state = {"devSessionsAllowed": False, "devSessionMaxLife": 0}
    api.on("GET", f"/fleets/{FLEET}/dev-sessions", lambda r: ok(dict(state)))

    def put(r: httpx.Request) -> httpx.Response:
        state.update(json.loads(r.content))
        return ok(dict(state))

    api.on("PUT", f"/fleets/{FLEET}/dev-sessions", put)
    pre = j(server.set_fleet_dev_sessions(FLEET, allowed=True, max_life_seconds=3600))
    assert pre["confirmation_required"] is True and not writes(api)
    assert pre["preview"]["before"] == {"allowed": False, "max_life_seconds": 0}
    assert pre["preview"]["after"] == {"allowed": True, "max_life_seconds": 3600}
    done = j(server.set_fleet_dev_sessions(**pre["next"]["arguments"]))
    assert json.loads(writes(api, "PUT")[0].content) == {"devSessionsAllowed": True, "devSessionMaxLife": 3600}
    assert done["read_back"]["confirmed"] is True and done["after"]["allowed"] is True


def test_set_fleet_dev_sessions_max_life_only_sends_only_that_field(api):
    state = {"devSessionsAllowed": True, "devSessionMaxLife": 0}
    api.on("GET", f"/fleets/{FLEET}/dev-sessions", lambda r: ok(dict(state)))
    api.on("PUT", f"/fleets/{FLEET}/dev-sessions", lambda r: (state.update(json.loads(r.content)), ok(dict(state)))[1])
    j(server.set_fleet_dev_sessions(FLEET, max_life_seconds=600, confirm=True))
    assert json.loads(writes(api, "PUT")[0].content) == {"devSessionMaxLife": 600}


def test_set_fleet_dev_sessions_turning_off_warns(api):
    api.on("GET", f"/fleets/{FLEET}/dev-sessions", ok({"devSessionsAllowed": True, "devSessionMaxLife": 0}))
    pre = j(server.set_fleet_dev_sessions(FLEET, allowed=False))
    assert any("ends every active session" in w for w in pre["warnings"]) and not writes(api)


def test_set_fleet_dev_sessions_refuses_noop_empty_and_negative(api):
    api.on("GET", f"/fleets/{FLEET}/dev-sessions", ok({"devSessionsAllowed": True, "devSessionMaxLife": 60}))
    assert j(server.set_fleet_dev_sessions(FLEET, allowed=True, max_life_seconds=60, confirm=True))["error"].startswith("No change")
    assert "nothing to change" in j(server.set_fleet_dev_sessions(FLEET, confirm=True))["error"]
    assert "0 or more" in j(server.set_fleet_dev_sessions(FLEET, max_life_seconds=-1, confirm=True))["error"]
    assert not writes(api)
