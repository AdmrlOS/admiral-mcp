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
    "visibility": "private",
    "created_at": "2026-10-22T01:00:00Z",
    "expires": "2026-10-22T13:00:00Z",
    "idle_timeout": 1800,
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
    assert {"tool": "close_proxy", "why": "Close it when finished.", "arguments": {"name": PROXY["name"]}} in out["next"]
    assert out["proxy"]["visibility"] == "private" and out["proxy"]["idle_timeout"] == 1800 and "warnings" not in out


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
    assert tools["get_fleet_proxy_policy"].readOnlyHint is True and tools["get_org_proxy_prefix"].readOnlyHint is True
    assert tools["set_fleet_proxy_policy"].destructiveHint is True
    for name in ("open_device_proxy", "close_proxy", "update_proxy", "set_org_proxy_prefix"):
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


# ------------------------------------------------------- proxy v2 ---

ORG_SETTINGS = "/organisations/org-1/proxy-settings"
PUBLIC = {**PROXY, "visibility": "public", "url": "https://brisk-otter-1234.example.com/"}
NEVER = {k: v for k, v in PUBLIC.items() if k != "expires"}


def test_open_device_proxy_sends_v2_options(api):
    api.on("POST", f"/devices/{DEV}/proxy", ok(NEVER))
    out = j(server.open_device_proxy("dan-qemu-3", "web", visibility="Public", expires_in=0, idle_timeout=0, name="demo"))
    assert json.loads(writes(api)[0].content) == {"service": "web", "visibility": "public", "expires_in": 0, "idle_timeout": 0, "name": "demo"}
    assert out["proxy"]["expires"] == "never" and out["proxy"]["visibility"] == "public"
    assert "anyone with this link" in out["warnings"][0] and "EdgeWire usage" in out["warnings"][0]


def test_open_device_proxy_validates_before_calling(api):
    for kw, text in (
        ({"visibility": "world"}, "private' or 'public"),
        ({"expires_in": -1}, "expires_in"),
        ({"idle_timeout": -5}, "idle_timeout"),
        ({"name": "Bad_Name"}, "name must be"),
        ({"name": "x" * 41}, "name must be"),
    ):
        assert text in j(server.open_device_proxy("dan-qemu-3", "web", **kw))["error"]
    assert not api.seen


def test_open_device_proxy_description_warns_about_public():
    d = {t.name: t.description for t in __import__("asyncio").run(server.mcp.list_tools())}["open_device_proxy"]
    assert "ANYONE WITH THE LINK" in d and "WITHOUT SIGNING IN" in d and "EdgeWire usage" in d and "fleet setting" in d


@pytest.mark.parametrize(
    "code, status, text",
    [
        ("public_not_allowed", 403, "not allowed"),
        ("exceeds_fleet_max", 400, "fleet's public proxy limit"),
        ("prefix_required", 400, "need an organisation proxy prefix"),
        ("name_taken", 409, "already open"),
    ],
)
def test_open_device_proxy_maps_error_codes(api, code, status, text):
    api.on("POST", f"/devices/{DEV}/proxy", httpx.Response(status, json={"msg": "refused", "data": {"error_code": code}}))
    out = j(server.open_device_proxy("dan-qemu-3", "web", visibility="public"))
    assert out["error_code"] == code and text in out["error"] and out["hint"]
    assert "permission_denied" not in out and "conflict" not in out


def test_error_code_at_top_level_is_also_mapped(api):
    api.on("POST", f"/devices/{DEV}/proxy", httpx.Response(409, json={"error_code": "name_taken", "msg": "taken"}))
    assert j(server.open_device_proxy("dan-qemu-3", "web", name="x"))["error_code"] == "name_taken"


def test_list_proxies_defaults_legacy_rows(api):
    legacy = {k: v for k, v in PROXY.items() if k not in ("visibility", "idle_timeout")}
    api.on("GET", "/proxy", ok([legacy, NEVER]))
    rows = j(server.list_proxies())["proxies"]
    assert rows[0]["visibility"] == "private" and rows[0]["idle_timeout"] is None
    assert rows[1]["visibility"] == "public" and rows[1]["expires"] == "never"


def test_update_proxy_patches_only_what_is_given(api):
    api.on("PATCH", f"/proxy/{PROXY['name']}", ok(PUBLIC))
    out = j(server.update_proxy(PROXY["name"], visibility="public", expires_in=3600))
    req = writes(api, "PATCH")[0]
    assert_pat(req)
    assert json.loads(req.content) == {"visibility": "public", "expires_in": 3600}
    assert out["proxy"]["visibility"] == "public" and out["warnings"]
    j(server.update_proxy(PROXY["name"], idle_timeout=0))
    assert json.loads(writes(api, "PATCH")[1].content) == {"idle_timeout": 0}


def test_update_proxy_refusals_and_errors(api):
    assert "nothing to change" in j(server.update_proxy(PROXY["name"]))["error"]
    assert "expires_in" in j(server.update_proxy(PROXY["name"], expires_in=-1))["error"]
    assert "name is required" in j(server.update_proxy(" "))["error"]
    assert not api.seen
    api.on("PATCH", "/proxy/gone", httpx.Response(404, json={"msg": "nf"}))
    assert "No open proxy" in j(server.update_proxy("gone", visibility="private"))["error"]
    api.on("PATCH", f"/proxy/{PROXY['name']}", httpx.Response(403, json={"msg": "no", "data": {"error_code": "public_not_allowed"}}))
    assert j(server.update_proxy(PROXY["name"], visibility="public"))["error_code"] == "public_not_allowed"
    api.on("PATCH", f"/proxy/{PROXY['name']}", httpx.Response(503, json={"msg": "off"}))
    assert "not enabled" in j(server.update_proxy(PROXY["name"], visibility="private"))["error"]


# --------------------------------------------------- fleet proxy policy ---


def test_get_fleet_proxy_policy(api):
    api.on("GET", f"/fleets/{FLEET}/proxy-policy", ok({"publicProxiesAllowed": True, "publicProxyMaxLife": 86400}))
    out = j(server.get_fleet_proxy_policy(FLEET))
    assert out["proxy_policy"] == {"public_allowed": True, "public_max_life_seconds": 86400}
    assert "public proxies allowed, max life 86400s" in out["summary"]


def test_set_fleet_proxy_policy_previews_then_writes_and_reads_back(api):
    state = {"publicProxiesAllowed": False, "publicProxyMaxLife": 0}
    api.on("GET", f"/fleets/{FLEET}/proxy-policy", lambda r: ok(dict(state)))

    def put(r: httpx.Request) -> httpx.Response:
        state.update(json.loads(r.content))
        return ok(dict(state))

    api.on("PUT", f"/fleets/{FLEET}/proxy-policy", put)
    pre = j(server.set_fleet_proxy_policy(FLEET, public_allowed=True, public_max_life_seconds=7200))
    assert pre["confirmation_required"] is True and not writes(api)
    assert pre["preview"]["after"] == {"public_allowed": True, "public_max_life_seconds": 7200}
    assert any("without signing in" in w for w in pre["warnings"])
    done = j(server.set_fleet_proxy_policy(**pre["next"]["arguments"]))
    assert json.loads(writes(api, "PUT")[0].content) == {"publicProxiesAllowed": True, "publicProxyMaxLife": 7200}
    assert done["read_back"]["confirmed"] is True and done["after"]["public_allowed"] is True


def test_set_fleet_proxy_policy_turning_off_warns_and_refuses_noop(api):
    api.on("GET", f"/fleets/{FLEET}/proxy-policy", ok({"publicProxiesAllowed": True, "publicProxyMaxLife": 60}))
    pre = j(server.set_fleet_proxy_policy(FLEET, public_allowed=False))
    assert any("stop serving" in w for w in pre["warnings"])
    assert j(server.set_fleet_proxy_policy(FLEET, public_allowed=True, public_max_life_seconds=60, confirm=True))["error"].startswith("No change")
    assert "nothing to change" in j(server.set_fleet_proxy_policy(FLEET, confirm=True))["error"]
    assert "0 or more" in j(server.set_fleet_proxy_policy(FLEET, public_max_life_seconds=-1, confirm=True))["error"]
    assert not writes(api)


# ------------------------------------------------------ org proxy prefix ---


def test_get_org_proxy_prefix(api):
    api.on("GET", ORG_SETTINGS, ok({"prefix": "acme"}))
    out = j(server.get_org_proxy_prefix())
    assert out["prefix"] == "acme" and "acme" in out["summary"]
    api.on("GET", ORG_SETTINGS, ok({"prefix": None}))
    assert j(server.get_org_proxy_prefix())["prefix"] is None


def test_set_org_proxy_prefix_writes_and_reads_back(api):
    state: dict[str, Any] = {"prefix": None}
    api.on("GET", ORG_SETTINGS, lambda r: ok(dict(state)))
    api.on("PUT", ORG_SETTINGS, lambda r: (state.update(json.loads(r.content)), ok(dict(state)))[1])
    out = j(server.set_org_proxy_prefix("AcmeCorp"))
    req = writes(api, "PUT")[0]
    assert req.url.path.endswith(ORG_SETTINGS) and json.loads(req.content) == {"prefix": "acmecorp"}
    assert out["read_back"]["confirmed"] is True and out["after"] == "acmecorp"
    out = j(server.set_org_proxy_prefix(""))
    assert json.loads(writes(api, "PUT")[1].content) == {"prefix": None} and out["after"] is None


def test_set_org_proxy_prefix_validates_and_refuses_noop(api):
    for bad in ("ab", "x" * 21, "-acme", "acme_corp", "a--b", "acme-corp"):
        assert "prefix must be" in j(server.set_org_proxy_prefix(bad))["error"]
    assert not api.seen
    api.on("GET", ORG_SETTINGS, ok({"prefix": "acme"}))
    assert j(server.set_org_proxy_prefix("acme"))["error"].startswith("No change")
    assert not writes(api)


@pytest.mark.parametrize(
    "code, status, text",
    [("prefix_taken", 409, "Another organisation"), ("prefix_reserved", 400, "reserved")],
)
def test_set_org_proxy_prefix_maps_error_codes(api, code, status, text):
    api.on("GET", ORG_SETTINGS, ok({"prefix": None}))
    api.on("PUT", ORG_SETTINGS, httpx.Response(status, json={"msg": "no", "data": {"error_code": code}}))
    out = j(server.set_org_proxy_prefix("acme"))
    assert out["error_code"] == code and text in out["error"]
