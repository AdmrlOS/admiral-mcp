from __future__ import annotations

import asyncio
import json
import re
import sys
from types import SimpleNamespace

import httpx
import pytest

from admrl_mcp import browser, server
from admrl_mcp.client import AdmiralAPIError, AdmiralClient
from admrl_mcp.config import AUTH_BEARER, ConfigError, Settings


@pytest.fixture(autouse=True)
def _isolate():
    saved = server._client
    server._client = None
    browser.reset()
    yield
    asyncio.get_event_loop_policy()  # keep loop policy untouched
    if server._client is not None:
        server._client.close()
    server._client = saved
    browser.reset()


def _devices_body():
    return {
        "code": 200,
        "msg": "Success",
        "data": [{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "shop-01", "status": "online"}],
        "counts": {"online": 1, "offline": 0, "total": 1},
    }


def _install(handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    browser._state.transport = httpx.MockTransport(wrapped)
    return seen


# ------------------------------------------------------------ bearer mode ---


def test_bearer_settings_configured():
    assert not Settings("b", "", "", None, auth_mode=AUTH_BEARER).configured
    assert Settings("b", "", "", None, auth_mode=AUTH_BEARER, bearer_token="t").configured
    assert Settings("b", "id", "sec", None).configured  # PAT unchanged
    assert Settings("b", "id", "sec", None).auth_mode == "pat"


def test_pat_client_headers_unchanged():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_devices_body())

    client = AdmiralClient(Settings("https://x/v1", "tid", "sec", "org-1"), transport=httpx.MockTransport(handler))
    client.list_devices()
    h = seen[0].headers
    assert h["x-api-token-id"] == "tid" and h["x-api-secret-key"] == "sec" and h["x-organization-id"] == "org-1"
    assert "authorization" not in h
    assert h["user-agent"] == "admrl-mcp/0.1"


def test_bearer_client_sends_token_and_org_and_reads_token_per_request():
    seen = _install(lambda r: httpx.Response(200, json=_devices_body()))
    browser.configure("https://api.example/v1")
    browser.set_auth("tok-1", "org-A")
    client = server.get_client()
    client.list_devices()
    browser.set_auth("tok-2", "org-B")  # hourly refresh + org switch, same client
    assert server.get_client() is client
    client.list_devices()
    assert [r.headers["authorization"] for r in seen] == ["Bearer tok-1", "Bearer tok-2"]
    assert [r.headers["x-organization-id"] for r in seen] == ["org-A", "org-B"]
    assert all("x-api-token-id" not in r.headers and "x-api-secret-key" not in r.headers for r in seen)
    assert str(seen[0].url).startswith("https://api.example/v1/devices")


def test_bearer_without_token_is_config_error():
    _install(lambda r: httpx.Response(200, json={}))
    browser.set_auth("", "org")
    with pytest.raises(ConfigError):
        server.get_client().list_devices()


def test_bearer_org_required_message_has_no_env_hint():
    _install(lambda r: httpx.Response(200, json={}))
    browser.set_auth("t", None)
    with pytest.raises(AdmiralAPIError) as err:
        server.get_client().list_devices()
    assert "ADMRL_ORG_ID" not in str(err.value)


# ------------------------------------------------------------- transport ---


class FakeXhr:
    last: "FakeXhr | None" = None

    def __init__(self, status=200, headers="Content-Type: application/json\r\nContent-Encoding: gzip\r\nX-Admrl-Push: yes\r\n", body=b'{"ok":true}', fail=None):
        self.status, self._headers, self.response, self.fail = status, headers, body, fail
        self.sent_headers: dict[str, str] = {}
        self.opened = None
        self.sent = "unset"
        self.responseType = ""
        FakeXhr.last = self

    def open(self, method, url, is_async):
        self.opened = (method, url, is_async)

    def setRequestHeader(self, name, value):
        self.sent_headers[name] = value

    def send(self, body=None):
        if self.fail:
            raise RuntimeError(self.fail)
        self.sent = body

    def getAllResponseHeaders(self):
        return self._headers


def test_parse_response_headers():
    assert browser.parse_response_headers("A: 1\r\nb:  two: three \r\n\r\n") == [("A", "1"), ("b", "two: three")]
    assert browser.parse_response_headers("") == []


def test_xhr_transport_sync_xhr_headers_and_body():
    transport = browser.XhrTransport(lambda: FakeXhr())
    with httpx.Client(transport=transport, headers={"User-Agent": "x"}) as client:
        r = client.post(
            "https://api.example/v1/devices?a=1",
            json={"k": "v"},
            headers={"Authorization": "Bearer t", "X-Organization-ID": "o", "Accept": "application/json"},
        )
    x = FakeXhr.last
    assert x.opened == ("POST", "https://api.example/v1/devices?a=1", False)  # synchronous
    assert x.responseType == "arraybuffer"
    names = {k.lower() for k in x.sent_headers}
    assert {"authorization", "x-organization-id", "accept", "content-type"} <= names
    # forbidden request headers are never set
    assert not names & {"user-agent", "host", "content-length", "accept-encoding", "connection"}
    assert x.sent == b'{"k":"v"}' or json.loads(x.sent) == {"k": "v"}
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert r.headers["x-admrl-push"] == "yes"
    assert "content-encoding" not in r.headers  # body already decoded by the browser


def test_xhr_transport_get_sends_no_body_and_http_errors_pass_through():
    transport = browser.XhrTransport(lambda: FakeXhr(status=403, body=b'{"msg":"nope"}'))
    with httpx.Client(transport=transport) as client:
        r = client.get("https://api.example/x")
    assert FakeXhr.last.sent is None
    assert r.status_code == 403 and r.json()["msg"] == "nope"


def test_xhr_transport_network_errors_become_httpx_errors():
    with httpx.Client(transport=browser.XhrTransport(lambda: FakeXhr(status=0))) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("https://api.example/x")
    with httpx.Client(transport=browser.XhrTransport(lambda: FakeXhr(fail="NetworkError: boom"))) as client:
        with pytest.raises(httpx.ConnectError):
            client.get("https://api.example/x")
    with httpx.Client(transport=browser.XhrTransport(lambda: FakeXhr(fail="TimeoutError"))) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get("https://api.example/x")


# ------------------------------------------------------ sequential fallback ---


def test_parallel_map_sequential_on_emscripten(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("threads unavailable")

    monkeypatch.setattr(server, "ThreadPoolExecutor", Boom)
    assert server._parallel_map(lambda x: x * 2, [3, 1, 2]) == [6, 2, 4]


def test_parallel_map_threaded_elsewhere_keeps_order():
    assert server._parallel_map(lambda x: x + 1, list(range(20))) == list(range(1, 21))


# ------------------------------------------------------------- annotations ---


def _all_tools():
    return asyncio.run(server.mcp.list_tools())


def test_every_tool_has_explicit_read_only_hint():
    missing = [t.name for t in _all_tools() if t.annotations is None or t.annotations.readOnlyHint is None]
    assert missing == []


def test_mutating_tools_declare_destructive_hint():
    bad = [
        t.name
        for t in _all_tools()
        if t.annotations.readOnlyHint is False and t.annotations.destructiveHint is None
    ]
    assert bad == []


def test_known_mutations_are_not_read_only():
    by = {t.name: t.annotations for t in _all_tools()}
    for name in (
        "reboot_device",
        "start_memory_test",
        "cancel_memory_test",
        "change_workload_status",
        "patch_device_document",
        "create_rollout",
        "rollout_control",
        "adopt_local_override",
        "discard_local_override",
        "duplicate_fleet",
        "duplicate_configuration",
        "render_device_document",
    ):
        assert by[name].readOnlyHint is False, name
    for name in ("list_devices", "get_device_logs", "get_device_screenshot", "search"):
        assert by[name].readOnlyHint is True, name


# ----------------------------------------------------------- tool filter ---


def test_browser_server_excludes_unsupported_tools_only():
    full = {t.name for t in _all_tools()}
    assert set(browser.EXCLUDED_TOOLS) <= full  # a rename must update the filter
    shown = {t.name for t in asyncio.run(browser._server().list_tools())}
    assert shown == full - set(browser.EXCLUDED_TOOLS)
    assert not any(n.startswith("watch_") for n in shown)
    # the PAT/stdio server is untouched
    assert {t.name for t in _all_tools()} == full


def test_browser_text_never_mentions_pat():
    text = browser.browser_instructions()
    assert "Personal API Token" not in text
    descriptions = [t.description for t in asyncio.run(browser._server().list_tools())]
    blob = text + "\n" + "\n".join(descriptions)
    assert not re.search(r"\bPATs?\b|ADMRL_|X-API-", blob), re.findall(r".{30}\bPAT\b.{30}", blob)
    for excluded in browser.EXCLUDED_TOOLS:
        assert excluded not in text
    # the unmodified server text still carries it (rewrites are targeted)
    assert "Personal API Token" in server.INSTRUCTIONS


def test_rewrites_still_match_server_text():
    # Fails loudly if server wording drifts so a rewrite silently stops applying.
    blob = server.INSTRUCTIONS + "\n".join(
        t.description or "" for t in server.mcp._tool_manager.list_tools()
    )
    for old, _ in browser._TEXT_REWRITES:
        assert old in blob, old


# ------------------------------------------------ end to end (in-memory MCP) ---


def test_session_list_tools_server_info_and_call_tool():
    seen = _install(lambda r: httpx.Response(200, json=_devices_body()))

    async def go():
        try:
            browser.configure("https://api.example/v1")
            browser.set_auth("jwt-123", "org-9")
            info = json.loads(await browser.server_info())
            tools = json.loads(await browser.list_tools())
            result = json.loads(await browser.call_tool("list_devices", "{}"))
            browser.set_auth("jwt-456", "org-9")
            await browser.call_tool("list_devices", "{}")
            bad = json.loads(await browser.call_tool("watch_rollout", json.dumps({"rollout_id": "x"})))
            return info, tools, result, bad
        finally:
            await browser.aclose()

    info, tools, result, bad = asyncio.run(go())
    assert info["name"] == "admrl" and info["runtime"]["protocol"] == "session"
    assert "Personal API Token" not in info["instructions"]
    assert {"name", "description", "inputSchema", "annotations"} <= set(tools[0])
    assert len(tools) == 45 - len(browser.EXCLUDED_TOOLS)
    assert result["isError"] is False and result["content"][0]["type"] == "text"
    assert "shop-01" in result["content"][0]["text"]
    tokens = [r.headers["authorization"] for r in seen]
    assert tokens[0] == "Bearer jwt-123" and tokens[-1] == "Bearer jwt-456"
    assert set(tokens) == {"Bearer jwt-123", "Bearer jwt-456"}
    assert {r.headers["x-organization-id"] for r in seen} == {"org-9"}
    assert bad["isError"] is True  # excluded tools do not exist in the browser server


def test_direct_fallback_when_session_cannot_start(monkeypatch):
    seen = _install(lambda r: httpx.Response(200, json=_devices_body()))

    async def boom(ready, stop, box):
        box["error"] = RuntimeError("no memory streams here")
        ready.set()

    monkeypatch.setattr(browser, "_run_session", boom)

    async def go():
        browser.set_auth("t", "o")
        info = json.loads(await browser.server_info())
        tools = json.loads(await browser.list_tools())
        ok = json.loads(await browser.call_tool("list_devices", "{}"))
        err = json.loads(await browser.call_tool("nope", "{}"))
        return info, tools, ok, err

    info, tools, ok, err = asyncio.run(go())
    assert info["runtime"]["protocol"] == "direct"
    assert len(tools) == 45 - len(browser.EXCLUDED_TOOLS)
    assert ok["isError"] is False and "shop-01" in ok["content"][0]["text"]
    assert err["isError"] is True
    assert seen[0].headers["authorization"] == "Bearer t"


def test_tool_http_failure_is_an_error_result_not_an_exception():
    _install(lambda r: httpx.Response(401, json={"msg": "token expired"}))

    async def go():
        try:
            browser.set_auth("old", "o")
            return json.loads(await browser.call_tool("list_devices", "{}"))
        finally:
            await browser.aclose()

    res = asyncio.run(go())
    text = res["content"][0]["text"]
    assert "token expired" in text and "401" in text
