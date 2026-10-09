from __future__ import annotations

import asyncio
import contextlib
import json
import threading

import httpx
import pytest

from admrl_mcp import browser, hosted, server
from admrl_mcp.hosted import HostedConfig

PRM_URL = "https://mcp.admrl.co/.well-known/oauth-protected-resource"
WWW_AUTH = f'Bearer resource_metadata="{PRM_URL}", scope="admrl:read"'
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
}


@pytest.fixture(autouse=True)
def _restore_server_state():
    saved = (server._hosted_mode, server._client)
    yield
    server._hosted_mode, server._client = saved


@contextlib.asynccontextmanager
async def running(config: HostedConfig):
    app = hosted.create_app(config)
    async with app.app.router.lifespan_context(app.app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://mcp.admrl.co") as client:
            yield client


def rpc(method: str, params: dict | None = None, id_: int = 2) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params or {}}


def bearer(token: str) -> dict:
    return {**MCP_HEADERS, "Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- metadata ---


async def test_metadata_documents_match_contract():
    async with running(HostedConfig()) as c:
        for path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
            r = await c.get(path)
            assert r.status_code == 200
            assert r.json() == {
                "resource": "https://mcp.admrl.co/mcp",
                "authorization_servers": ["https://mcp.admrl.co"],
                "scopes_supported": ["admrl:read", "admrl:write"],
                "bearer_methods_supported": ["header"],
                "resource_name": "Admiral",
            }
        r = await c.get("/.well-known/oauth-authorization-server")
        assert r.json() == {
            "issuer": "https://mcp.admrl.co",
            "authorization_endpoint": "https://app.admrl.co/oauth/authorize",
            "token_endpoint": "https://api.admrl.co/v1/oauth/mcp/token",
            "registration_endpoint": "https://api.admrl.co/v1/oauth/mcp/register",
            "revocation_endpoint": "https://api.admrl.co/v1/oauth/mcp/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["admrl:read", "admrl:write"],
            "authorization_response_iss_parameter_supported": True,
        }


async def test_metadata_follows_env():
    cfg = HostedConfig.from_env(
        {
            "ADMRL_MCP_ISSUER": "https://mcp.stg.example/",
            "ADMRL_MCP_RESOURCE_URL": "https://mcp.stg.example/mcp",
            "ADMRL_MCP_AUTHORIZE_URL": "https://app.stg.example/oauth/authorize",
            "ADMRL_MCP_PUBLIC_API_BASE": "https://api.stg.example/v1",
        }
    )
    assert cfg.protected_resource_metadata()["authorization_servers"] == ["https://mcp.stg.example"]
    assert cfg.authorization_server_metadata()["token_endpoint"] == "https://api.stg.example/v1/oauth/mcp/token"
    assert cfg.www_authenticate == (
        'Bearer resource_metadata="https://mcp.stg.example/.well-known/oauth-protected-resource", scope="admrl:read"'
    )
    assert cfg.allowed_hosts == ()
    hosted.build_hosted_server(cfg)  # builds; resource host is allowed implicitly


async def test_healthz_needs_no_auth():
    async with running(HostedConfig()) as c:
        r = await c.get("/healthz")
        assert r.status_code == 200


# -------------------------------------------------------------------- auth ---


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer abc"},
        {"Authorization": "Bearer admrl_mcp_rt_refreshtoken"},
        {"Authorization": "Bearer admrl_mcp_at_"},
        {"Authorization": "Basic YTpi"},
        {"Authorization": "admrl_mcp_at_x"},
    ],
)
async def test_missing_or_malformed_bearer_is_401(headers):
    async with running(HostedConfig()) as c:
        r = await c.post("/mcp", json=INIT, headers={**MCP_HEADERS, **headers})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == WWW_AUTH


async def test_initialize_and_tools_list_exclude_watchers():
    async with running(HostedConfig()) as c:
        r = await c.post("/mcp", json=INIT, headers=bearer("admrl_mcp_at_fake"))
        assert r.status_code == 200
        assert r.json()["result"]["serverInfo"]["name"] == "admrl"
        r = await c.post("/mcp", json=rpc("tools/list"), headers=bearer("admrl_mcp_at_fake"))
        tools = {t["name"]: t for t in r.json()["result"]["tools"]}
    assert "list_devices" in tools
    assert not [n for n in tools if n.startswith("watch_")]
    assert set(hosted.HOSTED_EXCLUDED_TOOLS) == {"watch_device_state", "watch_rollout"}
    assert all(t["annotations"]["readOnlyHint"] is not None for t in tools.values())
    blob = json.dumps(tools)
    assert "this PAT" not in blob and "ADMRL_ORG_ID" not in blob


async def test_dns_rebinding_blocks_other_hosts():
    async with running(HostedConfig()) as c:
        r = await c.post("/mcp", json=INIT, headers={**bearer("admrl_mcp_at_fake"), "Host": "evil.example"})
        assert r.status_code == 421


async def test_body_size_limit():
    cfg = HostedConfig(max_body_bytes=2048)
    async with running(cfg) as c:
        big = {**INIT, "params": {**INIT["params"], "pad": "x" * 5000}}
        r = await c.post("/mcp", json=big, headers=bearer("admrl_mcp_at_fake"))
        assert r.status_code == 413


# ------------------------------------------------------- credential scoping ---


async def test_concurrent_requests_use_their_own_tokens():
    seen: list[httpx.Request] = []
    barrier = threading.Barrier(2, timeout=10)  # both backend calls must be in flight together

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        barrier.wait()  # deadlocks (BrokenBarrierError) if sync tools block the event loop
        token = request.headers["authorization"].split(" ", 1)[1]
        return httpx.Response(200, json={"code": 200, "msg": "Success", "data": [{"id": f"org-of-{token}", "name": token}]})

    cfg = HostedConfig(api_transport=httpx.MockTransport(handler), api_base="https://api.test/v1")
    async with running(cfg) as c:
        call = rpc("tools/call", {"name": "list_organisations", "arguments": {}})
        r1, r2 = await asyncio.gather(
            c.post("/mcp", json=call, headers=bearer("admrl_mcp_at_AAAA")),
            c.post("/mcp", json=call, headers=bearer("admrl_mcp_at_BBBB")),
        )
    texts = {}
    for tok, r in (("AAAA", r1), ("BBBB", r2)):
        assert r.status_code == 200, r.text
        texts[tok] = r.json()["result"]["content"][0]["text"]
    assert "org-of-admrl_mcp_at_AAAA" in texts["AAAA"] and "BBBB" not in texts["AAAA"]
    assert "org-of-admrl_mcp_at_BBBB" in texts["BBBB"] and "AAAA" not in texts["BBBB"]
    assert len(seen) == 2
    assert {r.headers["authorization"] for r in seen} == {"Bearer admrl_mcp_at_AAAA", "Bearer admrl_mcp_at_BBBB"}
    for r in seen:
        assert "x-api-token-id" not in r.headers and "x-api-secret-key" not in r.headers
        assert "x-organization-id" not in r.headers  # no org given -> backend defaults to the grant's


async def test_org_header_sent_only_when_tool_gets_organization_id():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [], "counts": {}})

    cfg = HostedConfig(api_transport=httpx.MockTransport(handler), api_base="https://api.test/v1")
    async with running(cfg) as c:
        for args in ({}, {"organization_id": "org-9"}):
            call = rpc("tools/call", {"name": "list_devices", "arguments": args})
            r = await c.post("/mcp", json=call, headers=bearer("admrl_mcp_at_AAAA"))
            assert r.status_code == 200, r.text
    assert "x-organization-id" not in seen[0].headers
    assert seen[-1].headers["x-organization-id"] == "org-9"


async def test_no_shared_client_in_hosted_mode():
    hosted.create_app(HostedConfig())
    assert server._hosted_mode is True
    from admrl_mcp.config import ConfigError

    with pytest.raises(ConfigError):
        server.get_client()
    assert server._client is None


async def test_backend_401_becomes_http_401_with_challenge():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"msg": "expired"})

    cfg = HostedConfig(api_transport=httpx.MockTransport(handler), api_base="https://api.test/v1")
    async with running(cfg) as c:
        call = rpc("tools/call", {"name": "list_organisations", "arguments": {}})
        r = await c.post("/mcp", json=call, headers=bearer("admrl_mcp_at_AAAA"))
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == WWW_AUTH


async def test_backend_403_is_a_tool_result_with_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"msg": "write scope required"})

    cfg = HostedConfig(api_transport=httpx.MockTransport(handler), api_base="https://api.test/v1")
    async with running(cfg) as c:
        call = rpc("tools/call", {"name": "list_organisations", "arguments": {}})
        r = await c.post("/mcp", json=call, headers=bearer("admrl_mcp_at_AAAA"))
    assert r.status_code == 200
    assert "write scope required" in r.json()["result"]["content"][0]["text"]


async def test_tokens_are_not_logged(caplog):
    caplog.set_level("DEBUG")
    cfg = HostedConfig(api_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})))
    async with running(cfg) as c:
        await c.post("/mcp", json=INIT, headers=bearer("admrl_mcp_at_SECRETVALUE"))
        await c.post("/mcp", json=INIT, headers={**MCP_HEADERS, "Authorization": "Bearer nope"})
    assert "SECRETVALUE" not in caplog.text
    fmt = hosted.JsonFormatter()
    assert all(isinstance(json.loads(fmt.format(rec)), dict) for rec in caplog.records)


def test_parallel_map_propagates_context():
    holder = object()
    token = server._request_client.set(holder)
    try:
        out = server._parallel_map(lambda i: (server._request_client.get() is holder, i), list(range(20)))
    finally:
        server._request_client.reset(token)
    assert out == [(True, i) for i in range(20)]
    # and nothing leaks into the pool threads' own contexts afterwards
    assert server._parallel_map(lambda i: server._request_client.get(), [1, 2]) == [None, None]


async def test_sync_tools_run_off_the_event_loop():
    srv = hosted.build_hosted_server(HostedConfig())
    tool = srv._tool_manager.get_tool("list_organisations")
    assert tool.is_async
    original = server.mcp._tool_manager.get_tool("list_organisations")
    assert not original.is_async  # stdio tool untouched


# ----------------------------------------------- other builds are unchanged ---


def test_stdio_and_browser_builds_unchanged():
    names = {t.name for t in server.mcp._tool_manager.list_tools()}
    assert {"watch_device_state", "watch_rollout"} <= names
    assert all(not t.is_async for t in server.mcp._tool_manager.list_tools())
    b = {t.name for t in browser.build_browser_server()._tool_manager.list_tools()}
    assert b == names - set(browser.EXCLUDED_TOOLS)
    h = {t.name for t in hosted.build_hosted_server(HostedConfig())._tool_manager.list_tools()}
    assert h == b


# ------------------------------------------------------------------- v2 ---

JWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1MSJ9.c2lnLW5hdHVyZV8x"
HOP = "hop-secret-value"


def _echo_backend(seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [], "counts": {}})

    return handler


async def _list_devices(c, headers, args=None):
    call = rpc("tools/call", {"name": "list_devices", "arguments": args or {}})
    return await c.post("/mcp", json=call, headers=headers)


async def test_internal_hop_headers_on_every_backend_call():
    seen: list[httpx.Request] = []
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend(seen)), api_base="https://api.test/v1", internal_token=HOP)
    async with running(cfg) as c:
        h = {**bearer("admrl_mcp_at_AAAA"), "CF-Connecting-IP": "203.0.113.7", "User-Agent": "Claude/9"}
        r = await _list_devices(c, h)
        assert r.status_code == 200, r.text
        first, seen[:] = list(seen), []
        r = await _list_devices(c, {**h, "CF-Connecting-IP": "2001:db8::9", "User-Agent": "U" * 1000})
        assert r.status_code == 200, r.text
    assert first and seen  # a tool call may make several backend requests: every one carries the hop
    for req in first:
        assert req.headers["x-admrl-internal-token"] == HOP
        assert req.headers["x-admrl-client-ip"] == "203.0.113.7"
        assert req.headers["x-admrl-client-ua"] == "Claude/9"
        assert req.headers["authorization"] == "Bearer admrl_mcp_at_AAAA"
    for req in seen:
        assert req.headers["x-admrl-client-ip"] == "2001:db8::9"
        assert req.headers["x-admrl-client-ua"] == "U" * 256


async def test_internal_hop_client_ip_falls_back_to_peer_and_rejects_junk():
    seen: list[httpx.Request] = []
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend(seen)), api_base="https://api.test/v1", internal_token=HOP)
    app = hosted.create_app(cfg)
    async with app.app.router.lifespan_context(app.app):
        transport = httpx.ASGITransport(app=app, client=("198.51.100.4", 1234))
        async with httpx.AsyncClient(transport=transport, base_url="https://mcp.admrl.co") as c:
            await _list_devices(c, bearer("admrl_mcp_at_AAAA"))
            await _list_devices(c, {**bearer("admrl_mcp_at_AAAA"), "CF-Connecting-IP": "not-an-ip"})
    assert seen and {r.headers["x-admrl-client-ip"] for r in seen} == {"198.51.100.4"}


async def test_no_hop_headers_without_token_and_no_client_spoofing():
    seen: list[httpx.Request] = []
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend(seen)), api_base="https://api.test/v1")
    async with running(cfg) as c:
        await _list_devices(c, {**bearer("admrl_mcp_at_AAAA"), "X-Admrl-Internal-Token": "x", "X-Admrl-Client-IP": "1.2.3.4"})
    assert not any(k.lower().startswith("x-admrl-") for k in seen[0].headers)


async def test_hop_secret_never_logged(caplog):
    caplog.set_level("DEBUG")
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend([])), api_base="https://api.test/v1", internal_token=HOP)
    async with running(cfg) as c:
        await _list_devices(c, bearer("admrl_mcp_at_AAAA"))
    assert HOP not in caplog.text and HOP not in repr(cfg)


def test_config_reads_hop_and_cors_env():
    cfg = HostedConfig.from_env(
        {"ADMRL_MCP_INTERNAL_TOKEN": " tok ", "ADMRL_MCP_CORS_ORIGINS": "https://app.admrl.co, https://staging.example.com/"}
    )
    assert cfg.internal_token == "tok"
    assert cfg.cors_origins == ("https://app.admrl.co", "https://staging.example.com")
    assert "https://app.admrl.co" in cfg.allowed_origins and "https://staging.example.com" in cfg.allowed_origins
    assert "http://localhost:5173" not in HostedConfig.from_env({}).cors_origins
    dev = HostedConfig.from_env({"ADMRL_MCP_CORS_ALLOW_LOCALHOST": "true"})
    assert dev.cors_origins == ("http://localhost:5173",) and "http://localhost:5173" in dev.allowed_origins


@pytest.mark.parametrize(
    "token,ok",
    [
        (JWT, True),
        ("admrl_mcp_at_abc", True),
        ("a.b.c", True),
        ("a.b", False),
        ("a.b.c.d", False),
        ("a..c", False),
        ("a.b.c!", False),
        ("admrl_mcp_rt_abc", False),
        ("admrl_cli_abc", False),
        ("junk", False),
        ("a.b.c d", False),
    ],
)
def test_parse_bearer_shapes(token, ok):
    assert (hosted.parse_bearer(f"Bearer {token}") == token) is ok


async def test_jwt_bearer_accepted_and_forwarded_with_org():
    seen: list[httpx.Request] = []
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend(seen)), api_base="https://api.test/v1")
    async with running(cfg) as c:
        r = await _list_devices(c, {**bearer(JWT), "X-Organization-ID": "org-7"})
        assert r.status_code == 200, r.text
        a, seen[:] = list(seen), []
        r = await _list_devices(c, {**bearer(JWT), "X-Organization-ID": "org-7"}, {"organization_id": "org-9"})
        b, seen[:] = list(seen), []
        r = await _list_devices(c, bearer("admrl_mcp_at_AAAA"))
        r = await c.post("/mcp", json=INIT, headers=bearer("not-a-token"))
        assert r.status_code == 401 and r.headers["www-authenticate"] == WWW_AUTH
    assert a and b and seen
    assert all(r.headers["authorization"] == f"Bearer {JWT}" and r.headers["x-organization-id"] == "org-7" for r in a)
    assert all(r.headers["x-organization-id"] == "org-9" for r in b)  # explicit tool arg wins
    assert all("x-organization-id" not in r.headers for r in seen)


async def test_org_header_does_not_leak_between_requests():
    seen: list[httpx.Request] = []
    cfg = HostedConfig(api_transport=httpx.MockTransport(_echo_backend(seen)), api_base="https://api.test/v1", internal_token=HOP)
    async with running(cfg) as c:
        await asyncio.gather(
            _list_devices(c, {**bearer(JWT), "X-Organization-ID": "org-A", "CF-Connecting-IP": "1.1.1.1"}),
            _list_devices(c, {**bearer("admrl_mcp_at_BBBB"), "X-Organization-ID": "org-B", "CF-Connecting-IP": "2.2.2.2"}),
            _list_devices(c, {**bearer("admrl_mcp_at_CCCC"), "CF-Connecting-IP": "3.3.3.3"}),
        )
    expect = {
        JWT: ("org-A", "1.1.1.1"),
        "admrl_mcp_at_BBBB": ("org-B", "2.2.2.2"),
        "admrl_mcp_at_CCCC": (None, "3.3.3.3"),
    }
    assert {r.headers["authorization"].split()[1] for r in seen} == set(expect)
    for r in seen:
        org, ip = expect[r.headers["authorization"].split()[1]]
        assert r.headers.get("x-organization-id") == org and r.headers["x-admrl-client-ip"] == ip


CORS_CFG = dict(cors_origins=("https://app.admrl.co", "https://staging.example.com"))


async def test_cors_preflight_is_unauthenticated():
    cfg = HostedConfig(allowed_origins=(*hosted.DEFAULT_ALLOWED_ORIGINS, *CORS_CFG["cors_origins"]), **CORS_CFG)
    async with running(cfg) as c:
        r = await c.options(
            "/mcp",
            headers={
                "Origin": "https://app.admrl.co",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type,x-organization-id",
            },
        )
        assert r.status_code == 204
        assert r.headers["access-control-allow-origin"] == "https://app.admrl.co"
        allow = r.headers["access-control-allow-headers"].lower()
        for h in ("authorization", "content-type", "x-organization-id", "mcp-session-id", "mcp-protocol-version", "accept"):
            assert h in allow
        assert "access-control-allow-credentials" not in r.headers
        r = await c.options("/mcp", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 403 and "access-control-allow-origin" not in r.headers


async def test_cors_headers_on_responses_only_for_allowed_origins():
    cfg = HostedConfig(allowed_origins=(*hosted.DEFAULT_ALLOWED_ORIGINS, *CORS_CFG["cors_origins"]), **CORS_CFG)
    async with running(cfg) as c:
        r = await c.post("/mcp", json=INIT, headers={**MCP_HEADERS, "Origin": "https://staging.example.com"})
        assert r.status_code == 401  # CORS headers present even on the challenge, so the browser can read it
        assert r.headers["access-control-allow-origin"] == "https://staging.example.com"
        assert "www-authenticate" in r.headers["access-control-expose-headers"].lower()
        assert "mcp-session-id" in r.headers["access-control-expose-headers"].lower()
        r = await c.post("/mcp", json=INIT, headers={**bearer(JWT), "Origin": "https://app.admrl.co"})
        assert r.status_code == 200, r.text  # Origin allow-list includes the CORS origins
        assert r.headers["access-control-allow-origin"] == "https://app.admrl.co"
        r = await c.post("/mcp", json=INIT, headers={**bearer(JWT), "Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in r.headers


async def test_no_cors_configured_means_no_cors_headers():
    async with running(HostedConfig()) as c:
        r = await c.options("/mcp", headers={"Origin": "https://app.admrl.co", "Access-Control-Request-Method": "POST"})
        assert r.status_code == 403
        r = await c.post("/mcp", json=INIT, headers={**MCP_HEADERS, "Origin": "https://app.admrl.co"})
        assert "access-control-allow-origin" not in r.headers


def test_stdio_client_gets_no_hop_headers(monkeypatch):
    from admrl_mcp.client import AdmiralClient
    from admrl_mcp.config import Settings

    monkeypatch.setenv("ADMRL_MCP_INTERNAL_TOKEN", HOP)
    seen: list[httpx.Request] = []
    cl = AdmiralClient(Settings(api_base="https://api.test/v1", token_id="i", secret_key="s", org_id="o"),
                       transport=httpx.MockTransport(_echo_backend(seen)))
    cl.get("devices")
    assert not any(k.lower().startswith("x-admrl-") for k in seen[0].headers)
