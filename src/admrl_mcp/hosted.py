"""Hosted admrl-mcp: a stateless streamable-HTTP MCP server acting as an OAuth 2.1 protected resource.

This process neither mints nor validates tokens beyond their shape:
every ``/mcp`` request must carry ``Authorization: Bearer admrl_mcp_at_...`` and that token is
forwarded, per request, to the Admiral API (``ADMRL_API_BASE``), which validates it and enforces
scope and organisation.

v2: ``/mcp`` also accepts JWT-shaped dashboard bearers (shape only; the backend validates), forwards
``X-Organization-ID``, and answers CORS for ``ADMRL_MCP_CORS_ORIGINS``. When ``ADMRL_MCP_INTERNAL_TOKEN`` is
set, every backend call carries ``X-Admrl-Internal-Token`` / ``-Client-IP`` / ``-Client-UA`` (trusted hop;
hosted mode only; the secret is never logged).

Multi-tenant safety:

* The caller's token lives only in a request-scoped holder (a ``contextvars.ContextVar`` set by the
  bearer middleware). ``server.get_client()`` returns that holder's client; there is no module-level
  client or credential in hosted mode, and ``get_client()`` raises if no request context exists.
* FastMCP 1.x calls sync tool functions directly on the event loop, so every tool is wrapped to run
  in a worker thread via ``anyio.to_thread.run_sync`` (which copies the context). Worker pools inside
  tools (``server._parallel_map``) copy the context per task.
* Tools whose backend call returns 401 make the whole HTTP response a 401 with the contract's
  ``WWW-Authenticate`` header so the client refreshes its token (responses are JSON, never SSE, so the
  status can still be changed after the tool ran).
"""

from __future__ import annotations

import functools
import ipaddress
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import anyio.to_thread
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import server
from .client import AdmiralClient
from .config import AUTH_BEARER, DEFAULT_API_BASE, Settings
from .toolset import clone_tools, instructions_with_auth_paragraph

log = logging.getLogger("admrl_mcp.hosted")

TOKEN_PREFIX = "admrl_mcp_at_"
SCOPES = ["admrl:read", "admrl:write"]
DEFAULT_RESOURCE_URL = "https://mcp.admrl.co/mcp"
DEFAULT_ISSUER = "https://mcp.admrl.co"
DEFAULT_AUTHORIZE_URL = "https://app.admrl.co/oauth/authorize"
DEFAULT_PUBLIC_API_BASE = "https://api.admrl.co/v1"
DEFAULT_MAX_BODY_BYTES = 1024 * 1024
LOCALHOST_DEV_ORIGIN = "http://localhost:5173"
MAX_CLIENT_UA = 256
MAX_BEARER_LEN = 8192
# Three non-empty base64url segments (dashboard Firebase ID token). Shape only; the backend validates.
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
CORS_ALLOW_HEADERS = "Authorization, Content-Type, X-Organization-ID, Mcp-Session-Id, Mcp-Protocol-Version, Accept"
CORS_EXPOSE_HEADERS = "Mcp-Session-Id, WWW-Authenticate"
CORS_ALLOW_METHODS = "GET, POST, DELETE, OPTIONS"
# Browser-based MCP clients send an Origin header; the SDK rejects origins not listed here.
DEFAULT_ALLOWED_ORIGINS = ("https://claude.ai", "https://claude.com", "https://chatgpt.com")

# Tools that cannot work over stateless request/response HTTP: SSE streams (and, were there any,
# tools that write files on the server host).
HOSTED_EXCLUDED_TOOLS: dict[str, str] = {
    "watch_device_state": "SSE stream: not available over stateless HTTP.",
    "watch_rollout": "SSE stream: not available over stateless HTTP.",
}

_HOSTED_AUTH_PARAGRAPH = (
    "You are connected to Admiral through an OAuth-authorised connector, acting for one signed-in user in the "
    "organisation chosen when access was granted. You only see and change what that grant allows; actions that "
    "change things need the write permission granted at consent and fail with a 403 otherwise. organization_id "
    "can normally be omitted. Live log tailing and event streaming are not available; "
    "use the historical log and state tools."
)


# ------------------------------------------------------------------ config ---


def _env(env: dict[str, str], key: str, default: str) -> str:
    value = (env.get(key) or "").strip()
    return value or default


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


@dataclass(frozen=True)
class HostedConfig:
    resource_url: str = DEFAULT_RESOURCE_URL
    issuer: str = DEFAULT_ISSUER
    authorize_url: str = DEFAULT_AUTHORIZE_URL
    public_api_base: str = DEFAULT_PUBLIC_API_BASE
    api_base: str = DEFAULT_API_BASE
    allowed_hosts: tuple[str, ...] = ()  # extra Host values; the resource URL host is always allowed
    allowed_origins: tuple[str, ...] = DEFAULT_ALLOWED_ORIGINS
    cors_origins: tuple[str, ...] = ()  # browser origins allowed to call /mcp (dashboard)
    internal_token: str = field(default="", repr=False)  # shared secret for the trusted backend hop
    dns_rebinding_protection: bool = True
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_threads: int = 64
    # Tests inject an httpx transport for the Admiral API; never set in production.
    api_transport: httpx.BaseTransport | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> HostedConfig:
        e = dict(os.environ if env is None else env)
        resource = _env(e, "ADMRL_MCP_RESOURCE_URL", DEFAULT_RESOURCE_URL).rstrip("/")
        issuer = _env(e, "ADMRL_MCP_ISSUER", DEFAULT_ISSUER).rstrip("/")
        public_api = _env(e, "ADMRL_MCP_PUBLIC_API_BASE", DEFAULT_PUBLIC_API_BASE).rstrip("/")
        # Extra Host values (e.g. an in-cluster service name) are an explicit opt-in, never "*".
        hosts = _csv(e.get("ADMRL_MCP_ALLOWED_HOSTS"))
        cors = [o.rstrip("/") for o in _csv(e.get("ADMRL_MCP_CORS_ORIGINS"))]
        if _env(e, "ADMRL_MCP_CORS_ALLOW_LOCALHOST", "false").lower() in ("1", "true", "yes", "on"):
            cors.append(LOCALHOST_DEV_ORIGIN)
        cors = list(dict.fromkeys(cors))
        origins = [*DEFAULT_ALLOWED_ORIGINS, issuer, *_csv(e.get("ADMRL_MCP_ALLOWED_ORIGINS")), *cors]
        return cls(
            resource_url=resource,
            issuer=issuer,
            authorize_url=_env(e, "ADMRL_MCP_AUTHORIZE_URL", DEFAULT_AUTHORIZE_URL),
            public_api_base=public_api,
            api_base=_env(e, "ADMRL_API_BASE", DEFAULT_API_BASE).rstrip("/"),
            allowed_hosts=tuple(dict.fromkeys(hosts)),
            allowed_origins=tuple(dict.fromkeys(origins)),
            cors_origins=tuple(cors),
            internal_token=(e.get("ADMRL_MCP_INTERNAL_TOKEN") or "").strip(),
            dns_rebinding_protection=_env(e, "ADMRL_MCP_DNS_REBINDING_PROTECTION", "true").lower()
            not in ("0", "false", "no", "off"),
            max_body_bytes=int(_env(e, "ADMRL_MCP_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))),
            max_threads=int(_env(e, "ADMRL_MCP_MAX_THREADS", "64")),
        )

    @property
    def resource_metadata_url(self) -> str:
        parts = urlsplit(self.resource_url)
        return f"{parts.scheme}://{parts.netloc}/.well-known/oauth-protected-resource"

    @property
    def www_authenticate(self) -> str:
        return f'Bearer resource_metadata="{self.resource_metadata_url}", scope="admrl:read"'

    def protected_resource_metadata(self) -> dict[str, Any]:
        return {
            "resource": self.resource_url,
            "authorization_servers": [self.issuer],
            "scopes_supported": list(SCOPES),
            "bearer_methods_supported": ["header"],
            "resource_name": "Admiral",
        }

    def authorization_server_metadata(self) -> dict[str, Any]:
        base = self.public_api_base
        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.authorize_url,
            "token_endpoint": f"{base}/oauth/mcp/token",
            "registration_endpoint": f"{base}/oauth/mcp/register",
            "revocation_endpoint": f"{base}/oauth/mcp/revoke",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": list(SCOPES),
            "authorization_response_iss_parameter_supported": True,
        }


# ------------------------------------------------------- request credentials ---


class RequestCredentials:
    """One request's token and (lazily created) Admiral client. Never shared between requests."""

    def __init__(
        self,
        token: str,
        config: HostedConfig,
        *,
        client_ip: str = "",
        client_ua: str = "",
        org_id: str = "",
    ):
        self._token = token
        self._config = config
        self._client_ip = client_ip
        self._client_ua = client_ua
        self._org_id = org_id
        self._client: AdmiralClient | None = None
        self._lock = threading.Lock()
        self.backend_unauthorized = False

    def get(self) -> AdmiralClient:
        with self._lock:
            if self._client is None:
                settings = Settings(
                    api_base=self._config.api_base,
                    token_id="",
                    secret_key="",
                    org_id=None,
                    auth_mode=AUTH_BEARER,
                    bearer_token=self._token,
                    org_optional=True,
                )
                client = AdmiralClient(settings, transport=self._config.api_transport)
                client._client.event_hooks["request"].append(self._decorate_request)
                client._client.event_hooks["response"].append(self._note_response)
                self._client = client
            return self._client

    def _decorate_request(self, request: httpx.Request) -> None:
        """Hosted-only headers for every backend call (stdio/browser clients never get these)."""
        if self._config.internal_token:
            request.headers["X-Admrl-Internal-Token"] = self._config.internal_token
            if self._client_ip:
                request.headers["X-Admrl-Client-IP"] = self._client_ip
            if self._client_ua:
                request.headers["X-Admrl-Client-UA"] = self._client_ua
        # Org pinned by the caller (dashboard) unless the tool call named one explicitly.
        if self._org_id and "x-organization-id" not in request.headers:
            request.headers["X-Organization-ID"] = self._org_id

    def _note_response(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            self.backend_unauthorized = True

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


# -------------------------------------------------------------------- ASGI ---


def parse_bearer(header: str | None) -> str | None:
    """The token if ``header`` is ``Bearer admrl_mcp_at_<non-empty>`` or a JWT-shaped bearer, else None."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or len(token) > MAX_BEARER_LEN:
        return None
    if any(c.isspace() for c in token):
        return None
    if token.startswith(TOKEN_PREFIX):
        return token if len(token) > len(TOKEN_PREFIX) else None
    return token if _JWT_RE.match(token) else None


def _valid_ip(value: str | None) -> str:
    try:
        return str(ipaddress.ip_address((value or "").split(",")[0].strip()))
    except ValueError:
        return ""


def client_ip_from_scope(scope: Any, headers: dict[bytes, bytes]) -> str:
    """CF-Connecting-IP (the mcp host is behind the Cloudflare tunnel), else the TCP peer."""
    ip = _valid_ip(headers.get(b"cf-connecting-ip", b"").decode("latin-1"))
    if ip:
        return ip
    client = scope.get("client")
    return _valid_ip(client[0]) if client else ""


def _log(event: str, **fields: Any) -> None:
    log.info(event, extra={"fields": fields})


class BearerMiddleware:
    """Pure-ASGI bearer gate + per-request credential scope for ``/mcp``."""

    def __init__(self, app: Any, config: HostedConfig, protected_prefix: str = "/mcp"):
        self.app = app
        self.config = config
        self.prefix = protected_prefix

    def _is_protected(self, path: str) -> bool:
        return path == self.prefix or path.startswith(self.prefix + "/")

    async def _unauthorized(self, send: Any, detail: str) -> None:
        body = json.dumps({"error": "unauthorized", "error_description": detail}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", self.config.www_authenticate.encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    def _cors_origin(self, headers: dict[bytes, bytes]) -> str:
        origin = headers.get(b"origin", b"").decode("latin-1")
        return origin if origin and origin in self.config.cors_origins else ""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self._is_protected(scope["path"]):
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        cors_origin = self._cors_origin(headers)
        if scope["method"] == "OPTIONS" and b"origin" in headers and b"access-control-request-method" in headers:
            # CORS preflight: never authenticated.
            if not cors_origin:
                await send({"type": "http.response.start", "status": 403, "headers": [(b"content-length", b"0")]})
                await send({"type": "http.response.body", "body": b""})
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [
                        (b"access-control-allow-origin", cors_origin.encode()),
                        (b"access-control-allow-methods", CORS_ALLOW_METHODS.encode()),
                        (b"access-control-allow-headers", CORS_ALLOW_HEADERS.encode()),
                        (b"access-control-max-age", b"600"),
                        (b"vary", b"Origin"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        if cors_origin:
            inner_send = send

            async def send(message: Any) -> None:  # noqa: F811 - wraps the ASGI send with CORS headers
                if message["type"] == "http.response.start":
                    message = {
                        **message,
                        "headers": [
                            *message.get("headers", []),
                            (b"access-control-allow-origin", cors_origin.encode()),
                            (b"access-control-expose-headers", CORS_EXPOSE_HEADERS.encode()),
                            (b"vary", b"Origin"),
                        ],
                    }
                await inner_send(message)

        await self._handle(scope, receive, send, headers)

    async def _handle(self, scope: Any, receive: Any, send: Any, headers: dict[bytes, bytes]) -> None:
        started = time.monotonic()
        request_id = uuid.uuid4().hex[:12]
        status = {"code": 0}

        def note(message: Any) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]

        token = parse_bearer(headers.get(b"authorization", b"").decode("latin-1"))
        if token is None:
            async def send_401(message: Any) -> None:
                note(message)
                await send(message)

            await self._unauthorized(send_401, "A bearer access token is required.")
            _log("request", id=request_id, method=scope["method"], path=scope["path"], status=401,
                 reason="missing_or_malformed_bearer", ms=round((time.monotonic() - started) * 1000))
            return

        creds = RequestCredentials(
            token,
            self.config,
            client_ip=client_ip_from_scope(scope, headers),
            client_ua=headers.get(b"user-agent", b"").decode("latin-1").strip()[:MAX_CLIENT_UA],
            org_id=headers.get(b"x-organization-id", b"").decode("latin-1").strip()[:128],
        )
        var_token = server._request_client.set(creds)
        pending_start: dict[str, Any] | None = None
        replaced = False

        async def guarded_send(message: Any) -> None:
            # Hold the response head until the first body chunk; if the backend rejected the token
            # while a tool ran, answer 401 + WWW-Authenticate instead so the client refreshes.
            nonlocal pending_start, replaced
            if message["type"] == "http.response.start":
                pending_start = message
                return
            if message["type"] == "http.response.body":
                if replaced:
                    return
                if pending_start is not None:
                    if creds.backend_unauthorized:
                        replaced = True
                        pending_start = None
                        note({"type": "http.response.start", "status": 401})
                        await self._unauthorized(send, "The access token was rejected; re-authenticate.")
                        return
                    note(pending_start)
                    await send(pending_start)
                    pending_start = None
            await send(message)

        try:
            await self.app(scope, receive, guarded_send)
        except Exception:
            log.exception("unhandled error", extra={"fields": {"id": request_id, "path": scope["path"]}})
            raise
        finally:
            server._request_client.reset(var_token)
            await anyio.to_thread.run_sync(creds.close)
            _log(
                "request",
                id=request_id,
                method=scope["method"],
                path=scope["path"],
                status=status["code"],
                backend_unauthorized=creds.backend_unauthorized,
                ms=round((time.monotonic() - started) * 1000),
            )


# ----------------------------------------------------------- server building ---


def _offload(tool: Any) -> Any:
    """Copy of ``tool`` whose sync function runs in a worker thread (context copied by anyio)."""
    if tool.is_async:
        return tool
    fn = tool.fn

    @functools.wraps(fn)
    async def run(**kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

    return tool.model_copy(update={"fn": run, "is_async": True})


def build_hosted_server(config: HostedConfig) -> FastMCP:
    """A stateless, JSON-response FastMCP with the hosted-safe tool subset."""
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=config.dns_rebinding_protection,
        allowed_hosts=list(dict.fromkeys([urlsplit(config.resource_url).netloc, *config.allowed_hosts])),
        allowed_origins=list(config.allowed_origins),
    )
    srv = FastMCP(
        name="admrl",
        instructions=instructions_with_auth_paragraph(_HOSTED_AUTH_PARAGRAPH),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
        transport_security=security,
        max_request_body_size=config.max_body_bytes,
    )
    clone_tools(srv, HOSTED_EXCLUDED_TOOLS)
    manager = srv._tool_manager
    manager._tools = {name: _offload(tool) for name, tool in manager._tools.items()}
    return srv


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel((level or os.environ.get("ADMRL_MCP_LOG_LEVEL") or "INFO").upper())
    # URLs and headers of backend calls stay out of logs.
    for noisy in ("httpx", "httpcore", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _json(data: dict[str, Any]) -> JSONResponse:
    return JSONResponse(
        data,
        headers={"Cache-Control": "public, max-age=300", "Access-Control-Allow-Origin": "*"},
    )


def create_app(config: HostedConfig | None = None) -> Any:
    """The ASGI app: ``/mcp`` (bearer-gated), ``/healthz`` and the OAuth metadata documents."""
    config = config or HostedConfig.from_env()
    server._hosted_mode = True  # from here on get_client() only ever returns a request-scoped client
    if not config.internal_token:
        log.warning("ADMRL_MCP_INTERNAL_TOKEN is unset: backend calls carry no trusted client IP/UA")
    srv = build_hosted_server(config)
    app: Starlette = srv.streamable_http_app()

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    async def protected_resource(_: Request) -> Response:
        return _json(config.protected_resource_metadata())

    async def authorization_server(_: Request) -> Response:
        return _json(config.authorization_server_metadata())

    app.router.routes.extend(
        [
            Route("/healthz", healthz, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource", protected_resource, methods=["GET"]),
            Route("/.well-known/oauth-protected-resource/mcp", protected_resource, methods=["GET"]),
            Route("/.well-known/oauth-authorization-server", authorization_server, methods=["GET"]),
        ]
    )
    # Worker threads for sync tools (anyio's default limiter is 40).
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(a: Any):
        anyio.to_thread.current_default_thread_limiter().total_tokens = config.max_threads
        async with original_lifespan(a):
            yield

    app.router.lifespan_context = lifespan
    return BearerMiddleware(app, config)


def main() -> None:
    import uvicorn

    configure_logging()
    host = os.environ.get("ADMRL_MCP_HOST", "0.0.0.0")  # noqa: S104 - container entrypoint
    port = int(os.environ.get("ADMRL_MCP_PORT", "8080"))
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "*"),
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
