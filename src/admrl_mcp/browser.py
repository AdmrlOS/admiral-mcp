"""Run the Admiral MCP server inside a browser (Pyodide in a dedicated Web Worker).

The dashboard loads this module in a worker and talks to it through four
functions: ``configure``, ``set_auth``, ``server_info``/``list_tools`` and
``call_tool``. Nothing here changes the stdio/PAT server in ``server.py``;
the browser gets its own FastMCP instance built from the same tool objects.

* Auth is the dashboard user's Firebase ID token (``Authorization: Bearer``)
  plus ``X-Organization-ID``. ``set_auth`` runs before every call because the
  token refreshes hourly; the HTTP client reads the current token per request.
* HTTP is a synchronous ``XMLHttpRequest`` (``XhrTransport``): in a dedicated
  worker sync XHR is allowed, and every tool is a plain ``def``, so a blocking
  request is fine. It also means a call cannot be cancelled mid-flight.
* Tools run through a real MCP ``ClientSession`` over in-memory streams (the
  same JSON-RPC path stdio clients use). If that cannot start in the host
  event loop, ``PROTOCOL`` becomes ``"direct"`` and FastMCP is called directly.
* Tools that cannot work in a worker are not offered (``EXCLUDED_TOOLS``).
* Extra tool packages (``admrl_mcp.extensions``) are added with ``load_extension``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
from importlib import metadata
from typing import Any, Callable

import httpx

from .config import AUTH_BEARER, DEFAULT_API_BASE, Settings
from .toolset import PAT_PARAGRAPH_START, TEXT_REWRITES, clone_tools, instructions_with_auth_paragraph, rewrite_description

# Tools that cannot run in a browser worker, with the reason.
EXCLUDED_TOOLS: dict[str, str] = {
    "watch_device_state": "SSE stream: a synchronous XHR cannot deliver events before it ends.",
    "watch_rollout": "SSE stream: a synchronous XHR cannot deliver events before it ends.",
    "create_registry_credential": "Reads the secret from a local file or environment variable of the MCP process; the browser has neither.",
    "update_registry_credential": "Reads the secret from a local file or environment variable of the MCP process; the browser has neither.",
    "upload_secret_file": "Reads the file from a local path of the MCP process; the browser has no such filesystem.",
}

_PAT_PARAGRAPH_START = PAT_PARAGRAPH_START
_BROWSER_AUTH_PARAGRAPH = (
    "You are running inside the Admiral dashboard, signed in as the current user. Requests use that user's "
    "session and the dashboard's current organisation, so you only see what that user can see. "
    "Live log tailing and event streaming are not available; "
    "use the historical log and state tools."
)
_TEXT_REWRITES = TEXT_REWRITES

# Forbidden request header names for XHR (plus prefixes below).
_FORBIDDEN_HEADERS = frozenset(
    {
        "accept-charset",
        "accept-encoding",
        "connection",
        "content-length",
        "cookie",
        "cookie2",
        "date",
        "dnt",
        "expect",
        "host",
        "keep-alive",
        "origin",
        "referer",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "via",
        "user-agent",
    }
)
_FORBIDDEN_PREFIXES = ("proxy-", "sec-")
# The browser already decoded the body; httpx must not decode it again.
_DROP_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


# --------------------------------------------------------------- transport ---


def _to_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if hasattr(value, "to_py"):  # Pyodide JsProxy (ArrayBuffer)
        value = value.to_py()
    return bytes(value)


def _js_body(body: bytes) -> Any:
    try:
        from pyodide.ffi import to_js  # type: ignore[import-not-found]
    except ImportError:  # not in Pyodide (tests): the fake XHR takes bytes
        return body
    return to_js(body)


def _default_xhr() -> Any:
    from js import XMLHttpRequest  # type: ignore[import-not-found]

    return XMLHttpRequest.new()


def parse_response_headers(raw: str) -> list[tuple[str, str]]:
    """Parse ``XMLHttpRequest.getAllResponseHeaders()`` (CRLF separated)."""
    out: list[tuple[str, str]] = []
    for line in (raw or "").replace("\r\n", "\n").split("\n"):
        if not line.strip() or ":" not in line:
            continue
        name, value = line.split(":", 1)
        out.append((name.strip(), value.strip()))
    return out


class XhrTransport(httpx.BaseTransport):
    """httpx transport over synchronous XMLHttpRequest (dedicated Web Worker only)."""

    def __init__(self, xhr_factory: Callable[[], Any] | None = None):
        self._factory = xhr_factory or _default_xhr

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        xhr = self._factory()
        try:
            xhr.open(request.method, str(request.url), False)
            xhr.responseType = "arraybuffer"
            read_timeout = (request.extensions.get("timeout") or {}).get("read")
            if read_timeout:
                try:
                    xhr.timeout = int(read_timeout * 1000)
                except Exception:  # noqa: BLE001 - optional on some hosts
                    pass
            for name, value in request.headers.multi_items():
                lowered = name.lower()
                if lowered in _FORBIDDEN_HEADERS or lowered.startswith(_FORBIDDEN_PREFIXES):
                    continue
                xhr.setRequestHeader(name, value)
            xhr.send(_js_body(body) if body else None)
        except Exception as exc:  # noqa: BLE001 - JS errors arrive as JsException
            text = str(exc)
            if "timeout" in text.lower() or "TimeoutError" in text:
                raise httpx.ReadTimeout(text, request=request) from exc
            raise httpx.ConnectError(text, request=request) from exc

        status = int(xhr.status or 0)
        if status == 0:
            # CORS rejection, DNS failure, offline: the browser hides the reason.
            raise httpx.ConnectError("network error (blocked by CORS, offline or unreachable)", request=request)
        headers = [
            (k, v)
            for k, v in parse_response_headers(str(xhr.getAllResponseHeaders()))
            if k.lower() not in _DROP_RESPONSE_HEADERS
        ]
        return httpx.Response(status, headers=headers, content=_to_bytes(xhr.response), request=request)


# ----------------------------------------------------------------- state ---


@dataclasses.dataclass
class _State:
    api_base: str = DEFAULT_API_BASE
    token: str = ""
    organization_id: str | None = None
    transport: httpx.BaseTransport | None = None  # tests inject one; default XhrTransport
    client_base: str | None = None


_state = _State()
_extension_tools: dict[str, list[str]] = {}
_extensions: list[str] = []  # extension modules loaded into the browser server (re-applied on rebuild)
PROTOCOL = "session"  # "session" | "direct"; becomes "direct" if the session cannot start
_browser_server: Any = None
_session: Any = None
_init_result: Any = None
_runner_task: asyncio.Task[Any] | None = None
_stop_event: asyncio.Event | None = None


def _docs_transport() -> httpx.BaseTransport:
    # Same transport selection as the Admiral client, but a *separate* client (no auth).
    return _state.transport or XhrTransport()


def configure(api_base: str, docs_base: str | None = None) -> None:
    """Set the Admiral API base (e.g. ``https://api.admrl.co/v1``) and the public docs base."""
    from . import docs

    _state.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
    docs.configure(docs_base or docs.DEFAULT_DOCS_BASE, _docs_transport)


def set_auth(token: str, organization_id: str | None) -> None:
    """Install the current session token + org. Call before every tool call."""
    from . import server  # local import: keeps ``import admrl_mcp.browser`` cheap

    _state.token = token or ""
    _state.organization_id = organization_id or None
    client = server._client
    if client is None or _state.client_base != _state.api_base:
        if client is not None:
            client.close()
        from .client import AdmiralClient

        settings = Settings(
            api_base=_state.api_base,
            token_id="",
            secret_key="",
            org_id=_state.organization_id,
            auth_mode=AUTH_BEARER,
            bearer_token=_state.token,
        )
        server._client = AdmiralClient(settings, transport=_state.transport or XhrTransport())
        _state.client_base = _state.api_base
    else:
        client.settings = dataclasses.replace(
            client.settings, org_id=_state.organization_id, bearer_token=_state.token
        )


# ---------------------------------------------------------- server build ---


def browser_instructions() -> str:
    return instructions_with_auth_paragraph(_BROWSER_AUTH_PARAGRAPH)


def _rewrite_description(text: str | None) -> str:
    return rewrite_description(text)


def build_browser_server() -> Any:
    """A FastMCP instance with the browser-safe subset of the admrl tools."""
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP(name="admrl", instructions=browser_instructions())
    clone_tools(srv, EXCLUDED_TOOLS)
    for module_name in _extensions:
        _apply_extension(srv, module_name)
    return srv


def _apply_extension(srv: Any, module_name: str) -> list[str]:
    """Register an extension on the browser server, minus its browser-unsafe tools."""
    from .extensions import module_browser_excluded, register_extension

    added = register_extension(module_name, srv)
    excluded = module_browser_excluded(module_name)
    kept = []
    for name in added:
        if name in excluded:
            srv.remove_tool(name)
            continue
        tool = srv._tool_manager.get_tool(name)
        srv._tool_manager._tools[name] = tool.model_copy(update={"description": _rewrite_description(tool.description)})
        kept.append(name)
    return kept


def load_extension(module_name: str) -> str:
    """Add an extension module's tools to the browser server; returns JSON ``{"tools": [...]}``.

    The module (see ``admrl_mcp.extensions``) must already be importable, e.g. its wheel installed
    into the Pyodide filesystem with micropip. Tools it lists in ``BROWSER_EXCLUDED_TOOLS`` are
    dropped. Synchronous: the open in-memory MCP session reads the live tool table, so its next
    ``list_tools`` includes the new tools; call this before the host freezes its tool list.
    Loading the same module twice is a no-op that returns the same tool names.
    """
    name = module_name.strip()
    srv = _server()
    if name in _extensions:
        from .extensions import _tool_names  # already registered: report what it contributed

        return json.dumps({"tools": [n for n in _tool_names(srv) if n in _extension_tools.get(name, [])]})
    tools = _apply_extension(srv, name)
    _extensions.append(name)
    _extension_tools[name] = tools
    return json.dumps({"tools": tools})


def _server() -> Any:
    global _browser_server
    if _browser_server is None:
        # INFO logs would put every request line (and URLs) in the browser console.
        for noisy in ("httpx", "httpcore", "mcp"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _browser_server = build_browser_server()
    return _browser_server


# ------------------------------------------------------- session / direct ---


async def _run_session(ready: asyncio.Event, stop: asyncio.Event, box: dict[str, Any]) -> None:
    import anyio
    from mcp.client.session import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams

    srv = _server()._mcp_server
    try:
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    lambda: srv.run(
                        server_streams[0],
                        server_streams[1],
                        srv.create_initialization_options(),
                        raise_exceptions=False,
                    )
                )
                try:
                    async with ClientSession(*client_streams) as session:
                        box["init"] = await session.initialize()
                        box["session"] = session
                        ready.set()
                        await stop.wait()
                finally:
                    tg.cancel_scope.cancel()
    except BaseException as exc:  # noqa: BLE001 - report to the starter, never leave it hanging
        box["error"] = exc
        ready.set()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise


async def _ensure_session() -> Any | None:
    """Return the live ClientSession, starting it on first use. None => direct mode."""
    global PROTOCOL, _session, _init_result, _runner_task, _stop_event
    if PROTOCOL == "direct":
        return None
    if _session is not None and _runner_task is not None and not _runner_task.done():
        return _session
    ready = asyncio.Event()
    stop = asyncio.Event()
    box: dict[str, Any] = {}
    task = asyncio.ensure_future(_run_session(ready, stop, box))
    try:
        await asyncio.wait_for(ready.wait(), timeout=30)
    except Exception as exc:  # noqa: BLE001
        box.setdefault("error", exc)
    if box.get("error") is not None or "session" not in box:
        stop.set()
        task.cancel()
        PROTOCOL = "direct"
        print(f"[admrl-mcp] in-memory MCP session unavailable, using direct calls: {box.get('error')!r}", file=sys.stderr)
        return None
    _session, _init_result, _runner_task, _stop_event = box["session"], box["init"], task, stop
    return _session


async def aclose() -> None:
    """Stop the session task (tests, worker shutdown)."""
    global _session, _runner_task, _stop_event, _init_result
    if _stop_event is not None:
        _stop_event.set()
    if _runner_task is not None:
        try:
            await asyncio.wait_for(_runner_task, timeout=5)
        except Exception:  # noqa: BLE001
            _runner_task.cancel()
    _session = _runner_task = _stop_event = _init_result = None


def reset() -> None:
    """Forget cached server/session/state. For tests."""
    global _browser_server, PROTOCOL
    from . import docs

    docs.configure(None, None)
    _browser_server = None
    _extensions.clear()
    _extension_tools.clear()
    PROTOCOL = "session"
    _state.__dict__.update(_State().__dict__)


# --------------------------------------------------------------- public ---


def _version(dist: str, default: str = "unknown") -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return default


def _pyodide_version() -> str:
    try:
        import pyodide  # type: ignore[import-not-found]

        return str(pyodide.__version__)
    except Exception:  # noqa: BLE001
        return "none"


def _tool_info(tool: Any) -> dict[str, Any]:
    dumped = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    info: dict[str, Any] = {
        "name": dumped["name"],
        "description": dumped.get("description") or "",
        "inputSchema": dumped.get("inputSchema") or {"type": "object", "properties": {}},
    }
    if dumped.get("annotations"):
        info["annotations"] = dumped["annotations"]
    return info


async def server_info() -> str:
    """JSON ``McpServerInfo``."""
    session = await _ensure_session()
    srv = _server()
    name, instructions = "admrl", srv.instructions or ""  # live: includes extension instructions
    if session is not None and _init_result is not None:
        name = _init_result.serverInfo.name
    return json.dumps(
        {
            "name": name,
            "version": _version("admrl-mcp", "0.0.0"),
            "instructions": instructions,
            "runtime": {
                "pyodide": _pyodide_version(),
                "python": sys.version.split()[0],
                "mcpSdk": _version("mcp"),
                "protocol": PROTOCOL,
            },
        }
    )


async def list_tools() -> str:
    """JSON list of ``McpToolInfo``."""
    session = await _ensure_session()
    if session is not None:
        tools: list[Any] = []
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
    else:
        tools = list(await _server().list_tools())
    return json.dumps([_tool_info(t) for t in tools])


def _content_item(item: Any) -> dict[str, Any] | None:
    kind = getattr(item, "type", None)
    if kind == "text":
        return {"type": "text", "text": item.text}
    if kind == "image":
        return {"type": "image", "data": item.data, "mimeType": item.mimeType}
    return None


async def call_tool(name: str, arguments_json: str) -> str:
    """Run one tool; JSON ``McpToolResult`` (``content`` text/image items, ``isError``)."""
    arguments = json.loads(arguments_json) if arguments_json else {}
    if not isinstance(arguments, dict):
        raise ValueError("arguments_json must encode an object")
    session = await _ensure_session()
    if session is not None:
        result = await session.call_tool(name, arguments)
        content = [c for c in (_content_item(i) for i in result.content) if c]
        return json.dumps({"content": content, "isError": bool(result.isError)})
    try:
        raw = await _server().call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001 - mirror the protocol: errors become isError results
        return json.dumps({"content": [{"type": "text", "text": f"Error executing tool {name}: {exc}"}], "isError": True})
    items = raw[0] if isinstance(raw, tuple) else raw  # (content, structured) when an output schema exists
    content = [c for c in (_content_item(i) for i in items) if c]
    return json.dumps({"content": content, "isError": False})
