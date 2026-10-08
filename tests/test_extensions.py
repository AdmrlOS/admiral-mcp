"""Generic extension hook: register_extension, ADMRL_MCP_EXTENSIONS, browser.load_extension."""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from admrl_mcp import browser, server
from admrl_mcp.extensions import ExtensionError, load_env_extensions, register_extension


def _module(name: str, *, annotated: bool = True, excluded: dict | None = None, instructions: str | None = None):
    mod = types.ModuleType(name)

    def ext_read() -> str:
        return "r"

    def ext_write() -> str:
        return "w"

    def register(mcp):
        ann = (lambda ro: ToolAnnotations(readOnlyHint=ro)) if annotated else (lambda ro: None)
        mcp.tool(description="read thing, via PAT", annotations=ann(True))(ext_read)
        mcp.tool(description="write thing", annotations=ann(False))(ext_write)
        return ["ext_read", "ext_write"]

    mod.register = register
    if excluded is not None:
        mod.BROWSER_EXCLUDED_TOOLS = excluded
    if instructions is not None:
        mod.INSTRUCTIONS = instructions
    sys.modules[name] = mod
    return mod


@pytest.fixture(autouse=True)
def _isolate():
    saved = server._client
    browser.reset()
    yield
    server._client = saved
    browser.reset()
    for n in [n for n in sys.modules if n.startswith("fake_ext_")]:
        del sys.modules[n]


def _names(srv: FastMCP) -> list[str]:
    return [t.name for t in asyncio.run(srv.list_tools())]


def test_register_extension_adds_tools_and_instructions():
    _module("fake_ext_ok", instructions="- ext tools exist.")
    srv = FastMCP(name="t", instructions="base")
    assert register_extension("fake_ext_ok", srv) == ["ext_read", "ext_write"]
    assert _names(srv) == ["ext_read", "ext_write"]
    assert srv._mcp_server.instructions == "base\n- ext tools exist."


def test_missing_annotation_rejects_and_rolls_back():
    _module("fake_ext_bad", annotated=False)
    srv = FastMCP(name="t")
    with pytest.raises(ExtensionError, match="readOnlyHint"):
        register_extension("fake_ext_bad", srv)
    assert _names(srv) == []


def test_missing_module_and_missing_register():
    srv = FastMCP(name="t")
    with pytest.raises(ExtensionError, match="not installed"):
        register_extension("fake_ext_nope", srv)
    sys.modules["fake_ext_empty"] = types.ModuleType("fake_ext_empty")
    with pytest.raises(ExtensionError, match="register"):
        register_extension("fake_ext_empty", srv)


def test_env_loader_skips_bad_modules_and_logs(capsys):
    _module("fake_ext_one")
    srv = FastMCP(name="t")
    loaded = load_env_extensions(srv, {"ADMRL_MCP_EXTENSIONS": " fake_ext_missing , fake_ext_one,"})
    assert loaded == {"fake_ext_one": ["ext_read", "ext_write"]}
    assert "fake_ext_missing" in capsys.readouterr().err
    assert load_env_extensions(srv, {}) == {}


def test_browser_load_extension_in_open_session_filters_excluded_and_rewrites():
    _module("fake_ext_br", excluded={"ext_write": "no disk"})

    async def go():
        try:
            before = {t["name"] for t in json.loads(await browser.list_tools())}  # session is open now
            out = json.loads(browser.load_extension("fake_ext_br"))
            again = json.loads(browser.load_extension("fake_ext_br"))
            after = {t["name"]: t for t in json.loads(await browser.list_tools())}
            return before, out, again, after
        finally:
            await browser.aclose()

    before, out, again, after = asyncio.run(go())
    assert out == again == {"tools": ["ext_read"]}
    assert "ext_read" not in before and set(after) - before == {"ext_read"}
    assert after["ext_read"]["annotations"]["readOnlyHint"] is True
    assert "ext_read" not in {t.name for t in server.mcp._tool_manager.list_tools()}  # stdio server untouched


def test_public_server_has_no_extension_tools_by_default():
    assert not any(t.name.startswith("ext_") for t in server.mcp._tool_manager.list_tools())
