"""Generic extension hook: load extra tool packages into a FastMCP server.

An extension is any importable module with a ``register(mcp: FastMCP) -> list[str]`` function
that adds tools to ``mcp`` and returns their names. Optional module attributes:

* ``INSTRUCTIONS: str`` - appended to the server instructions.
* ``BROWSER_EXCLUDED_TOOLS: dict[str, str]`` - ``{tool name: reason}`` for tools the browser
  runtime must not offer (see ``admrl_mcp.browser.load_extension``).

Every tool an extension adds must carry ``ToolAnnotations`` with an explicit ``readOnlyHint``
(the dashboard auto-runs only read-only tools), and every tool with ``destructiveHint=True`` must
take a boolean ``confirm`` parameter that defaults to false (without it the tool only previews;
see ``admrl_mcp.ops.confirmation``); otherwise the whole extension is rejected and its tools are
removed again.

stdio: set ``ADMRL_MCP_EXTENSIONS=mod1,mod2`` and ``server.main()`` loads them at startup.
A module that cannot be loaded is logged and skipped; the server still starts.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Mapping

from mcp.server.fastmcp import FastMCP

ENV_VAR = "ADMRL_MCP_EXTENSIONS"
log = logging.getLogger("admrl_mcp.extensions")


class ExtensionError(RuntimeError):
    """An extension is missing, malformed, or adds tools without explicit annotations."""


def _tool_names(target: FastMCP) -> list[str]:
    return [t.name for t in target._tool_manager.list_tools()]


def has_confirm_gate(parameters: Mapping[str, object] | None) -> bool:
    """True if a tool's JSON parameter schema has a boolean ``confirm`` that defaults to false."""
    props = (parameters or {}).get("properties")
    prop = props.get("confirm") if isinstance(props, Mapping) else None
    return isinstance(prop, Mapping) and prop.get("type") == "boolean" and prop.get("default") is False


def register_extension(module_name: str, target: FastMCP) -> list[str]:
    """Import ``module_name`` and call its ``register(target)``; return the tool names it added.

    Raises ``ExtensionError`` (leaving ``target`` unchanged) if the module cannot be imported,
    has no ``register``, any added tool lacks an explicit ``readOnlyHint``, or a destructive tool
    has no ``confirm`` parameter.
    """
    name = (module_name or "").strip()
    if not name:
        raise ExtensionError("empty extension module name")
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise ExtensionError(f"extension {name!r} is not installed: {exc}") from exc
    register = getattr(module, "register", None)
    if not callable(register):
        raise ExtensionError(f"extension {name!r} has no register(mcp) function")

    before = set(_tool_names(target))
    register(target)
    added = [n for n in _tool_names(target) if n not in before]
    missing = []
    ungated = []
    for tool_name in added:
        tool = target._tool_manager.get_tool(tool_name)
        ann = tool.annotations
        if ann is None or ann.readOnlyHint is None:
            missing.append(tool_name)
        elif ann.destructiveHint is True and not has_confirm_gate(tool.parameters):
            ungated.append(tool_name)
    if missing or ungated:
        for tool_name in added:
            target.remove_tool(tool_name)
    if missing:
        raise ExtensionError(f"extension {name!r} tools without an explicit readOnlyHint annotation: {missing}")
    if ungated:
        raise ExtensionError(
            f"extension {name!r} destructive tools without a `confirm` parameter (boolean, default false; "
            f"without it the tool must only preview): {ungated}"
        )

    extra = getattr(module, "INSTRUCTIONS", None)
    if added and isinstance(extra, str) and extra.strip():
        server = target._mcp_server
        server.instructions = ((server.instructions or "").rstrip() + "\n" + extra.strip()).strip()
    return added


def module_browser_excluded(module_name: str) -> dict[str, str]:
    """``BROWSER_EXCLUDED_TOOLS`` declared by an (already imported) extension module."""
    excluded = getattr(importlib.import_module(module_name.strip()), "BROWSER_EXCLUDED_TOOLS", None)
    return dict(excluded) if isinstance(excluded, Mapping) else {}


def load_env_extensions(target: FastMCP, env: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Load the modules named in ``ADMRL_MCP_EXTENSIONS``; log and skip any that fail."""
    raw = (env if env is not None else os.environ).get(ENV_VAR, "")
    loaded: dict[str, list[str]] = {}
    for name in (n.strip() for n in raw.split(",")):
        if not name:
            continue
        try:
            loaded[name] = register_extension(name, target)
        except Exception as exc:  # noqa: BLE001 - a bad extension must never stop the server
            log.error("%s: could not load extension %r: %s", ENV_VAR, name, exc)
            print(f"[admrl-mcp] {ENV_VAR}: could not load extension {name!r}: {exc}", file=sys.stderr)
    return loaded
