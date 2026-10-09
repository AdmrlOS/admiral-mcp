"""Helpers shared by the browser and hosted (streamable HTTP) builds.

Both derive their tool set from the stdio server's ``mcp`` instance: the same ``Tool`` objects
(schemas, annotations) minus the tools they cannot run, with PAT-oriented wording rewritten.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

PAT_PARAGRAPH_START = "Auth is a Personal API Token"

# (old, new) exact-text rewrites applied to the PAT-oriented server text. A
# test fails if the server text drifts so that one of these stops matching.
TEXT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "  Do not try to open the live websocket tail; PAT cannot authenticate it.\n",
        "  The live websocket tail is not available here.\n",
    ),
    (
        "- Observed state: get_device_state (stored; live=true asks the device), watch_device_state (SSE timeline of\n"
        "  condition transitions and progress operations). Desired state:",
        "- Observed state: get_device_state (stored; live=true asks the device). Desired state:",
    ),
    (", watch_rollout (SSE until terminal).", "."),
    (
        "List organisations visible to this PAT. Use when ADMRL_ORG_ID is unset or the user asks which org to use.",
        "List organisations visible to the signed-in user. Use when the user asks which org to use.",
    ),
    ("PAT cannot open the live websocket tail. ", "The live websocket tail is not available here. "),
    (
        "- Local secrets (stdio only): create_registry_credential / update_registry_credential read the secret from a local secret_file\n"
        "  or a secret_env variable and upload_secret_file reads a local source_path; secrets and contents are never tool arguments.\n",
        "",
    ),
)


def rewrite_description(text: str | None) -> str:
    out = text or ""
    for old, new in TEXT_REWRITES:
        out = out.replace(old, new)
    return out


def instructions_with_auth_paragraph(auth_paragraph: str) -> str:
    """The stdio INSTRUCTIONS with the PAT auth paragraph replaced and PAT wording rewritten."""
    from .server import INSTRUCTIONS

    text = INSTRUCTIONS
    text = text[: text.index(PAT_PARAGRAPH_START)] + auth_paragraph
    for old, new in TEXT_REWRITES:
        text = text.replace(old, new)
    return text


def clone_tools(target: Any, excluded: Collection[str] | Mapping[str, str]) -> None:
    """Replace ``target``'s tools with copies of the stdio server's tools minus ``excluded``."""
    from . import server

    tools = {}
    for tool in server.mcp._tool_manager.list_tools():
        if tool.name in excluded:
            continue
        tools[tool.name] = tool.model_copy(update={"description": rewrite_description(tool.description)})
    # Same Tool objects (schemas, output handling, annotations); private dict
    # because FastMCP has no public "register an existing Tool" API.
    target._tool_manager._tools = tools
