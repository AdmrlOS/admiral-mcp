"""Readable forms of the 409 refusals the API returns when a configuration cannot be delivered or saved.

The API answers these with ``{code, msg, data: {error: <code>, ...}}``. Tools that assign, roll out, move,
override or save a configuration all surface them through ``server._err``, which merges the dict from
``describe`` into its answer so the caller sees which fleet or device refused, why, and what to do.
"""

from __future__ import annotations

from typing import Any

HOST_ACCESS = "host_access_not_allowed"
UNSUPPORTED_BY_AGENT = "unsupported_by_agent"
BASE_VERSION_CONFLICT = "base_version_conflict"
BASE_VERSION_REQUIRED = "base_version_required"
EXPORT_UNSUPPORTED = "compose_export_unsupported"

CODES = (HOST_ACCESS, UNSUPPORTED_BY_AGENT, BASE_VERSION_CONFLICT, BASE_VERSION_REQUIRED, EXPORT_UNSUPPORTED)

_MAX_LISTED = 10


def error_code(body: Any) -> str | None:
    """The API error code of a response body: ``data.error`` first, then top-level ``error`` / ``code``."""
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    for source in (data if isinstance(data, dict) else None, body):
        if not source:
            continue
        for key in ("error", "code"):
            value = source.get(key)
            if isinstance(value, str) and value in CODES:
                return value
    return None


def _data(body: Any) -> dict[str, Any]:
    data = body.get("data") if isinstance(body, dict) else None
    return data if isinstance(data, dict) else {}


def _use_text(use: Any) -> str:
    if not isinstance(use, dict):
        return str(use)
    where = f"service {use['service']}" if use.get("service") else "set level"
    what = " ".join(str(p) for p in (use.get("kind"), use.get("detail")) if p)
    return f"{where}: {what}".strip()


def _host_access(data: dict[str, Any]) -> dict[str, Any]:
    uses = [_use_text(u) for u in data.get("uses") or []]
    fleet = {"id": data.get("fleetId"), "name": data.get("fleetName")}
    return {
        "refusal": HOST_ACCESS,
        "fleet": {k: v for k, v in fleet.items() if v},
        "uses": uses,
        "hint": (
            "This configuration asks for host access (privileged mode, dangerous capabilities, devices or host namespaces) "
            "that the fleet's security policy does not allow. Nothing was changed. Either remove those settings from the "
            "configuration, or ask someone who manages the fleet's security policy to allow them under workload_host_access "
            "(allow_privileged, allow_capabilities, allow_devices, allow_host_namespaces) in the dashboard, then retry. "
            "Each use is checked against its own class, so allowing privileged does not allow devices."
        ),
    }


def _unsupported(data: dict[str, Any]) -> dict[str, Any]:
    devices = [d for d in data.get("devices") or [] if isinstance(d, dict)]
    rows = []
    for d in devices[:_MAX_LISTED]:
        row = {"id": d.get("id"), "name": d.get("name")}
        for src, dst in (("agent_version", "agent_version"), ("protocol", "protocol"), ("missingFeatures", "missing_features")):
            if d.get(src) not in (None, "", []):
                row[dst] = d[src]
        rows.append(row)
    reported = data.get("count")
    count: int = reported if isinstance(reported, int) else len(devices)
    out: dict[str, Any] = {
        "refusal": UNSUPPORTED_BY_AGENT,
        "devices_refusing": count,
        "devices": rows,
        "hint": (
            "Some devices run an agent that cannot run this service set, or lacks agent features the set needs "
            "(for example healthchecks or privileged mode). Nothing was changed. Update those devices' agents, move them to "
            "another fleet (move_device_to_fleet), or remove the settings that need the missing features, then retry. "
            "list_fleet_protocols shows each device's agent protocol."
        ),
    }
    if count > len(rows):
        out["devices_omitted"] = count - len(rows)
    for src, dst in (("missingFeatures", "missing_features"), ("blockers", "blockers")):
        if data.get(src):
            out[dst] = data[src]
    if data.get("required_protocol") is not None:
        out["required_protocol"] = data["required_protocol"]
    return out


def _base_conflict(data: dict[str, Any]) -> dict[str, Any]:
    current = data.get("currentVersion")
    out: dict[str, Any] = {"refusal": BASE_VERSION_CONFLICT, "base_version": data.get("baseVersion"), "current_version": current}
    for src, dst in (("updatedBy", "updated_by"), ("updatedAt", "updated_at")):
        if data.get(src):
            out[dst] = data[src]
    out["hint"] = (
        f"The configuration is at version {current}, not the version this change was based on: someone saved a newer one. "
        "Nothing was saved. To retry: export the latest version (export_configuration_compose), re-apply your edits to it, "
        f"then call import_configuration_compose again with base_version={current} and dry_run=true first. "
        "Do not just raise base_version on the old file: that would overwrite the newer version's changes."
    )
    return out


def _base_required() -> dict[str, Any]:
    return {
        "refusal": BASE_VERSION_REQUIRED,
        "hint": (
            "Say which version the file was based on: pass base_version (the version you exported or last read), or use a file "
            "exported from this configuration, which records it under x-admrl.configuration."
        ),
    }


def _export_unsupported(data: dict[str, Any]) -> dict[str, Any]:
    issues = [f"{i.get('path')}: {i.get('message')}" if i.get("path") else str(i.get("message")) for i in data.get("issues") or [] if isinstance(i, dict)]
    return {
        "refusal": EXPORT_UNSUPPORTED,
        "issues": issues,
        "hint": "This version uses settings docker-compose cannot express, so it cannot be exported. Use get_configuration to read its spec instead.",
    }


def describe(body: Any) -> dict[str, Any] | None:
    """Structured, readable detail for a known refusal body, else ``None``."""
    code = error_code(body)
    if code is None:
        return None
    data = _data(body)
    if code == HOST_ACCESS:
        return _host_access(data)
    if code == UNSUPPORTED_BY_AGENT:
        return _unsupported(data)
    if code == BASE_VERSION_CONFLICT:
        return _base_conflict(data)
    if code == BASE_VERSION_REQUIRED:
        return _base_required()
    return _export_unsupported(data)


PROXY_ERRORS: dict[str, tuple[str, str]] = {
    "public_not_allowed": (
        "Public proxies are not allowed for this device's fleet.",
        "Turn on public proxies for the fleet first (set_fleet_proxy_policy with public_allowed=true; needs the remote access management permission), or open a private proxy.",
    ),
    "exceeds_fleet_max": (
        "The requested lifetime is not allowed by the fleet's public proxy limit.",
        "A fleet with a public proxy max life requires expires_in between 1 and that many seconds (0/never is refused). Read it with get_fleet_proxy_policy.",
    ),
    "prefix_required": (
        "Custom proxy names need an organisation proxy prefix, and this organisation has none.",
        "Set one with set_org_proxy_prefix (organisation admin), or omit name to get a random one.",
    ),
    "name_taken": (
        "A proxy with that name is already open.",
        "Pick another name, or close the existing proxy with close_proxy if it is yours (list_proxies).",
    ),
    "prefix_taken": (
        "Another organisation already uses that proxy prefix.",
        "Choose a different prefix.",
    ),
    "prefix_reserved": (
        "That proxy prefix is reserved (platform, brand or common-word names cannot be used).",
        "Choose a different prefix of 3-20 lowercase letters, digits and hyphens.",
    ),
}


def proxy_error(body: Any) -> dict[str, Any] | None:
    """Clear message for an EdgeWire proxy refusal (``error_code`` at top level, in ``data``, or ``data.error``)."""
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    for source in (body, data if isinstance(data, dict) else None):
        if not source:
            continue
        for key in ("error_code", "error", "code"):
            value = source.get(key)
            if isinstance(value, str) and value in PROXY_ERRORS:
                message, hint = PROXY_ERRORS[value]
                return {"error": message, "error_code": value, "hint": hint}
    return None
