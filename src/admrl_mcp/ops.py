"""Pure helpers for the operations tools: configuration spec edits and diffs.

Nothing here talks to the network. Specs are the JSON objects the configuration API returns
(``image``, ``desiredState``, ``command``, ``environment``, ``mounts``, ``options``, ``volumes``,
``ports``, ``registry_credential_id``, ``signaturePolicy``).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

# Environment variables whose names look like credentials are masked in tool output (the real
# values still travel to the API when an edit is applied).
SECRET_KEY_RE = re.compile(r"(pass(word|wd)?|secret|token|api[_-]?key|private[_-]?key|credential|auth)", re.IGNORECASE)
MASK = "***"

_COLLECTIONS_LIST = ("command", "mounts", "ports")
_COLLECTIONS_DICT = ("environment", "volumes", "options")


def norm_spec(spec: Any) -> dict[str, Any]:
    """Canonical copy of a spec, mirroring the backend's own normalisation.

    ``null`` collections become ``[]`` / ``{}``, ``desiredState`` is upper-cased (default RUNNING),
    port protocols are lower-cased (default tcp) and the image is trimmed. Unknown keys are kept.
    """
    out = copy.deepcopy(spec) if isinstance(spec, dict) else {}
    for key in _COLLECTIONS_LIST:
        if out.get(key) is None:
            out[key] = []
    for key in _COLLECTIONS_DICT:
        if out.get(key) is None:
            out[key] = {}
    if isinstance(out.get("image"), str):
        out["image"] = out["image"].strip()
    state = str(out.get("desiredState") or "").strip().upper()
    out["desiredState"] = state or "RUNNING"
    for port in out["ports"]:
        if isinstance(port, dict):
            port["protocol"] = str(port.get("protocol") or "tcp").strip().lower()
    if out.get("environment"):
        out["environment"] = {str(k): v for k, v in out["environment"].items()}
    return out


def is_secret_key(name: str) -> bool:
    return bool(SECRET_KEY_RE.search(str(name)))


def mask_spec(spec: Any) -> Any:
    """Copy of ``spec`` with credential-looking environment values masked."""
    out = copy.deepcopy(spec)
    env = out.get("environment") if isinstance(out, dict) else None
    if isinstance(env, dict):
        for key in env:
            if is_secret_key(key):
                env[key] = MASK
    return out


# ----------------------------------------------------------------- edits ---


def retag_image(image: str, tag: str) -> str:
    """Replace the tag of an image reference, dropping any digest. ``host:5000/a/b:1`` -> ``host:5000/a/b:<tag>``."""
    ref = (image or "").strip()
    if not ref:
        raise ValueError("The configuration has no image to re-tag; pass image=<full reference>.")
    tag = tag.strip()
    if not tag or any(c in tag for c in " /@"):
        raise ValueError(f"Invalid image tag {tag!r}.")
    ref = ref.split("@", 1)[0]
    last = ref.rsplit("/", 1)[-1]
    if ":" in last:
        ref = ref[: len(ref) - len(last)] + last.split(":", 1)[0]
    return f"{ref}:{tag}"


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7386 JSON merge patch: objects merge recursively, ``null`` deletes, everything else replaces."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def apply_spec_edits(
    spec: dict[str, Any],
    *,
    image: str | None = None,
    image_tag: str | None = None,
    env_set: dict[str, Any] | None = None,
    env_unset: list[str] | None = None,
    patch: dict[str, Any] | None = None,
    replace: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(new_spec, notes)`` for the requested edits applied to ``spec``.

    ``replace`` (a whole spec) is exclusive with every other edit. Order otherwise: image/tag, env set,
    env unset, then the merge patch (so the patch has the last word).
    """
    notes: list[str] = []
    if replace is not None:
        if any(v for v in (image, image_tag, env_set, env_unset, patch)):
            raise ValueError("spec (full replacement) cannot be combined with image, image_tag, env_set, env_unset or patch.")
        if not isinstance(replace, dict):
            raise ValueError("spec must be a JSON object.")
        return norm_spec(replace), notes
    if image and image_tag:
        raise ValueError("Pass image (full reference) or image_tag (tag only), not both.")
    new = norm_spec(spec)
    if image:
        new["image"] = image.strip()
    if image_tag:
        new["image"] = retag_image(new.get("image", ""), image_tag)
    if env_set:
        if not isinstance(env_set, dict):
            raise ValueError("env_set must be an object of NAME: value.")
        for key, value in env_set.items():
            if not str(key).strip():
                raise ValueError("Environment variable names cannot be empty.")
            new["environment"][str(key)] = "" if value is None else str(value)
    for key in env_unset or []:
        if key in new["environment"]:
            del new["environment"][key]
        else:
            notes.append(f"env_unset: {key} was not set (ignored)")
    if patch:
        if not isinstance(patch, dict):
            raise ValueError("patch must be a JSON object (RFC 7386 merge patch).")
        new = norm_spec(merge_patch(new, patch))
    return new, notes


# ------------------------------------------------------------------ diff ---


def _flatten(spec: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
            return
        if value in (None, "", [], {}):
            return
        flat[prefix] = value

    for key, value in spec.items():
        if key == "ports":
            for port in value or []:
                if isinstance(port, dict):
                    flat[f"ports[{port.get('protocol', 'tcp')}/{port.get('port')}]"] = port
        elif key == "mounts":
            for mount in value or []:
                if isinstance(mount, dict):
                    flat[f"mounts[{mount.get('destination')}]"] = mount
        else:
            walk(str(key), value)
    return flat


def spec_diff(old: Any, new: Any, *, mask: bool = True) -> list[dict[str, Any]]:
    """Structured differences between two specs, one row per changed leaf.

    Rows are ``{path, change: added|removed|changed, old?, new?}``, sorted by path. ``null`` / empty
    collections compare equal. Credential-looking environment values are masked unless ``mask=False``.
    """
    before, after = _flatten(norm_spec(old)), _flatten(norm_spec(new))
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path), after.get(path)
        if path in before and path in after:
            if a == b:
                continue
            row: dict[str, Any] = {"path": path, "change": "changed", "old": a, "new": b}
        elif path in after:
            row = {"path": path, "change": "added", "new": b}
        else:
            row = {"path": path, "change": "removed", "old": a}
        if mask and path.startswith("environment.") and is_secret_key(path.split(".", 1)[1]):
            for k in ("old", "new"):
                if k in row:
                    row[k] = MASK
        rows.append(row)
    return rows


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def diff_summary(rows: list[dict[str, Any]], limit: int = 6) -> str:
    """One line: ``image: a:1 -> a:2; environment.X added; ...``."""
    if not rows:
        return "no changes"
    parts = []
    for row in rows[:limit]:
        if row["change"] == "changed" and row["old"] == MASK and row["new"] == MASK:
            parts.append(f"{row['path']} changed (value masked)")
        elif row["change"] == "changed":
            parts.append(f"{row['path']}: {_short(row['old'])} → {_short(row['new'])}")
        else:
            parts.append(f"{row['path']} {row['change']}")
    if len(rows) > limit:
        parts.append(f"+{len(rows) - limit} more")
    return "; ".join(parts)


def spec_summary(spec: Any) -> dict[str, Any]:
    """Short human view of a spec: image, state, counts. Safe to put in lists."""
    s = norm_spec(spec)
    return {
        "image": s.get("image"),
        "desiredState": s.get("desiredState"),
        "env_vars": len(s["environment"]),
        "ports": len(s["ports"]),
        "volumes": len(s["volumes"]),
        "mounts": len(s["mounts"]),
        "signature_policy": bool(s.get("signaturePolicy")),
    }


# ---------------------------------------------------------- confirmation ---

CONFIRM_INSTRUCTION = "Show this to the user and call again with confirm=true only after they explicitly agree."


def confirmation(
    tool: str,
    effect: str,
    *,
    action: dict[str, Any],
    preview: dict[str, Any] | None = None,
    arguments: dict[str, Any],
    irreversible: bool = False,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """The uniform answer of a destructive tool called without ``confirm=true``.

    Nothing has been changed when this is returned. ``arguments`` are the fully resolved arguments
    (UUIDs, not names) so the confirmed call cannot hit a different resource; ``confirm`` is added here.
    """
    out: dict[str, Any] = {
        "summary": f"Not done yet: {effect}",
        "confirmation_required": True,
        "action": {"tool": tool, **action},
        "preview": preview or {},
        "irreversible": bool(irreversible),
    }
    if warnings:
        out["warnings"] = warnings
    out["next"] = {"tool": tool, "arguments": {**{k: v for k, v in arguments.items() if v is not None}, "confirm": True}}
    out["instruction"] = CONFIRM_INSTRUCTION
    return out


def is_confirmation(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("confirmation_required") is True


def json_diff(old: Any, new: Any) -> list[dict[str, Any]]:
    """Leaf-level differences between two JSON documents, rows ``{path, change, old?, new?}`` sorted by path."""

    def flat(node: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict) and value:
                for key, item in value.items():
                    walk(f"{prefix}.{key}" if prefix else str(key), item)
            else:
                out[prefix] = value

        walk("", node)
        return out

    before, after = flat(old), flat(new)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        if path in before and path in after:
            if before[path] != after[path]:
                rows.append({"path": path, "change": "changed", "old": before[path], "new": after[path]})
        elif path in after:
            rows.append({"path": path, "change": "added", "new": after[path]})
        else:
            rows.append({"path": path, "change": "removed", "old": before[path]})
    return rows


# ------------------------------------------------------- network config ---

PSK_MASK = "***"
_NETWORK_READONLY = {
    "id", "scope", "fleet_id", "host_id", "organisation_id", "version", "created_by", "updated_by", "created_at", "updated_at",
}
_IFACE_TYPES = ("wifi", "ethernet", "bridge")
_WIFI_MODES = ("client", "ap")


def _only(obj: Any, allowed: set[str], where: str, readonly: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object.")
    unknown = sorted(set(obj) - allowed - readonly)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}; allowed: {sorted(allowed)}.")
    return {k: copy.deepcopy(v) for k, v in obj.items() if k in allowed}


def _need(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    for key in keys:
        if obj.get(key) in (None, "", []):
            raise ValueError(f"{where}: {key} is required.")


def _ip_settings(obj: Any, where: str, fleet: bool) -> dict[str, Any]:
    if fleet:
        out = _only(obj, {"proxy"}, where)
    else:
        out = _only(obj, {"dhcp_v4", "ipv4", "ipv6", "proxy"}, where)
        for fam in ("ipv4", "ipv6"):
            if out.get(fam) is not None:
                out[fam] = _only(out[fam], {"address", "gateway", "dns"}, f"{where}.{fam}")
                _need(out[fam], ("address",), f"{where}.{fam}")
    if out.get("proxy") is not None:
        out["proxy"] = _only(out["proxy"], {"server", "ignore_tls"}, f"{where}.proxy")
    return out


def clean_network_config(cfg: Any, *, fleet: bool) -> dict[str, Any]:
    """Validate a network configuration request body against the backend's request model.

    Unknown keys are rejected (the backend decodes non-strictly and would drop them silently). Read-only
    fields a GET returns (``id``, ``version``, ``has_psk``...) are stripped, so a stored configuration can
    be edited and sent back. Fleet configurations only carry a proxy under ``ip`` (no static addressing).
    """
    where = "network configuration"
    top = _only(cfg, {"interfaces", "client_networks", "bridges", "nameservers"}, where, _NETWORK_READONLY)
    if not isinstance(top.get("interfaces"), list):
        raise ValueError("interfaces is required (a list; [] leaves the interfaces to the device's defaults).")
    out: dict[str, Any] = {"interfaces": []}
    for i, raw in enumerate(top["interfaces"]):
        w = f"interfaces[{i}]"
        item = _only(raw, {"match_mac", "enabled", "type", "wifi", "ethernet"}, w)
        _need(item, ("match_mac", "type"), w)
        item["type"] = str(item["type"]).strip().lower()
        if item["type"] not in _IFACE_TYPES:
            raise ValueError(f"{w}.type must be one of {list(_IFACE_TYPES)}.")
        item["enabled"] = bool(item.get("enabled", False))
        if item.get("wifi") is not None:
            wifi = _only(item["wifi"], {"mode", "country_code", "powersave", "access_point"}, f"{w}.wifi")
            _need(wifi, ("mode",), f"{w}.wifi")
            if str(wifi["mode"]).lower() not in _WIFI_MODES:
                raise ValueError(f"{w}.wifi.mode must be one of {list(_WIFI_MODES)}.")
            if wifi.get("access_point") is not None:
                ap = _only(wifi["access_point"], {"ssid", "psk", "channel", "hidden", "security", "ip"}, f"{w}.wifi.access_point")
                _need(ap, ("ssid",), f"{w}.wifi.access_point")
                if ap.get("ip") is not None:
                    ap["ip"] = _ip_settings(ap["ip"], f"{w}.wifi.access_point.ip", fleet)
                wifi["access_point"] = ap
            item["wifi"] = wifi
        if item.get("ethernet") is not None:
            eth = _only(item["ethernet"], {"ip"}, f"{w}.ethernet")
            if eth.get("ip") is not None:
                eth["ip"] = _ip_settings(eth["ip"], f"{w}.ethernet.ip", fleet)
            item["ethernet"] = eth
        out["interfaces"].append(item)
    if top.get("client_networks") is not None:
        out["client_networks"] = []
        for i, raw in enumerate(top["client_networks"]):
            w = f"client_networks[{i}]"
            cn = _only(raw, {"ssid", "psk", "priority", "hidden", "interface_mac", "ip"}, w, {"has_psk"})
            _need(cn, ("ssid",), w)
            if cn.get("psk") == PSK_MASK:
                cn.pop("psk")  # a masked echo from a preview: keep the stored password
            if cn.get("ip") is not None:
                cn["ip"] = _ip_settings(cn["ip"], f"{w}.ip", fleet)
            out["client_networks"].append(cn)
    if top.get("bridges") is not None:
        out["bridges"] = []
        for i, raw in enumerate(top["bridges"]):
            w = f"bridges[{i}]"
            br = _only(raw, {"name", "member_macs", "ip"}, w)
            _need(br, ("name", "member_macs"), w)
            if br.get("ip") is not None:
                br["ip"] = _ip_settings(br["ip"], f"{w}.ip", fleet)
            out["bridges"].append(br)
    if top.get("nameservers") is not None:
        if not isinstance(top["nameservers"], list):
            raise ValueError("nameservers must be a list of addresses.")
        out["nameservers"] = [str(n) for n in top["nameservers"]]
    for ap_iface in out["interfaces"]:
        ap = ((ap_iface.get("wifi") or {}).get("access_point")) or {}
        if ap.get("psk") == PSK_MASK:
            ap.pop("psk")
    return out


_NETWORK_KEYS = {"interfaces": "match_mac", "client_networks": "ssid", "bridges": "name"}


def merge_network(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` into a network configuration.

    Objects merge recursively (``null`` deletes a key). The three keyed lists are upserted item by item
    (``interfaces`` by ``match_mac``, ``client_networks`` by ``ssid``, ``bridges`` by ``name``), so changing one
    network does not drop the others; removing an item needs mode ``replace``. Other lists (``nameservers``) replace.
    """
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (patch or {}).items():
        ident = _NETWORK_KEYS.get(key)
        if value is None:
            out.pop(key, None)
        elif ident and isinstance(value, list):
            items = [copy.deepcopy(i) for i in out.get(key) or [] if isinstance(i, dict)]
            for item in value:
                if not isinstance(item, dict) or not item.get(ident):
                    raise ValueError(f"{key} items need a {ident} (merge matches on it); use mode 'replace' to send a whole list.")
                match = next((i for i in items if str(i.get(ident, "")).lower() == str(item[ident]).lower()), None)
                if match is None:
                    items.append(copy.deepcopy(item))
                else:
                    match.update(merge_patch(match, {k: v for k, v in item.items() if k != ident}))
            out[key] = items
        else:
            out[key] = merge_patch(out.get(key), value)
    return out


def mask_network_secrets(cfg: Any) -> Any:
    """Copy with every ``psk`` replaced by ``***`` (previews and next-arguments never carry a Wi-Fi password)."""
    out = copy.deepcopy(cfg)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "psk" and value:
                    node[key] = PSK_MASK
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    return out


def network_psk_supplied(cfg: Any) -> list[str]:
    """SSIDs for which the request carries a (non-masked) password."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("psk") and node.get("psk") != PSK_MASK and node.get("ssid"):
                found.append(str(node["ssid"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cfg)
    return found


def _net_flat(cfg: Any) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _NETWORK_READONLY or key == "has_psk":
                    continue
                walk(f"{prefix}.{key}" if prefix else key, value)
        elif isinstance(node, list) and node and all(isinstance(i, dict) for i in node):
            for idx, item in enumerate(node):
                ident = item.get("match_mac") or item.get("ssid") or item.get("name") or idx
                walk(f"{prefix}[{ident}]", item)
        elif node not in (None, "", [], {}):
            flat[prefix] = node

    walk("", cfg or {})
    return flat


def network_diff(old: Any, new: Any) -> list[dict[str, Any]]:
    """Leaf-level differences between two network configurations (list items keyed by MAC / SSID / name)."""
    before, after = _net_flat(mask_network_secrets(old)), _net_flat(mask_network_secrets(new))
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path), after.get(path)
        if path in before and path in after:
            if a != b:
                rows.append({"path": path, "change": "changed", "old": a, "new": b})
        elif path in after:
            rows.append({"path": path, "change": "added", "new": b})
        else:
            rows.append({"path": path, "change": "removed", "old": a})
    return rows


def network_brief(cfg: Any) -> dict[str, Any] | None:
    """Short view of a stored network configuration (no secrets)."""
    if not isinstance(cfg, dict) or not cfg:
        return None
    return {
        "version": cfg.get("version"),
        "interfaces": [
            {
                k: v
                for k, v in (
                    ("match_mac", i.get("match_mac")),
                    ("type", i.get("type")),
                    ("enabled", i.get("enabled")),
                    ("mode", (i.get("wifi") or {}).get("mode")),
                    ("dhcp_v4", ((i.get("ethernet") or {}).get("ip") or {}).get("dhcp_v4")),
                    ("static_ipv4", (((i.get("ethernet") or {}).get("ip") or {}).get("ipv4") or {}).get("address")),
                )
                if v is not None
            }
            for i in cfg.get("interfaces") or []
        ],
        "client_networks": [
            {"ssid": c.get("ssid"), "has_psk": bool(c.get("has_psk")), "priority": c.get("priority")}
            for c in cfg.get("client_networks") or []
        ],
        "bridges": [b.get("name") for b in cfg.get("bridges") or []],
        "nameservers": cfg.get("nameservers") or [],
        "updated_at": cfg.get("updated_at"),
    }


# ------------------------------------------------------------ local secrets ---


class SecretInputError(ValueError):
    """The local secret source could not be read. The message never contains the secret."""


def read_local_secret(secret_file: str | None, secret_env: str | None, *, max_bytes: int = 64 * 1024) -> str:
    """Read a secret from a local file or an environment variable of this process.

    Exactly one source must be given. Trailing newlines are stripped. The value is returned to the caller
    only; errors name the source, never the content.
    """
    import os

    if bool(secret_file) == bool(secret_env):
        raise SecretInputError("Pass exactly one of secret_file (a local path) or secret_env (an environment variable name).")
    if secret_env:
        value = os.environ.get(secret_env.strip())
        if not value:
            raise SecretInputError(f"Environment variable {secret_env!r} is not set (or empty) in the MCP process.")
        return value
    path = os.path.expanduser(str(secret_file).strip())
    try:
        size = os.path.getsize(path)
        if size > max_bytes:
            raise SecretInputError(f"{path} is {size} bytes; the limit is {max_bytes}.")
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise SecretInputError(f"Cannot read {path}: {exc.strerror or type(exc).__name__}.") from None
    try:
        value = raw.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError:
        raise SecretInputError(f"{path} is not UTF-8 text.") from None
    if not value:
        raise SecretInputError(f"{path} is empty.")
    return value
