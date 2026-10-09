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
