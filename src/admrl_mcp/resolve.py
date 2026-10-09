from __future__ import annotations

import re
from typing import Any

from .client import AdmiralAPIError, AdmiralClient

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def is_uuid(value: str | None) -> bool:
    return bool(value and UUID_RE.match(value.strip()))


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "devices", "hosts", "fleets", "organisations", "organizations"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _tags(device: dict[str, Any]) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    for key in ("effective_tags", "tags", "fleet_tags"):
        raw = device.get(key) or []
        if isinstance(raw, list):
            for tag in raw:
                if isinstance(tag, dict) and tag.get("key"):
                    tags.append({"key": str(tag.get("key")), "value": str(tag.get("value") or "")})
    return tags


def _tag_blob(device: dict[str, Any]) -> str:
    parts = []
    for tag in _tags(device):
        parts.append(f"{tag['key']}={tag['value']}")
        parts.append(tag["key"])
        parts.append(tag["value"])
    return " ".join(parts).lower()


def _status(device: dict[str, Any]) -> str:
    status = _norm(device.get("status"))
    online = device.get("isOnline") or device.get("onlineStatus") or {}
    if isinstance(online, dict) and online.get("isOnline") is True:
        return "online"
    if isinstance(online, dict) and online.get("isOnline") is False and not status:
        return "offline"
    return status or "unknown"


def _ip(device: dict[str, Any]) -> str:
    if device.get("ipAddress"):
        return str(device["ipAddress"])
    spec = device.get("systemSpec") or device.get("system_spec") or {}
    nets = []
    if isinstance(spec, dict):
        nets = spec.get("network") or spec.get("Network") or []
    if not nets:
        nets = device.get("networkInterfaces") or []
    for iface in nets if isinstance(nets, list) else []:
        if not isinstance(iface, dict):
            continue
        ip = iface.get("ipV4Address") or iface.get("ipv4_address") or iface.get("ip")
        if ip and ip not in ("127.0.0.1", "::1"):
            return str(ip)
    return ""


def summarise_device(device: dict[str, Any]) -> dict[str, Any]:
    fleet = device.get("fleet") or {}
    online = device.get("isOnline") or device.get("onlineStatus") or {}
    last_seen = None
    latency = None
    if isinstance(online, dict):
        last_seen = online.get("lastSeen") or online.get("last_seen")
        latency = online.get("latencyMs") or online.get("latency_ms")
    tags = []
    seen = set()
    for tag in _tags(device):
        key = (tag["key"].lower(), tag["value"].lower())
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return {
        "id": device.get("id"),
        "name": device.get("name"),
        "status": _status(device),
        "ip": _ip(device) or None,
        "hardware": device.get("hardwareType") or device.get("hardware_type"),
        "fleet": {"id": fleet.get("id"), "name": fleet.get("name")} if fleet else None,
        "last_seen": last_seen,
        "latency_ms": latency,
        "tags": tags,
        "notes": device.get("notes") or None,
        "location": device.get("location"),
    }


def _tokens(value: str) -> list[str]:
    return [t for t in re.split(r"[\s,=/:_\-]+", _norm(value)) if t]


def match_score(query: str, device: dict[str, Any]) -> int:
    q = _norm(query)
    if not q:
        return 0
    name = _norm(device.get("name"))
    notes = _norm(device.get("notes"))
    hardware = _norm(device.get("hardwareType") or device.get("hardware_type"))
    ip = _norm(_ip(device))
    device_id = _norm(device.get("id"))
    tags = _tag_blob(device)
    fleet = device.get("fleet") or {}
    fleet_name = _norm(fleet.get("name") if isinstance(fleet, dict) else "")
    score = 0
    if q == name or q == device_id:
        score += 100
    if name.startswith(q):
        score += 40
    if q in name:
        score += 30
    if q in tags:
        score += 35
    if q in notes:
        score += 20
    if q in hardware:
        score += 15
    if q in fleet_name:
        score += 15
    if q == ip or (q and q in ip):
        score += 25
    tokens = _tokens(q)
    hay = " ".join([name, notes, hardware, tags, fleet_name, ip])
    if tokens and all(t in hay for t in tokens):
        score += 20
    hay_tokens = set(_tokens(hay))
    if tokens and any(t in hay_tokens for t in tokens):
        score += 15
    return score


class DeviceResolver:
    def __init__(self, client: AdmiralClient):
        self.client = client

    def resolve_org(self, organization_id: str | None = None) -> str:
        if organization_id:
            return organization_id
        if self.client.settings.org_id:
            return self.client.settings.org_id
        if self.client.settings.org_optional:
            return ""  # hosted: the backend uses the grant's organisation when no header is sent
        orgs = self.client.list_organisations()
        if len(orgs) == 1:
            return str(orgs[0].get("id") or orgs[0].get("ID"))
        if not orgs:
            raise AdmiralAPIError(400, "No organisations visible to this PAT. Set ADMRL_ORG_ID.")
        names = ", ".join(f"{o.get('name')} ({o.get('id')})" for o in orgs[:12])
        raise AdmiralAPIError(
            400,
            "Multiple organisations. Pass organization_id or set ADMRL_ORG_ID. Available: " + names,
        )

    def list_devices(
        self,
        *,
        org_id: str,
        query: str | None = None,
        status: str | None = None,
        fleet_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        payload = self.client.list_devices(
            org_id=org_id,
            search=query if query and not _looks_like_tag_query(query) else None,
            status=status,
            fleet_id=fleet_id,
            page=1,
            limit=min(max(limit, 1), 100),
        )
        devices = [d for d in _as_list(payload) if isinstance(d, dict)]
        if query:
            ranked = sorted(
                ((match_score(query, d), d) for d in devices),
                key=lambda item: item[0],
                reverse=True,
            )
            scored = [d for score, d in ranked if score > 0]
            if scored:
                devices = scored
            else:
                # Tag/role queries miss list-search; widen, but never return score-0 rows.
                wider = self.client.list_devices(
                    org_id=org_id,
                    status=status,
                    fleet_id=fleet_id,
                    page=1,
                    limit=100,
                )
                extra = [d for d in _as_list(wider) if isinstance(d, dict)]
                ranked = sorted(
                    ((match_score(query, d), d) for d in extra),
                    key=lambda item: item[0],
                    reverse=True,
                )
                devices = [d for score, d in ranked if score > 0]
        return devices

    def list_not_online(
        self,
        *,
        org_id: str,
        fleet_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Devices the platform's counts.offline bucket counts: not currently online.

        The `status=offline` list filter matches almost nothing — claimed,
        erroring, and stale devices keep their provisioning status while
        `counts.offline` still counts them as not-online. Fetch unfiltered and
        keep rows whose resolved status is not `online`, client-side.
        """
        payload = self.client.list_devices(
            org_id=org_id, status=None, fleet_id=fleet_id, page=1, limit=min(max(limit, 1), 100)
        )
        devices = [d for d in _as_list(payload) if isinstance(d, dict)]
        return [d for d in devices if _status(d) != "online"]

    def find(
        self,
        query: str,
        *,
        org_id: str,
        status: str | None = None,
        fleet_id: str | None = None,
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if is_uuid(query):
            try:
                device = self.client.get_device(query, org_id=org_id)
                if isinstance(device, dict):
                    return {"match": summarise_device(device), "candidates": [], "raw": device}
            except AdmiralAPIError as exc:
                if exc.status not in (404, 400):
                    raise

        devices = self.list_devices(org_id=org_id, query=query, status=status, fleet_id=fleet_id, limit=100)
        search = None
        if not devices:
            search = self.client.search(query, org_id=org_id, limit=20)
            devices = self._devices_from_search(search, org_id=org_id, status=status, fleet_id=fleet_id, query=query)
        summarised = [summarise_device(d) for d in devices]
        if not summarised:
            return {
                "match": None,
                "candidates": [],
                "search": search,
                "error": f"No device matched {query!r}.",
            }
        exact = [s for s in summarised if _norm(s.get("name")) == _norm(query)]
        if len(exact) == 1:
            raw = next(d for d in devices if _norm(d.get("name")) == _norm(query))
            return {"match": exact[0], "candidates": [], "raw": raw}
        if len(exact) > 1:
            # The query IS the name here, so it carries no disambiguating
            # intent. A score lead between same-named twins would come from
            # incidental notes/hardware/fleet text — never silently pick one
            # (reboot_device sits behind this resolver). UUID queries never
            # reach this branch; they short-circuit to a direct GET.
            return {
                "match": None,
                "candidates": exact[:15],
                "search": search,
                "error": (
                    f"{len(exact)} devices are named {query!r}. "
                    "Pass a device id (UUID), or add a tag, fleet, or IP to the query."
                ),
            }
        unique = _unique_top(query, devices)
        if unique is not None:
            return {"match": summarise_device(unique), "candidates": [], "raw": unique}
        if len(summarised) == 1:
            return {"match": summarised[0], "candidates": [], "raw": devices[0]}
        return {
            "match": None,
            "candidates": summarised[:15],
            "search": search,
            "error": f"{len(summarised)} devices matched {query!r}. Pass a device id or a more specific name/tag.",
        }

    def _devices_from_search(
        self,
        search: Any,
        *,
        org_id: str,
        status: str | None,
        fleet_id: str | None,
        query: str,
    ) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(device: dict[str, Any]) -> None:
            device_id = str(device.get("id") or "")
            if not device_id or device_id in seen:
                return
            seen.add(device_id)
            found.append(device)

        payload = search if isinstance(search, dict) else {}
        for row in payload.get("devices") or []:
            if isinstance(row, dict):
                add(row)
        for fleet in payload.get("fleets") or []:
            if not isinstance(fleet, dict):
                continue
            fid = fleet.get("id")
            if not fid or (fleet_id and fid != fleet_id):
                continue
            extra = self.list_devices(org_id=org_id, status=status, fleet_id=str(fid), limit=100)
            for device in extra:
                add(device)
        ranked = sorted(((match_score(query, d), d) for d in found), key=lambda item: item[0], reverse=True)
        return [d for score, d in ranked if score > 0]


def _unique_top(query: str, devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not devices:
        return None
    ranked = sorted(((match_score(query, d), d) for d in devices), key=lambda item: item[0], reverse=True)
    top_score, top = ranked[0]
    if top_score <= 0:
        return None
    if len(ranked) == 1:
        return top
    second = ranked[1][0]
    if top_score >= second + 10:
        return top
    return None


def _looks_like_tag_query(query: str) -> bool:
    q = query.strip()
    return "=" in q or ":" in q or " " in q


def collect_ips(device: dict[str, Any], network_status: Any | None = None) -> list[dict[str, Any]]:
    """IPs for a device, spec-first.

    The system spec attached to the device object (GET /devices/{id} →
    systemSpec.network) is the canonical IP source and keeps interface
    attribution. The flat list ipAddress is a fallback only and must not
    shadow a spec entry (which knows its interface). Live network_status is
    an optional extra, never required.
    """
    ips: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ip: str | None, *, iface: str | None = None, source: str, version: str = "v4") -> None:
        if not ip or ip in seen or ip in ("127.0.0.1", "::1"):
            return
        seen.add(ip)
        ips.append({"ip": ip, "interface": iface, "version": version, "source": source})

    spec = device.get("systemSpec") or device.get("system_spec") or {}
    if isinstance(spec, dict):
        for iface in spec.get("network") or spec.get("Network") or []:
            if not isinstance(iface, dict):
                continue
            add(iface.get("ipV4Address") or iface.get("ipv4_address"), iface=iface.get("name"), source="system_spec")
            add(
                iface.get("ipV6Address") or iface.get("ipv6_address"),
                iface=iface.get("name"),
                source="system_spec",
                version="v6",
            )

    add(_ip(device) or None, source="device_list")

    if isinstance(network_status, dict):
        for iface in network_status.get("interfaces") or []:
            if not isinstance(iface, dict):
                continue
            name = iface.get("name")
            for addr in iface.get("ips") or []:
                if isinstance(addr, dict):
                    add(addr.get("address") or addr.get("ip"), iface=name, source="network_status")
                elif isinstance(addr, str):
                    add(addr, iface=name, source="network_status")

    return ips
