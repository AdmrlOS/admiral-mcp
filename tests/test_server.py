import base64
import json

import httpx

from mcp.server.fastmcp import Image

from admrl_mcp import server
from admrl_mcp.client import AdmiralClient, parse_rfc3339
from admrl_mcp.config import Settings

DOTMATRIX = {
    "id": "5f0c2a1e-7b3d-4c8e-9a6f-1d2e3f4a5b6c",
    "name": "dotmatrixboi",
    "status": "online",
    "ipAddress": "192.0.2.10",
    "systemSpec": {
        "network": [
            {"name": "eth0", "macAddress": "02:00:c6:ce:88:b1"},
            {
                "name": "wlan0",
                "macAddress": "88:00:33:77:a2:26",
                "ipV4Address": "192.0.2.10",
                "ipV6Address": "fe80::1ab3:1cdd:6a39:9246",
            },
        ]
    },
}

RESOLVED = {
    "match": {"id": DOTMATRIX["id"], "name": "dotmatrixboi"},
    "candidates": [],
    "organization_id": "org-1",
}


def _settings() -> Settings:
    return Settings(api_base="https://api.test/v1", token_id="tok", secret_key="sec", org_id=None)


def _patch_resolution(monkeypatch):
    monkeypatch.setattr(
        server,
        "_resolve_device",
        lambda query, organization_id=None, fleet_id=None: dict(RESOLVED),
    )


def test_get_device_network_reads_spec_without_live_probe(monkeypatch):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith(f"/devices/{DOTMATRIX['id']}"):
            return httpx.Response(200, json={"code": 200, "msg": "Success", "data": DOTMATRIX})
        return httpx.Response(500, json={"code": 500, "msg": f"unexpected {request.url.path}"})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "get_client", lambda: client)
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_network("dotmatrixboi"))

    assert not any("network/status" in path for path in paths)
    assert out["primary_ip"] == "192.0.2.10"
    assert out["ips"][0] == {
        "ip": "192.0.2.10",
        "interface": "wlan0",
        "version": "v4",
        "source": "system_spec",
    }
    assert not any(row["interface"] is None for row in out["ips"])
    versions = {row["version"] for row in out["ips"]}
    assert {"v4", "v6"} <= versions


def test_get_device_logs_pages_with_before_cursor(monkeypatch):
    captured: dict = {}

    class StubClient:
        def fetch_logs(self, **kwargs):
            captured.update(kwargs)
            return {
                "logs": [
                    {"timestamp": "2026-09-21T14:23:00.176512335Z", "level": "DEBUG", "message": "old"},
                    {"timestamp": "2026-09-21T20:00:00.100000000Z", "level": "DEBUG", "message": "new"},
                ],
                "has_more": True,
                "source": "device_logs",
                "query_error": None,
            }

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    _patch_resolution(monkeypatch)

    out = json.loads(
        server.get_device_logs(
            "dotmatrixboi",
            lookback_hours=8,
            before="2026-09-21T22:00:00Z",
        )
    )

    assert captured["start"] == "2026-09-21T14:00:00Z"
    assert captured["end"] == "2026-09-21T22:00:00Z"
    assert out["page"]["has_more"] is True
    cursor = out["page"]["next_before"]
    assert cursor is not None
    oldest = parse_rfc3339("2026-09-21T14:23:00.176512335Z")
    parsed_cursor = parse_rfc3339(cursor)
    assert parsed_cursor < oldest
    assert (oldest - parsed_cursor).total_seconds() <= 0.002


def test_get_device_logs_page_end_without_more(monkeypatch):
    class StubClient:
        def fetch_logs(self, **kwargs):
            return {
                "logs": [{"timestamp": "2026-09-21T14:23:00Z", "level": "DEBUG", "message": "old"}],
                "has_more": False,
                "source": "device_logs",
                "query_error": None,
            }

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_logs("dotmatrixboi"))
    assert out["page"] == {"has_more": False, "next_before": None}


PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

def _screenshot_stub(payload: dict):
    class StubClient:
        def get_device_screenshot(self, device_id, **kwargs):
            captured = payload.setdefault("_call", {})
            captured.update({"device_id": device_id, **kwargs})
            return payload["response"]

    return StubClient()


def test_get_device_screenshot_returns_image_block(monkeypatch):
    payload = {
        "response": {
            "deviceId": DOTMATRIX["id"],
            "format": "png",
            "width": 640,
            "height": 480,
            "imageData": base64.b64encode(PNG_1PX).decode(),
            "timestamp": "2026-09-22T00:21:06Z",
        }
    }
    monkeypatch.setattr(server, "get_client", lambda: _screenshot_stub(payload))
    _patch_resolution(monkeypatch)

    out = server.get_device_screenshot("dotmatrixboi", display=0)
    assert isinstance(out, list) and len(out) == 2
    meta = json.loads(out[0])
    image = out[1]
    assert meta["device"]["name"] == "dotmatrixboi"
    assert meta["format"] == "png"
    assert meta["width"] == 640 and meta["height"] == 480
    assert isinstance(image, Image)
    assert image.data == PNG_1PX
    assert image.to_image_content().mimeType == "image/png"
    assert payload["_call"] == {
        "device_id": DOTMATRIX["id"],
        "org_id": "org-1",
        "display": 0,
    }


def test_get_device_screenshot_magic_bytes_beat_reported_format(monkeypatch):
    # The reported `format` can disagree with the bytes; trust the bytes.
    jpeg_bytes = b"\xff\xd8\xff\xe0" + b"stub-jpeg-body"
    payload = {
        "response": {
            "deviceId": DOTMATRIX["id"],
            "format": "png",
            "width": 640,
            "height": 480,
            "imageData": base64.b64encode(jpeg_bytes).decode(),
            "timestamp": "2026-09-22T00:21:06Z",
        }
    }
    monkeypatch.setattr(server, "get_client", lambda: _screenshot_stub(payload))
    _patch_resolution(monkeypatch)

    out = server.get_device_screenshot("dotmatrixboi")
    meta = json.loads(out[0])
    assert meta["format"] == "jpeg"
    assert out[1].to_image_content().mimeType == "image/jpeg"


def test_get_device_screenshot_empty_image_data(monkeypatch):
    payload = {
        "response": {"deviceId": DOTMATRIX["id"], "format": "png", "imageData": ""},
    }
    monkeypatch.setattr(server, "get_client", lambda: _screenshot_stub(payload))
    _patch_resolution(monkeypatch)

    out = server.get_device_screenshot("dotmatrixboi")
    assert "Empty screenshot data" in json.loads(out[0])["error"]


def test_get_device_screenshot_without_match_skips_capture(monkeypatch):
    payload = {"response": {}}

    def no_match(query, organization_id=None, fleet_id=None):
        return {"match": None, "candidates": [{"id": "x", "name": "other"}], "organization_id": "org-1"}

    monkeypatch.setattr(server, "get_client", lambda: _screenshot_stub(payload))
    monkeypatch.setattr(server, "_resolve_device", no_match)

    out = server.get_device_screenshot("missing")
    assert json.loads(out[0])["candidates"]
    assert "_call" not in payload


def test_list_devices_status_offline_returns_not_online_set(monkeypatch):
    """status=offline must surface claimed/stale devices the API filter hides."""

    class StubResolver:
        def __init__(self, client):
            pass

        def resolve_org(self, organization_id=None):
            return "org-1"

        def list_not_online(self, *, org_id, fleet_id=None, limit=100):
            assert fleet_id is None
            return [{"id": "claimed-1", "name": "r36s-kaira", "status": "claimed"}]

        def list_devices(self, **kwargs):
            raise AssertionError("status=offline must not use the API status filter")

    class StubClient:
        def list_devices(self, **kwargs):
            raise AssertionError("status=offline must not re-query with the status filter")

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    monkeypatch.setattr(server, "DeviceResolver", StubResolver)

    out = json.loads(server.list_devices(status="offline"))
    assert out["counts"] is None
    assert [d["id"] for d in out["devices"]] == ["claimed-1"]


SYSTEM_SERVICES = {
    "deviceId": DOTMATRIX["id"],
    "supervisor": "s6",
    "collectedAt": 1791089327000,
    "summary": {"total": 3, "stable": 1, "flapping": 1, "stopped": 1},
    "services": [
        {"name": "dhcpcd", "health": "stable", "normallyUp": True},
        {"name": "workload", "health": "flapping", "normallyUp": True, "reason": "4 unclean exits in the last 5 minutes"},
        {"name": "bluetoothd", "health": "stopped", "normallyUp": False},
    ],
}


def test_get_device_system_services_surfaces_problems(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith(f"/devices/{DOTMATRIX['id']}/system-services"):
            return httpx.Response(200, json={"code": 200, "msg": "ok", "data": SYSTEM_SERVICES})
        return httpx.Response(500, json={"code": 500, "msg": f"unexpected {request.url.path}"})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "get_client", lambda: client)
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_system_services("dotmatrixboi", service="workload", deaths=10))

    assert dict(seen[0].url.params) == {"name": "workload", "deaths": "10"}
    # Only the crash loop needs attention: a service that ships disabled is not a problem.
    assert out["needs_attention"] == [
        {"name": "workload", "health": "flapping", "reason": "4 unclean exits in the last 5 minutes"}
    ]
    assert out["system_services"]["summary"]["flapping"] == 1


def test_get_device_system_services_without_filters_sends_no_query(monkeypatch):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"code": 200, "msg": "ok", "data": {"services": []}})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "get_client", lambda: client)
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_system_services("dotmatrixboi"))

    assert dict(seen[0].url.params) == {}
    assert out["needs_attention"] == []


def test_get_device_system_services_offline_is_an_error_not_a_crash(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(504, json={"code": 504, "msg": "Device is not contactable"})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "get_client", lambda: client)
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_system_services("dotmatrixboi"))

    assert "error" in out


def test_get_device_system_services_unavailable_is_not_flagged(monkeypatch):
    data = dict(SYSTEM_SERVICES, services=[
        {"name": "wpa_supplicant", "health": "unavailable", "normallyUp": True, "reason": "no Wi-Fi hardware"},
        {"name": "bluetoothd", "health": "unavailable", "normallyUp": False},
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 200, "msg": "ok", "data": data})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "get_client", lambda: client)
    _patch_resolution(monkeypatch)

    out = json.loads(server.get_device_system_services("dotmatrixboi"))
    assert out["needs_attention"] == []
