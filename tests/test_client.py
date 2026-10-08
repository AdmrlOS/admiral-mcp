from __future__ import annotations

import httpx
import pytest

from admrl_mcp.client import AdmiralAPIError, AdmiralClient, log_page_cursor, parse_rfc3339, _unwrap
from admrl_mcp.config import Settings
from admrl_mcp.resolve import DeviceResolver


def test_unwrap_keeps_counts_and_pagination():
    payload = {
        "code": 200,
        "msg": "Success",
        "data": [{"id": "1", "name": "a"}],
        "counts": {"online": 1, "offline": 0, "total": 1},
        "pagination": {"page": 1, "limit": 10, "total": 1, "totalPages": 1},
    }
    unwrapped = _unwrap(payload)
    assert unwrapped["items"][0]["name"] == "a"
    assert unwrapped["counts"]["total"] == 1


def _settings() -> Settings:
    return Settings(
        api_base="https://api.admrl.co/v1",
        token_id="tok",
        secret_key="sec",
        org_id="org-1",
    )


def test_client_sends_pat_and_org_headers():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["url"] = str(request.url)
        body = {
            "code": 200,
            "msg": "Success",
            "data": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "shop-printer-01",
                    "status": "offline",
                    "ipAddress": "10.1.1.42",
                    "effective_tags": [{"key": "role", "value": "printer"}],
                    "fleet": {"id": "f1", "name": "retail"},
                }
            ],
            "counts": {"offline": 1, "online": 0, "total": 1},
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    client = AdmiralClient(settings=_settings(), transport=transport)
    payload = client.list_devices(search="printer", status="offline")
    assert captured["headers"]["x-api-token-id"] == "tok"
    assert captured["headers"]["x-api-secret-key"] == "sec"
    assert captured["headers"]["x-organization-id"] == "org-1"
    assert "status=offline" in captured["url"]
    assert payload["items"][0]["name"] == "shop-printer-01"


def test_client_surfaces_api_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"code": 401, "msg": "Authentication required"})

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(AdmiralAPIError) as exc:
        client.list_devices()
    assert exc.value.status == 401
    assert "Authentication required" in str(exc.value)


def test_resolver_finds_tagged_printer():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/devices"):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": [
                        {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "name": "shop-printer-01",
                            "status": "offline",
                            "ipAddress": "10.1.1.42",
                            "effective_tags": [{"key": "role", "value": "printer"}],
                        },
                        {
                            "id": "660e8400-e29b-41d4-a716-446655440000",
                            "name": "kiosk-01",
                            "status": "online",
                            "ipAddress": "10.1.1.9",
                            "effective_tags": [{"key": "role", "value": "kiosk"}],
                        },
                    ],
                },
            )
        raise AssertionError(request.url)

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    found = DeviceResolver(client).find("printer", org_id="org-1")
    assert found["match"]["name"] == "shop-printer-01"
    assert found["match"]["ip"] == "10.1.1.42"


def test_query_logs_falls_back_to_device_logs_on_403():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith("/metrics/logs/query"):
            return httpx.Response(403, json={"code": 403, "msg": "Access Denied"})
        if request.url.path.endswith("/logs"):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": {
                        "logs": [
                            {
                                "timestamp": "2026-09-21T04:15:16Z",
                                "level": "info",
                                "source": "workload.default-workload.stdout",
                                "message": "cuda-vector-add ok",
                            },
                            {
                                "timestamp": "2026-09-21T00:44:09Z",
                                "level": "info",
                                "source": "system",
                                "message": "older line",
                            },
                        ],
                        "has_more": True,
                    },
                },
            )
        raise AssertionError(request.url)

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    payload = client.fetch_logs(
        device_id="3bfd1506-542d-4cac-a664-5393b6b84d41",
        start="2026-09-21T00:00:00Z",
        end="2026-09-21T05:00:00Z",
        search=None,
        limit=1,
    )
    assert calls[0].endswith("/metrics/logs/query")
    assert any(c.endswith("/logs") for c in calls)
    logs = payload["logs"]
    assert len(logs) == 1
    assert logs[0]["message"] == "cuda-vector-add ok"
    assert payload["source"] == "device_logs"
    assert payload["query_error"]


def test_parse_rfc3339_handles_nanosecond_fractions():
    parsed = parse_rfc3339("2026-09-21T14:23:00.176512335Z")
    assert parsed.year == 2026
    assert parsed.microsecond == 176512
    assert parse_rfc3339("2026-09-21T14:23:00Z").microsecond == 0


def test_log_page_cursor_pages_older():
    logs = [
        {"timestamp": "2026-09-21T20:00:00.100000000Z", "level": "DEBUG", "message": "new"},
        {"timestamp": "2026-09-21T14:23:00.176512335Z", "level": "DEBUG", "message": "old"},
    ]
    cursor = log_page_cursor(logs, has_more=True)
    assert cursor is not None
    oldest = parse_rfc3339("2026-09-21T14:23:00.176512335Z")
    parsed_cursor = parse_rfc3339(cursor)
    assert parsed_cursor < oldest
    assert (oldest - parsed_cursor).total_seconds() <= 0.002
    assert cursor.endswith("Z")


def test_log_page_cursor_none_when_exhausted():
    logs = [{"timestamp": "2026-09-21T14:23:00Z", "level": "DEBUG", "message": "old"}]
    assert log_page_cursor(logs, has_more=False) is None
    assert log_page_cursor([], has_more=True) is None


def test_get_device_screenshot_uses_long_timeout():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "code": 200,
                "msg": "Success",
                "data": {
                    "deviceId": "d1",
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "imageData": "aGVsbG8=",
                    "timestamp": "2026-09-22T00:21:06Z",
                },
            },
        )

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    payload = client.get_device_screenshot("d1", display=0)
    assert captured["url"].split("?")[0].endswith("/devices/d1/screenshot")
    assert captured["params"] == {"display": "0"}
    assert captured["timeout"]["read"] == 90.0
    assert payload["imageData"] == "aGVsbG8="
    assert payload["format"] == "jpeg"
