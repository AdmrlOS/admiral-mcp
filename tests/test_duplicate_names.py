from __future__ import annotations

import httpx

from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings
from admrl_mcp.resolve import DeviceResolver


def _client(devices: list[dict], get_device_id: str | None = None) -> AdmiralClient:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/devices"):
            fleet_id = request.url.params.get("fleet_id")
            rows = devices if not fleet_id else [d for d in devices if (d.get("fleet") or {}).get("id") == fleet_id]
            return httpx.Response(200, json={"code": 200, "data": rows})
        if path.endswith("/search"):
            return httpx.Response(200, json={"code": 200, "data": {"devices": None, "total": 0}})
        if get_device_id and path.endswith(f"/devices/{get_device_id}"):
            row = next(d for d in devices if d["id"] == get_device_id)
            return httpx.Response(200, json={"code": 200, "data": row})
        raise AssertionError(request.url)

    return AdmiralClient(
        settings=Settings(
            api_base="https://api.admrl.co/v1",
            token_id="tok",
            secret_key="sec",
            org_id="org-1",
        ),
        transport=httpx.MockTransport(handler),
    )


TWIN_A = {
    "id": "11111111-1111-1111-1111-111111111111",
    "name": "pi",
    "status": "online",
    "ipAddress": "10.0.0.5",
    "hardwareType": "Raspberry Pi 4",
    "notes": "",
    "fleet": {"id": "f-1", "name": "field"},
}
TWIN_B = {
    "id": "22222222-2222-2222-2222-222222222222",
    "name": "pi",
    "status": "offline",
    "ipAddress": "10.0.0.9",
    "hardwareType": "Raspberry Pi 3",
    # Incidental mention that used to win the score tie-break silently.
    "notes": "pi unit for lab",
    "fleet": {"id": "f-2", "name": "depot"},
}


def test_duplicate_names_never_silently_pick_via_notes():
    """Same-named twins: notes/hardware text must not break the tie.

    Regression: TWIN_B scored +20 for `q in notes` and was returned as a
    unique match even though the query was the bare shared name.
    """
    found = DeviceResolver(_client([TWIN_A, TWIN_B])).find("pi", org_id="org-1")
    assert found["match"] is None
    assert {c["id"] for c in found["candidates"]} == {TWIN_A["id"], TWIN_B["id"]}
    assert "Pass a device id (UUID)" in found["error"]


def test_duplicate_names_still_disambiguated_by_tag_query():
    """A tag query is not an exact name match, so scoring may pick a winner."""
    twin_b_tagged = dict(TWIN_B, effective_tags=[{"key": "role", "value": "printer"}])
    found = DeviceResolver(_client([TWIN_A, twin_b_tagged])).find(
        "role=printer", org_id="org-1"
    )
    assert found["match"] is not None
    assert found["match"]["id"] == TWIN_B["id"]


def test_duplicate_names_disambiguated_by_fleet_filter():
    found = DeviceResolver(_client([TWIN_A, TWIN_B])).find("pi", org_id="org-1", fleet_id="f-2")
    assert found["match"] is not None
    assert found["match"]["id"] == TWIN_B["id"]


def test_uuid_short_circuits_to_direct_get_even_with_name_twin():
    """Platform identity is the UUID: bypass scoring entirely."""
    client = _client([TWIN_A, TWIN_B], get_device_id=TWIN_B["id"])
    found = DeviceResolver(client).find(TWIN_B["id"], org_id="org-1")
    assert found["match"]["id"] == TWIN_B["id"]
    assert found["candidates"] == []
