"""create_support_ticket / list_support_tickets: request bodies, subject resolution, gating."""

from __future__ import annotations

import json

import httpx
import pytest

from admrl_mcp import server
from test_w1_tools import CONFIG, DEV, FLEET, FakeAPI, assert_pat, envelope, install

TICKET = {
    "id": "t-1",
    "ref": "T-123",
    "title": "Edge down",
    "status": "OPEN",
    "priority": 2,
    "supportType": "TECHNICAL",
    "createdAt": "2026-10-09T00:00:00Z",
}


def created(**extra):
    return httpx.Response(201, json=envelope({**TICKET, **extra}))


def call(**kw):
    return json.loads(server.create_support_ticket(title="Edge down", description="It stopped. Tried reboot.", **kw))


def test_plain_ticket_body_and_summary(monkeypatch):
    api = FakeAPI()
    api.on("POST", "/helpdesk/tickets", created())
    install(monkeypatch, api)
    out = call()
    req = api.only()
    assert_pat(req)
    assert json.loads(req.content) == {
        "title": "Edge down",
        "description": "It stopped. Tried reboot.",
        "priority": 2,
        "supportType": "TECHNICAL",
    }
    assert list(out)[0] == "summary"
    assert out["summary"] == "Ticket T-123 raised (technical, normal)"
    assert out["ticket"]["ref"] == "T-123" and out["diagnostics"] is None
    assert "Help page" in out["next_step"]


@pytest.mark.parametrize("prio,num", [("high", 1), ("normal", 2), ("LOW", 3)])
def test_priority_mapping(monkeypatch, prio, num):
    api = FakeAPI()
    api.on("POST", "/helpdesk/tickets", created())
    install(monkeypatch, api)
    call(priority=prio, support_type="Billing")
    body = json.loads(api.only().content)
    assert body["priority"] == num and body["supportType"] == "BILLING"


@pytest.mark.parametrize("bad", ["urgent", "sev1", "0", ""])
def test_urgent_not_exposed(monkeypatch, bad):
    api = FakeAPI()
    install(monkeypatch, api)
    out = call(priority=bad) if bad else None
    if out is None:
        return
    assert "error" in out and "urgent" not in out["choices"] and "Help page" in out["note"]
    assert api.seen == []


def test_bad_support_type(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    assert "error" in call(support_type="sales") and api.seen == []


def test_device_subject_with_diagnostics_requested(monkeypatch):
    api = FakeAPI()
    api.on(
        "POST",
        "/helpdesk/tickets",
        created(
            subject={"type": "device", "id": DEV, "name": "dan-qemu-3"},
            diagnosticsBundleId="b-1",
            diagnostics={"status": "requested", "bundleId": "b-1"},
        ),
    )
    install(monkeypatch, api)
    out = call(device="dan-qemu-3", attach_diagnostics=True)
    body = json.loads(api.only().content)
    assert body["subject"] == {"type": "device", "id": DEV} and body["attachDiagnostics"] is True
    assert out["summary"] == "Ticket T-123 raised (technical, normal) about device dan-qemu-3; diagnostics bundle requested"
    assert out["diagnostics"]["bundleId"] == "b-1"


def test_diagnostics_unavailable_is_surfaced(monkeypatch):
    api = FakeAPI()
    api.on(
        "POST",
        "/helpdesk/tickets",
        created(diagnostics={"status": "unavailable", "reason": "device offline"}),
    )
    install(monkeypatch, api)
    out = call(device="x", attach_diagnostics=True)
    assert "diagnostics unavailable (device offline)" in out["summary"]
    assert out["diagnostics"]["reason"] == "device offline"
    assert out["ticket"]["subject"]["name"] == "dan-qemu-3"  # falls back to the resolved subject


def test_device_without_attach_omits_flag(monkeypatch):
    api = FakeAPI()
    api.on("POST", "/helpdesk/tickets", created())
    install(monkeypatch, api)
    call(device="x")
    assert "attachDiagnostics" not in json.loads(api.only().content)


def test_ambiguous_device_creates_nothing(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    cands = [{"id": DEV, "name": "a"}, {"id": "other", "name": "a"}]
    monkeypatch.setattr(
        server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: {"match": None, "candidates": cands, "organization_id": "org-1"}
    )
    out = call(device="a")
    assert out["candidates"] == cands and "No ticket created" in out["note"]
    assert api.seen == []


def test_fleet_uuid_is_verified_then_used(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/fleets/{FLEET}", httpx.Response(200, json=envelope({"id": FLEET, "name": "Shops"})))
    api.on("POST", "/helpdesk/tickets", created())
    install(monkeypatch, api)
    out = call(fleet=FLEET)
    body = json.loads(api.seen[-1].content)
    assert body["subject"] == {"type": "fleet", "id": FLEET}
    assert "about fleet Shops" in out["summary"]


def test_configuration_by_name(monkeypatch):
    api = FakeAPI()
    api.on(
        "GET",
        "/configurations",
        lambda r: httpx.Response(
            200, json=envelope([{"id": CONFIG, "name": "kiosk"}, {"id": "zz", "name": "kiosk-beta"}])
        ),
    )
    api.on("POST", "/helpdesk/tickets", created())
    install(monkeypatch, api)
    call(configuration="kiosk")
    assert api.seen[0].url.params["search"] == "kiosk"
    assert json.loads(api.seen[-1].content)["subject"] == {"type": "configuration", "id": CONFIG}


def test_ambiguous_fleet_returns_candidates(monkeypatch):
    api = FakeAPI()
    api.on(
        "GET",
        "/fleets",
        httpx.Response(200, json=envelope([{"id": "f1", "name": "Shops"}, {"id": "f2", "name": "shops"}])),
    )
    install(monkeypatch, api)
    out = call(fleet="Shops")
    assert len(out["candidates"]) == 2 and out["match"] is None
    assert [r.method for r in api.seen] == ["GET"]


def test_unknown_fleet_name_no_ticket(monkeypatch):
    api = FakeAPI()
    api.on("GET", "/fleets", httpx.Response(200, json=envelope([])))
    install(monkeypatch, api)
    out = call(fleet="nope")
    assert out["match"] is None and out["candidates"] == []
    assert all(r.method == "GET" for r in api.seen)


def test_multiple_subjects_rejected(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    out = call(device="a", fleet="b")
    assert "at most one" in out["error"] and api.seen == []


def test_attach_diagnostics_requires_device(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    assert "requires a device" in call(fleet=FLEET, attach_diagnostics=True)["error"]
    assert "requires a device" in call(attach_diagnostics=True)["error"]
    assert api.seen == []


def test_subject_404_from_backend(monkeypatch):
    api = FakeAPI()
    api.on("POST", "/helpdesk/tickets", httpx.Response(404, json={"msg": "Subject not found"}))
    install(monkeypatch, api)
    out = call(device="x")
    assert "404" in out["error"] and "Subject not found" in out["error"]


def test_title_and_description_required(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    assert "error" in json.loads(server.create_support_ticket(title=" ", description="d"))
    assert "error" in json.loads(server.create_support_ticket(title="t", description=""))
    assert api.seen == []


def test_list_support_tickets_compact(monkeypatch):
    api = FakeAPI()
    page = {
        "tickets": [{**TICKET, "previewText": "long", "slackChannel": "x", "subject": {"type": "fleet", "id": FLEET, "name": "Shops"}}],
        "hasNext": True,
        "nextCursor": "c2",
        "totalCount": 7,
    }
    api.on("GET", "/helpdesk/tickets", httpx.Response(200, json=page))
    install(monkeypatch, api)
    out = json.loads(server.list_support_tickets(limit=1))
    req = api.only()
    assert req.url.params["first"] == "1" and "after" not in req.url.params
    assert out["summary"] == "1 of 7 tickets" and out["next_cursor"] == "c2" and out["has_next"] is True
    row = out["tickets"][0]
    assert row["ref"] == "T-123" and row["subject"]["name"] == "Shops" and "slackChannel" not in row


def test_annotations():
    import asyncio

    by = {t.name: t.annotations for t in asyncio.run(server.mcp.list_tools())}
    a = by["create_support_ticket"]
    assert (a.readOnlyHint, a.destructiveHint, a.idempotentHint) == (False, False, False)
    assert by["list_support_tickets"].readOnlyHint is True
