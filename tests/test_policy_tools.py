"""Device policy / duplicate / diagnose-schedule tools (CONTRACT §5)."""

from __future__ import annotations

import json

import httpx

from admrl_mcp import server

from test_w1_tools import CONFIG, DEV, FLEET, RESOLVED, FakeAPI, assert_pat, envelope, install


def test_get_device_policy_reads_status_and_override(monkeypatch):
    doc = {
        "kind": "Device",
        "metadata": {"resourceVersion": "rv9"},
        "spec": {"policy": {"health": {"maxOffline": "5d"}}},
        "status": {
            "protocol": 1,
            "policy": {
                "requested": {"policy": {"health": {"maxOffline": "5d"}}, "sources": {"health.maxOffline": "device"}},
                "reported": {"health": {"maxOffline": "5d"}},
                "clamped": [],
            },
        },
    }
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json=doc, headers={"ETag": '"rv9"'}))
    install(monkeypatch, api)

    out = json.loads(server.get_device_policy("dan-qemu-3"))

    req = api.only()
    assert req.headers["Accept"] == "application/json"
    assert_pat(req)
    assert out["policy"]["requested"]["sources"]["health.maxOffline"] == "device"
    assert out["override"] == {"health": {"maxOffline": "5d"}}
    assert out["resource_version"] == "rv9"
    assert "note" not in out


def test_get_device_policy_without_status_policy_says_so(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"spec": {}, "status": {"protocol": 0}}))
    install(monkeypatch, api)

    out = json.loads(server.get_device_policy("dan-qemu-3"))

    assert out["policy"] is None and "protocol 0" in out["note"]


def test_get_device_policy_unresolved_returns_candidates(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: {"match": None, "candidates": []})
    out = json.loads(server.get_device_policy("nope"))
    assert out["match"] is None and not api.seen


def test_duplicate_fleet_by_uuid(monkeypatch):
    api = FakeAPI()
    result = {
        "fleet": {"id": "new", "name": "prod copy"},
        "copied": ["tags"],
        "skipped": [{"field": "airdetect", "reason": "billed feature; enable it on the new fleet"}],
    }
    api.on("POST", f"/fleets/{FLEET}/duplicate", httpx.Response(201, json=envelope(result)))
    install(monkeypatch, api)

    out = json.loads(server.duplicate_fleet(FLEET, " prod copy ", copy_configuration=False))

    req = api.only()
    assert_pat(req)
    assert json.loads(req.content) == {"name": "prod copy", "copyConfiguration": False}
    assert out["result"]["skipped"][0]["field"] == "airdetect"


def test_duplicate_fleet_resolves_name_and_sends_description(monkeypatch):
    api = FakeAPI()
    api.on("GET", "/fleets", httpx.Response(200, json=envelope([{"id": FLEET, "name": "prod"}, {"id": "x", "name": "prod-2"}])))
    api.on("POST", f"/fleets/{FLEET}/duplicate", httpx.Response(201, json=envelope({"fleet": {"id": "new"}})))
    install(monkeypatch, api)

    out = json.loads(server.duplicate_fleet("prod", "copy", description="d"))

    assert out["source"]["id"] == FLEET
    post = [r for r in api.seen if r.method == "POST"][0]
    assert json.loads(post.content) == {"name": "copy", "description": "d", "copyConfiguration": True}


def test_duplicate_fleet_requires_name_and_reports_api_errors(monkeypatch):
    api = FakeAPI()
    api.on("POST", f"/fleets/{FLEET}/duplicate", httpx.Response(403, json={"msg": "need fleet_creator"}))
    install(monkeypatch, api)

    assert "name is required" in server.duplicate_fleet(FLEET, "  ")
    assert not api.seen
    err = json.loads(server.duplicate_fleet(FLEET, "x"))
    assert "error" in err and "403" in json.dumps(err)


def test_duplicate_configuration_version(monkeypatch):
    api = FakeAPI()
    result = {
        "configuration": {"id": "new", "name": "kiosk v2", "latestVersion": 1},
        "sourceVersion": 7,
        "skipped": [{"field": "secretFiles", "reason": "secret files are not copied; attach them to the new configuration"}],
    }
    api.on("POST", f"/configurations/{CONFIG}/duplicate", httpx.Response(201, json=envelope(result)))
    install(monkeypatch, api)

    out = json.loads(server.duplicate_configuration(CONFIG, "kiosk v2", version=7))

    req = api.only()
    assert json.loads(req.content) == {"name": "kiosk v2", "version": 7}
    assert out["result"]["sourceVersion"] == 7

    api.seen.clear()
    server.duplicate_configuration(CONFIG, "latest")
    assert json.loads(api.only().content) == {"name": "latest"}  # no version = latest


def test_duplicate_configuration_rejects_negative_version(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    assert "version" in server.duplicate_configuration(CONFIG, "x", version=-1)
    assert not api.seen


SCHEDULE = {
    "heartbeatIntervalMs": 60000,
    "presenceWindowMs": 150000,
    "overdue": False,
    "restart": {"at": "2026-10-10T01:00:00Z", "reason": "deadman", "basis": "projected", "thenEveryMs": 3600000},
    "source": "device",
}


def test_diagnose_surfaces_schedule(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(200, json=envelope({"verdict": {"online": False}, "schedule": SCHEDULE})))
    install(monkeypatch, api)
    monkeypatch.setattr(server.DeviceResolver, "find", lambda self, q, org_id=None, fleet_id=None: dict(RESOLVED))

    [report] = json.loads(server.diagnose_device("dan-qemu-3"))["reports"]

    assert report["schedule"] == SCHEDULE
    assert report["diagnosis"]["schedule"] == SCHEDULE


def test_diagnose_without_schedule_has_no_schedule_key(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(200, json=envelope({"verdict": {"online": True}})))
    install(monkeypatch, api)
    monkeypatch.setattr(server.DeviceResolver, "find", lambda self, q, org_id=None, fleet_id=None: dict(RESOLVED))

    [report] = json.loads(server.diagnose_device("dan-qemu-3"))["reports"]

    assert "schedule" not in report
