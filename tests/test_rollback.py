"""Manual rollback: rollback_device, the rollback rollout type and the rollback summary in device state."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from admrl_mcp import server
from admrl_mcp.watch import summarise_rollback, summarise_state

from test_ops_tools import FLEET, ROLLOUT, bodies, envelope, fleet_row, j, ok, page
from test_w1_tools import DEV, FakeAPI, install

SYSTEM = {"available": True, "fromVersion": "2.0.0", "toVersion": "1.9.0", "slot": "b"}
AGENT = {"available": True, "fromVersion": "0.9.1", "toVersion": "0.9.0"}


def state(rollback: dict[str, Any] | None) -> httpx.Response:
    return ok({"state": {"boot": {"rollback": rollback} if rollback is not None else {}}})


@pytest.fixture
def api(monkeypatch) -> FakeAPI:
    fake = FakeAPI()
    fake.on("GET", f"/devices/{DEV}/state", state({"rootfs": SYSTEM, "initram": AGENT}))
    install(monkeypatch, fake)
    return fake


def posts(api: FakeAPI) -> list[httpx.Request]:
    return [r for r in api.seen if r.method == "POST"]


# ------------------------------------------------------------ rollback_device ---


def test_preview_reads_the_target_and_sends_nothing(api):
    out = j(server.rollback_device("dan-qemu-3", "system"))
    assert out["confirmation_required"] is True and not posts(api)
    assert out["next"]["arguments"] == {"device": DEV, "component": "system", "to_version": "1.9.0", "organization_id": "org-1", "confirm": True}
    assert out["preview"]["from_version"] == "2.0.0" and out["preview"]["slot"] == "b"
    for phrase in ("restarts", "bootloader", "will not be reinstalled", "forced update"):
        assert phrase in out["summary"] or phrase in json.dumps(out), phrase


def test_agent_preview_says_the_agent_restarts(api):
    out = j(server.rollback_device("dan-qemu-3", "agent"))
    assert "The agent restarts now" in json.dumps(out) and "bootloader" not in out["summary"]


def test_confirmed_system_rollback_posts_expect_version_and_reports_accepted(api):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(202, json={"status": "accepted", "component": "rootfs", "from_version": "2.0.0", "to_version": "1.9.0"}))
    out = j(server.rollback_device("dan-qemu-3", " System ", confirm=True))
    assert bodies(api, "POST", "/rollback") == [{"component": "rootfs", "expect_version": "1.9.0"}]
    assert out["status"] == "accepted" and out["from_version"] == "2.0.0" and out["to_version"] == "1.9.0"
    assert out["summary"].startswith("Accepted") and "get_device_state" in out["next"]


def test_agent_maps_to_initram(api):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(202, json={"status": "accepted", "from_version": "0.9.1", "to_version": "0.9.0"}))
    j(server.rollback_device("dan-qemu-3", "agent", confirm=True))
    assert bodies(api, "POST", "/rollback") == [{"component": "initram", "expect_version": "0.9.0"}]


def test_explicit_to_version_is_the_expectation(api):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(202, json={"status": "accepted"}))
    j(server.rollback_device("dan-qemu-3", "system", to_version="1.9.0", confirm=True))
    assert bodies(api, "POST", "/rollback") == [{"component": "rootfs", "expect_version": "1.9.0"}]


def test_to_version_differing_from_reported_target_warns_in_the_preview(api):
    out = j(server.rollback_device("dan-qemu-3", "system", to_version="1.8.0"))
    assert any("1.9.0" in w for w in out["warnings"]) and out["next"]["arguments"]["to_version"] == "1.8.0"


def test_refusal_returns_reason_and_detail(api):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(409, json={"status": "refused", "reason": "trial_pending", "detail": "running system not yet confirmed"}))
    out = j(server.rollback_device("dan-qemu-3", "system", confirm=True))
    assert out["status"] == "refused" and out["reason"] == "trial_pending" and out["detail"] == "running system not yet confirmed"
    assert out["nothing_changed"] is True and "trial_pending" in out["summary"]


def test_unavailable_target_never_posts(api):
    api.on("GET", f"/devices/{DEV}/state", state({"rootfs": {"available": False, "reason": "no_target", "detail": "only one slot is populated"}}))
    out = j(server.rollback_device("dan-qemu-3", "system", confirm=True))
    assert out["status"] == "unavailable" and out["reason"] == "no_target" and "only one slot" in out["detail"]
    assert not posts(api)


@pytest.mark.parametrize("rollback", [None, {}])
def test_agent_without_rollback_reporting_is_unsupported(api, rollback):
    api.on("GET", f"/devices/{DEV}/state", state(rollback))
    out = j(server.rollback_device("dan-qemu-3", "agent", confirm=True))
    assert out["reason"] == "unsupported" and not posts(api)


@pytest.mark.parametrize("status,needle", [(503, "offline"), (504, "did not reply")])
def test_offline_and_timeout_are_explained(api, status, needle):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(status, json={"msg": "x"}))
    out = j(server.rollback_device("dan-qemu-3", "system", confirm=True))
    assert needle in out["hint"] and "get_device_state" in out["next"]


def test_other_api_errors_use_the_standard_error_shape(api):
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(403, json={"msg": "no"}))
    out = j(server.rollback_device("dan-qemu-3", "system", confirm=True))
    assert out["permission_denied"] is True
    api.on("POST", f"/devices/{DEV}/rollback", httpx.Response(409, json={"msg": "busy"}))
    assert j(server.rollback_device("dan-qemu-3", "system", confirm=True))["conflict"] is True


def test_bad_component_is_rejected_before_any_request(api):
    out = j(server.rollback_device("dan-qemu-3", "rootfs", confirm=True))
    assert "'system'" in out["error"] and not api.seen


def test_ambiguous_device_sends_nothing(api, monkeypatch):
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: {"match": None, "candidates": [{"id": DEV}, {"id": "x"}], "organization_id": "org-1"})
    out = j(server.rollback_device("a", "system", confirm=True))
    assert out["candidates"] and not api.seen


# ------------------------------------------------------------- state summary ---


def test_state_summary_exposes_rollback_availability():
    out = summarise_state({"boot": {"rollback": {"rootfs": SYSTEM, "initram": {"available": False, "reason": "unsupported", "detail": "old agent"}}}})
    assert out["rollback"] == {
        "system": {"available": True, "from_version": "2.0.0", "to_version": "1.9.0", "slot": "b"},
        "agent": {"available": False, "reason": "unsupported", "detail": "old agent"},
    }


def test_state_summary_without_rollback_is_none():
    assert summarise_state({"boot": {}})["rollback"] is None
    assert summarise_state(None)["rollback"] is None
    assert summarise_rollback({"unknown": 1}) is None


def test_get_device_state_tool_carries_the_summary(api):
    out = j(server.get_device_state("dan-qemu-3"))
    assert out["summary"]["rollback"]["system"]["to_version"] == "1.9.0"


# ------------------------------------------------------------ rollback rollouts ---


def rollout_api(api: FakeAPI) -> None:
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", "/rollouts", page([]))
    api.on("POST", "/rollouts/impact", ok({"total_devices": 4, "online_count": 3, "is_disruptive": True}))
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))


def test_create_rollback_rollout_body(api):
    rollout_api(api)
    j(server.create_rollout(FLEET, type="rollback", component="system", to_version="1.9.0", confirm=True))
    j(server.create_rollout(FLEET, type="rollback", component="agent", confirm=True))
    posted = bodies(api, "POST", "/rollouts")
    assert posted[0] == {"name": f"rollback → {FLEET}", "type": "rollback", "fleet_ids": [FLEET], "rollback_spec": {"component": "rootfs", "to_version": "1.9.0"}}
    assert posted[1]["rollback_spec"] == {"component": "initram"}


def test_create_rollback_rollout_preview_keeps_arguments(api):
    rollout_api(api)
    out = j(server.create_rollout(FLEET, type="rollback", component="agent", to_version="0.9.0"))
    assert out["confirmation_required"] is True and not bodies(api, "POST", "/rollouts")
    args = out["next"]["arguments"]
    assert args["type"] == "rollback" and args["component"] == "agent" and args["to_version"] == "0.9.0" and args["confirm"] is True
    assert out["preview"]["request"]["rollback_spec"] == {"component": "initram", "to_version": "0.9.0"}


def test_rollback_rollout_requires_a_valid_component(api):
    rollout_api(api)
    assert "component must be" in j(server.create_rollout(FLEET, type="rollback", confirm=True))["error"]
    assert "component must be" in j(server.create_rollout(FLEET, type="rollback", component="kernel", confirm=True))["error"]
    assert "component must be" in j(server.preview_rollout(FLEET, type="rollback"))["error"]
    assert not bodies(api, "POST", "/rollouts")


def test_preview_rollback_rollout(api):
    rollout_api(api)
    out = j(server.preview_rollout(FLEET, type="rollback", component="system", to_version=" 1.9.0 "))
    assert out["create_rollout"]["arguments"] == {"fleet": FLEET, "type": "rollback", "component": "system", "to_version": "1.9.0"}
    assert json.loads(next(r for r in api.seen if r.url.path.endswith("/impact")).content)["type"] == "rollback"
    assert any("not reinstalled" in w for w in out["warnings"]) and any("disruptive" in w for w in out["warnings"])
    assert j(server.preview_rollout(FLEET, type="rollback", component="agent"))["create_rollout"]["arguments"] == {"fleet": FLEET, "type": "rollback", "component": "agent"}


def test_list_rollouts_accepts_the_rollback_type(api):
    api.on("GET", "/rollouts", page([{"id": ROLLOUT, "name": "r", "type": "rollback", "status": "completed", "progress_pct": 100, "fleet_ids": [FLEET], "stats": {"total": 1}}]))
    out = j(server.list_rollouts(type="rollback"))
    assert len(out["rollouts"]) == 1 and "type rollback" in out["summary"]
