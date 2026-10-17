"""Backend permission gates: configuration/rollout reads 403 and contents are withheld without configs.view."""

from __future__ import annotations

import httpx
import pytest

from admrl_mcp import server

from test_ops_tools import CONFIG, DEV, FLEET, ROLLOUT, api, bodies, calls, dev_cfg, fleet_row, j, ok  # noqa: F401

DENIED = httpx.Response(403, json={"code": 403, "msg": "forbidden"})
VIEW = "View configurations"


def withheld_dev_cfg(has_override: bool = True) -> httpx.Response:
    return ok({"host_id": DEV, "fleet_id": FLEET, "configuration_id": CONFIG, "config_version": 7, "is_pinned": False,
               "has_override": has_override, "overridden_paths": [], "host_version": 1})


@pytest.mark.parametrize("call,path", [
    (lambda: server.get_rollout(ROLLOUT), f"/rollouts/{ROLLOUT}"),
    (lambda: server.list_rollouts(), "/rollouts"),
    (lambda: server.list_configurations(include_fleets=False), "/configurations"),
])
def test_403_on_config_and_rollout_reads_is_a_permission_message(api, call, path):
    api.on("GET", path, DENIED)
    out = j(call())
    assert out["permission_denied"] is True and VIEW in out["hint"]


def test_empty_lists_note_possible_permission(api):
    api.on("GET", "/configurations", ok([]))
    api.on("GET", "/rollouts", ok([]))
    assert VIEW in j(server.list_configurations(include_fleets=False))["note"]
    assert "permission" in j(server.list_rollouts())["note"]


def test_withheld_device_configuration_is_not_reported_as_no_override(api):
    api.on("GET", f"/devices/{DEV}/configuration", withheld_dev_cfg())
    out = j(server.get_device_configuration("x"))
    assert "override (content withheld)" in out["summary"] and VIEW in out["details_withheld"]
    assert out["has_override"] is True
    assert not {"override", "merged", "overridden_fields", "overridden_paths"} & set(out)
    api.on("GET", f"/devices/{DEV}/configuration", withheld_dev_cfg(False))
    assert "no override" in j(server.get_device_configuration("x"))["summary"]


def test_visible_device_configuration_has_no_withheld_marker(api):
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg({"image": "a"}))
    assert "details_withheld" not in j(server.get_device_configuration("x"))


def test_override_edits_are_refused_when_override_withheld(api):
    api.on("GET", f"/devices/{DEV}/configuration", withheld_dev_cfg())
    for out in (
        j(server.set_device_configuration_override("x", {"image": "a"}, confirm=True)),
        j(server.reset_device_configuration_override("x", ["image"], confirm=True)),
    ):
        assert out["error"].startswith("Refused") and VIEW in out["error"]
    assert not [r for r in api.seen if r.method != "GET"]


def test_patch_document_refuses_override_edit_when_withheld_but_allows_others(api):
    api.on("GET", f"/devices/{DEV}/configuration", withheld_dev_cfg())
    out = j(server.patch_device_document("x", {"spec": {"workload": {"override": {"image": "a"}}}}, confirm=True))
    assert out["error"].startswith("Refused")
    api.on("GET", f"/devices/{DEV}/configuration", DENIED)
    assert j(server.patch_device_document("x", {"spec": {"workload": None}}, confirm=True))["error"].startswith("Refused")
    assert not [r for r in api.seen if r.method == "PATCH"]
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"spec": {}}, headers={"ETag": '"v1"'}))
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(200, json={"spec": {}}, headers={"ETag": '"v2"'}))
    j(server.patch_device_document("x", {"spec": {"system": {"timezone": "UTC"}}}, confirm=True))
    assert len(bodies(api, "PATCH", "/document")) == 1


def test_withheld_fleet_configuration_gives_no_fake_diff(api):
    api.on("GET", f"/fleets/{FLEET}/configuration", ok({"fleet_id": FLEET, "configuration_id": CONFIG, "config_version": 7, "is_pinned": False}))
    assignment, spec = server._fleet_assignment(server.get_client(), "org-1", FLEET)
    assert spec is None and VIEW in assignment["details_withheld"]
    assert server._diff_or_none(spec, {"image": "x"}) is None


def test_safe_reports_403_as_permission(api):
    warnings: list[str] = []
    api.on("GET", "/rollouts", DENIED)
    server._safe(lambda: server._active_rollouts(server.get_client(), "org-1", [FLEET]), warnings, "rollouts")
    assert "not permitted (403)" in warnings[0] and VIEW in warnings[0]


def test_workload_withheld_note(api):
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "running", "running": None, "latestReport": {"configuration": None}}))
    assert VIEW in j(server.get_device_workload("x"))["note"]
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "running", "running": {"image": "a"}}))
    assert "note" not in j(server.get_device_workload("x"))


def test_render_redacts_secret_content_and_notes_missing_workload(api):
    bundle = {"secrets": [{"path": "/s", "content": "W3JlZGFjdGVkXQ=="}], "workload": None}
    api.on("POST", f"/devices/{DEV}/document:render", ok({"bundle": bundle, "diff": []}))
    out = j(server.render_device_document("x"))
    assert out["render"]["bundle"]["secrets"][0]["content"] == "[redacted]" and "[redacted]" in server._dumps(out)
    assert "W3Jl" not in server._dumps(out) and VIEW in out["note"]
    bundle = {"workload": {"secretFiles": [{"path": "/w", "content": "W3JlZGFjdGVkXQ=="}]}}
    api.on("POST", f"/devices/{DEV}/document:render", ok({"bundle": bundle}))
    out = j(server.render_device_document("x"))
    assert out["render"]["bundle"]["workload"]["secretFiles"][0]["content"] == "[redacted]" and "note" not in out


def test_document_note_when_override_absent(api):
    doc = {"spec": {"workload": {"from": {"configuration": CONFIG}}}}
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json=doc, headers={"ETag": '"v1"'}))
    assert VIEW in j(server.get_device_document("x"))["note"]
