"""Two-way compose tools: export_configuration_compose and import_configuration_compose, plus the readable
409 refusals (host access, agent support, stale base version) that the assignment, rollout and edit tools surface.

The API is faked with httpx.MockTransport (tests/test_w1_tools.FakeAPI); nothing here touches a network.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
import pytest
from test_confirmation import world
from test_ops_tools import CONFIG, FLEET, FLEET2, cfg, details, fleet_row, ok, page
from test_w1_tools import FakeAPI, install

from admrl_mcp import ops, refusals, server

COMPOSE = "name: kiosk\nservices:\n  app:\n    image: registry.example.com/kiosk:2.0\n"
SAVE_PATH = f"/configurations/{CONFIG}:importCompose"
EXPORT_PATH = f"/configurations/{CONFIG}:exportCompose"


def update_answer(**over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "spec": {"version": 2, "services": {"app": {"image": "registry.example.com/kiosk:2.0"}}},
        "diff": [
            {"path": "services.app.image", "from": "registry.example.com/kiosk:1.0", "to": "registry.example.com/kiosk:2.0"},
            {"path": "services.app.environment.API_TOKEN", "from": "old-secret", "to": "new-secret"},
            {"path": "services.app.environment.MODE", "from": None, "to": "prod"},
        ],
        "warnings": [{"path": "services.app.restart", "message": "unless-stopped maps to always"}],
        "errors": [],
        "carriedOver": [{"path": "x-admrl.edgewire", "from": "version 7"}],
        "baseVersion": 7,
        "currentVersion": 7,
        "unchanged": False,
        "version": None,
    }
    out.update(over)
    return out


def refusal_response(status: int, code: str, message: str, **data: Any) -> httpx.Response:
    return httpx.Response(status, json={"success": False, "code": status, "msg": message, "data": {"error": code, "message": message, **data}})


@pytest.fixture
def api(monkeypatch) -> FakeAPI:
    fake = FakeAPI()
    fake.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    fake.on("GET", f"/configurations/{CONFIG}/details", details())
    fake.on("GET", "/configurations", page([cfg()]))
    fake.on("GET", "/fleets", page([fleet_row(), fleet_row(FLEET2, "lab", cid=None)]))
    install(monkeypatch, fake)
    return fake


def posts(api: FakeAPI) -> list[httpx.Request]:
    return [r for r in api.seen if r.method != "GET"]


def call(fn: Any, **kwargs: Any) -> dict[str, Any]:
    return json.loads(fn(**kwargs))


# ---------------------------------------------------------------- registry ---


def test_tools_are_registered_with_the_right_hints_and_schema():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    export, imp = tools["export_configuration_compose"], tools["import_configuration_compose"]
    assert export.annotations and export.annotations.readOnlyHint is True
    assert imp.annotations and imp.annotations.readOnlyHint is False and imp.annotations.destructiveHint is True
    props = imp.inputSchema["properties"]
    assert props["dry_run"]["default"] is True and props["confirm"]["default"] is False
    assert list(props)[-1] == "confirm" and imp.inputSchema["required"] == ["compose"]
    assert {"configuration", "base_version", "profiles", "change_reason"} <= set(props)
    assert export.inputSchema["required"] == ["configuration"] and "version" in export.inputSchema["properties"]


def test_descriptions_tell_the_caller_what_matters_and_stay_public():
    tools = {t.name: t.description or "" for t in asyncio.run(server.mcp.list_tools())}
    text = tools["import_configuration_compose"]
    for needle in ("x-admrl", "dry_run", "confirm=true", "NEW version", "409", "base_version_conflict", "256 KiB", "host_access_not_allowed", "unsupported_by_agent"):
        assert needle in text, needle
    assert "x-admrl" in tools["export_configuration_compose"]
    for name in ("import_configuration_compose", "export_configuration_compose"):
        assert not re.search(r"admin|staff|internal", tools[name], re.IGNORECASE), name
    hints = json.dumps([refusals.describe(b) for b in (HOST_BODY, UNSUPPORTED_BODY, CONFLICT_BODY)])
    assert not re.search(r"admin|staff|internal", hints, re.IGNORECASE)


# ---------------------------------------------------------------- refusals ---

HOST_BODY = {
    "code": 409,
    "msg": "configuration kiosk v8 uses host access that fleet shop does not allow: service frigate: privileged",
    "data": {
        "error": "host_access_not_allowed",
        "message": "configuration kiosk v8 uses host access ...",
        "fleetId": FLEET,
        "fleetName": "shop",
        "uses": [{"service": "frigate", "kind": "privileged"}, {"service": "frigate", "kind": "device", "detail": "/dev/apex_0"}, {"kind": "namespace", "detail": "pid"}],
        "hint": "allow it in the fleet security policy (workload_host_access)",
    },
}
UNSUPPORTED_BODY = {
    "code": 409,
    "msg": "2 device(s) run an agent without the features this set needs (health)",
    "data": {
        "error": "unsupported_by_agent",
        "count": 14,
        "required_protocol": 2,
        "devices": [{"id": f"d{i}", "name": f"screen-{i}", "agent_version": "v1.0.0", "protocol": 2, "missingFeatures": ["health"]} for i in range(12)],
        "blockers": ["services.db.healthcheck"],
        "missingFeatures": ["health"],
    },
}
CONFLICT_BODY = {
    "code": 409,
    "msg": "base_version_conflict: the change is based on version 7 but the configuration is at version 9",
    "data": {"error": "base_version_conflict", "baseVersion": 7, "currentVersion": 9, "updatedBy": "ada@example.com", "updatedAt": "2026-11-04T01:02:03Z"},
}


def described(body: Any) -> dict[str, Any]:
    out = refusals.describe(body)
    assert out is not None
    return out


def test_error_code_prefers_data_error_and_ignores_unknown():
    assert refusals.error_code(HOST_BODY) == "host_access_not_allowed"
    assert refusals.error_code({"error": "base_version_required"}) == "base_version_required"
    assert refusals.error_code({"code": 409, "data": {"error": "something_else"}}) is None
    assert refusals.error_code(None) is None and refusals.describe("nope") is None


def test_host_access_refusal_lists_the_fleet_and_each_use():
    out = described(HOST_BODY)
    assert out["refusal"] == "host_access_not_allowed" and out["fleet"] == {"id": FLEET, "name": "shop"}
    assert out["uses"] == ["service frigate: privileged", "service frigate: device /dev/apex_0", "set level: namespace pid"]
    assert "workload_host_access" in out["hint"] and "Nothing was changed" in out["hint"]


def test_unsupported_by_agent_refusal_caps_the_device_list_and_names_the_features():
    out = described(UNSUPPORTED_BODY)
    assert out["devices_refusing"] == 14 and len(out["devices"]) == 10 and out["devices_omitted"] == 4
    assert out["devices"][0] == {"id": "d0", "name": "screen-0", "agent_version": "v1.0.0", "protocol": 2, "missing_features": ["health"]}
    assert out["missing_features"] == ["health"] and out["blockers"] == ["services.db.healthcheck"] and out["required_protocol"] == 2
    assert "move_device_to_fleet" in out["hint"]


def test_base_version_conflict_refusal_says_how_to_retry():
    out = described(CONFLICT_BODY)
    assert out["base_version"] == 7 and out["current_version"] == 9 and out["updated_by"] == "ada@example.com"
    assert "export_configuration_compose" in out["hint"] and "base_version=9" in out["hint"] and "dry_run=true" in out["hint"]


def test_assign_surfaces_a_host_access_refusal(monkeypatch):
    fake = world()
    install(monkeypatch, fake)
    fake.on("POST", f"/fleets/{FLEET2}/configuration", httpx.Response(409, json=HOST_BODY))
    out = call(server.assign_fleet_configuration, fleet=FLEET2, configuration=CONFIG, version=3, confirm=True)
    assert out["conflict"] is True and out["refusal"] == "host_access_not_allowed"
    assert out["uses"][0] == "service frigate: privileged" and out["fleet"]["name"] == "shop"
    assert "host access" in out["error"]


def test_create_rollout_surfaces_an_unsupported_by_agent_refusal(monkeypatch):
    fake = world()
    install(monkeypatch, fake)
    fake.on("POST", "/rollouts", httpx.Response(409, json=UNSUPPORTED_BODY))
    out = call(server.create_rollout, fleet=FLEET, configuration=CONFIG, version=6, confirm=True)
    assert out["refusal"] == "unsupported_by_agent" and out["devices_refusing"] == 14 and out["missing_features"] == ["health"]


def test_plain_409s_keep_the_generic_hint(monkeypatch):
    fake = world()
    install(monkeypatch, fake)
    fake.on("POST", f"/fleets/{FLEET2}/configuration", httpx.Response(409, json={"msg": "stale"}))
    out = call(server.assign_fleet_configuration, fleet=FLEET2, configuration=CONFIG, version=3, confirm=True)
    assert out["conflict"] is True and "refusal" not in out and "Re-read it" in out["hint"]


# ------------------------------------------------------------------ export ---


def test_export_returns_the_yaml_warnings_and_the_next_call(api):
    yaml = "name: kiosk\nx-admrl:\n  version: 1\nservices:\n  app:\n    image: registry.example.com/kiosk:1.0\n"
    api.on("GET", EXPORT_PATH, ok({"compose": yaml, "warnings": [{"path": "services.app.environment.API_TOKEN", "message": "looks like a credential: move it to a secret file"}], "version": 7}))
    out = call(server.export_configuration_compose, configuration=CONFIG)
    export = next(r for r in api.seen if r.url.path.endswith(":exportCompose"))
    assert export.method == "GET" and export.url.params.get("version") is None
    assert export.headers["Accept"] == "application/json" and export.headers["X-Organization-ID"] == "org-1"
    assert out["compose"] == yaml and out["version"] == 7 and out["configuration"] == {"id": CONFIG, "name": "kiosk", "latest_version": 7}
    assert out["warnings"] == ["services.app.environment.API_TOKEN: looks like a credential: move it to a secret file"]
    assert any("x-admrl" in n for n in out["notes"])
    nxt = out["next"][0]
    assert nxt["tool"] == "import_configuration_compose" and nxt["arguments"]["base_version"] == 7 and nxt["arguments"]["dry_run"] is True
    assert not posts(api)


def test_export_of_an_older_version_sends_version_and_explains_the_base(api):
    api.on("GET", EXPORT_PATH, ok({"compose": COMPOSE, "warnings": [], "version": 5}))
    out = call(server.export_configuration_compose, configuration="kiosk", version=5)
    sent = next(r for r in api.seen if r.url.path.endswith(":exportCompose"))
    assert sent.url.params["version"] == "5"
    assert out["version"] == 5 and out["next"][0]["arguments"]["base_version"] == 7
    assert any("not the latest (v7)" in n for n in out["notes"])


def test_export_rejects_a_version_beyond_latest_without_asking_the_api(api):
    out = call(server.export_configuration_compose, configuration=CONFIG, version=9)
    assert "between 1 and 7" in out["error"]
    assert not [r for r in api.seen if r.url.path.endswith(":exportCompose")]


def test_export_404_and_unsupported_are_readable(api):
    api.on("GET", EXPORT_PATH, httpx.Response(404, json={"msg": "Configuration version not found"}))
    out = call(server.export_configuration_compose, configuration=CONFIG, version=3)
    assert "v3" in out["error"] and out["latest_version"] == 7
    api.on("GET", EXPORT_PATH, refusal_response(409, "compose_export_unsupported", "cannot export", issues=[{"path": "options.namespaces", "message": "host pid has no compose form"}]))
    out = call(server.export_configuration_compose, configuration=CONFIG)
    assert out["refusal"] == "compose_export_unsupported" and out["issues"] == ["options.namespaces: host pid has no compose form"]


# --------------------------------------------------------- update: dry run ---


def test_dry_run_is_the_default_and_sends_dryrun_true(api):
    api.on("POST", SAVE_PATH, ok(update_answer()))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7)
    (sent,) = posts(api)
    assert sent.url.path == f"/v1{SAVE_PATH}" and sent.url.params["dryRun"] == "true" and "baseVersion" not in sent.url.params
    assert sent.headers["Content-Type"] == "application/json" and sent.headers["X-Organization-ID"] == "org-1"
    assert json.loads(sent.content) == {"compose": COMPOSE, "baseVersion": 7}
    assert out["confirmation_required"] is True and out["dry_run"] is True
    assert out["summary"].startswith("Not done yet: save kiosk v8 (from v7) from compose")
    paths = [r["path"] for r in out["preview"]["diff"]]
    assert paths == ["services.app.image", "services.app.environment.API_TOKEN", "services.app.environment.MODE"]
    assert out["preview"]["diff"][1]["old"] == ops.MASK and out["preview"]["diff"][1]["new"] == ops.MASK
    assert out["preview"]["diff"][2] == {"path": "services.app.environment.MODE", "change": "added", "new": "prod"}
    assert out["preview"]["carried_over"] == [{"path": "x-admrl.edgewire", "from": "version 7"}]
    assert any("unless-stopped" in w for w in out["warnings"]) and any("kept from v7" in w for w in out["warnings"])
    assert any("Fleets following 'latest'" in w for w in out["warnings"])
    assert out["preview"]["fleets"]["follow_latest"][0]["name"] == "shop"
    nxt = out["next"]
    assert nxt["tool"] == "import_configuration_compose"
    assert nxt["arguments"] == {"configuration": CONFIG, "compose": COMPOSE, "base_version": 7, "dry_run": False, "organization_id": "org-1", "confirm": True}


def test_dry_run_does_not_save_even_with_confirm(api):
    api.on("POST", SAVE_PATH, ok(update_answer()))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, confirm=True)
    (sent,) = posts(api)
    assert sent.url.params["dryRun"] == "true" and out["confirmation_required"] is True


def test_base_version_defaults_to_what_the_api_resolves_and_is_pinned_in_next(api):
    api.on("POST", SAVE_PATH, ok(update_answer(baseVersion=7)))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG)
    assert "baseVersion" not in json.loads(posts(api)[0].content)
    assert out["next"]["arguments"]["base_version"] == 7


def test_compose_errors_are_refused_with_paths_and_nothing_is_saved(api):
    api.on("POST", SAVE_PATH, ok(update_answer(spec=None, diff=[], errors=[{"path": "services.web.build", "message": "build is not supported: push the image to a registry"}])))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["errors"] == ["services.web.build: build is not supported: push the image to a registry"]
    assert "nothing was saved" in out["error"] and "confirmation_required" not in out
    assert len(posts(api)) == 1


def test_unchanged_file_reports_no_change(api):
    api.on("POST", SAVE_PATH, ok(update_answer(diff=[], unchanged=True, carriedOver=[])))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7)
    assert out["unchanged"] is True and out["saved"] is False and out["summary"].startswith("No change")
    assert "next" not in out


def test_stale_base_version_conflict_says_how_to_retry(api):
    api.on("POST", SAVE_PATH, httpx.Response(409, json=CONFLICT_BODY))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7)
    assert out["refusal"] == "base_version_conflict" and out["current_version"] == 9 and out["conflict"] is True
    assert "export_configuration_compose" in out["hint"] and "base_version=9" in out["hint"]


def test_missing_base_version_is_explained(api):
    api.on("POST", SAVE_PATH, refusal_response(400, "base_version_required", "base_version_required: send ?baseVersion=N"))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG)
    assert out["refusal"] == "base_version_required" and "base_version" in out["hint"]


def test_backend_without_compose_update_is_named(api):
    api.on("POST", SAVE_PATH, httpx.Response(405, json={"msg": "method not allowed"}))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7)
    assert out["compose_update_supported"] is False and "edit_configuration" in out["error"]


def test_dry_run_host_access_warnings_reach_the_caller(api):
    warn = {"path": "services.frigate.privileged", "message": "fleet shop does not allow privileged: saving is refused (host_access_not_allowed); allow it in the fleet security policy"}
    api.on("POST", SAVE_PATH, ok(update_answer(warnings=[warn])))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7)
    assert any("host_access_not_allowed" in w and w.startswith("services.frigate.privileged:") for w in out["warnings"])


# ------------------------------------------------------------ update: save ---


def test_confirmed_save_posts_dryrun_false_and_reads_back(api):
    api.on("POST", SAVE_PATH, ok(update_answer(version=8, currentVersion=8), msg="Configuration updated to version 8"))
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(8)))
    out = call(
        server.import_configuration_compose,
        compose=COMPOSE,
        configuration=CONFIG,
        base_version=7,
        dry_run=False,
        confirm=True,
        profiles=["debug", " "],
        change_reason="bump to 2.0",
    )
    (sent,) = posts(api)
    assert sent.url.params["dryRun"] == "false"
    assert json.loads(sent.content) == {"compose": COMPOSE, "baseVersion": 7, "profiles": ["debug"], "changeReason": "bump to 2.0"}
    assert out["saved"] is True and out["old_version"] == 7 and out["new_version"] == 8
    assert out["read_back"] == {"latest_version": 8, "confirmed": True}
    assert out["configuration"]["latest_version"] == 8
    assert out["diff"][1]["old"] == ops.MASK and "old-secret" not in json.dumps(out) and "new-secret" not in json.dumps(out)
    assert out["carried_over"] == [{"path": "x-admrl.edgewire", "from": "version 7"}]
    assert out["next"][0]["tool"] == "preview_rollout" and "delivery" in out


def test_confirmed_save_flags_a_read_back_mismatch(api):
    api.on("POST", SAVE_PATH, ok(update_answer(version=8, currentVersion=8)))
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(9)))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["read_back"]["confirmed"] is False and any("now at v9" in w for w in out["warnings"])


def test_save_that_finds_no_change_creates_no_version(api):
    api.on("POST", SAVE_PATH, ok(update_answer(diff=[], unchanged=True, version=None)))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["unchanged"] is True and out["saved"] is False


def test_save_refused_for_host_access_names_fleet_and_uses(api):
    api.on("POST", SAVE_PATH, httpx.Response(409, json=HOST_BODY))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["refusal"] == "host_access_not_allowed" and out["fleet"]["name"] == "shop"
    assert "service frigate: device /dev/apex_0" in out["uses"]


def test_save_refused_for_unsupported_agents(api):
    api.on("POST", SAVE_PATH, httpx.Response(409, json=UNSUPPORTED_BODY))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["refusal"] == "unsupported_by_agent" and out["devices_refusing"] == 14


def test_a_403_is_a_permission_message(api):
    api.on("POST", SAVE_PATH, httpx.Response(403, json={"msg": "access denied"}))
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, base_version=7, dry_run=False, confirm=True)
    assert out["permission_denied"] is True


# ----------------------------------------------------------------- create ---


def test_create_dry_run_translates_only_and_previews(api):
    api.on("POST", "/configurations:importCompose", ok({"spec": {"version": 2, "services": {"app": {"image": "registry.example.com/kiosk:2.0", "environment": {"API_TOKEN": "s3cret"}}}}, "name": "kiosk-new", "warnings": [{"path": "services.app.restart", "message": "defaulted"}], "errors": []}))
    out = call(server.import_configuration_compose, compose=COMPOSE, profiles="debug, tools")
    (sent,) = posts(api)
    assert sent.url.path == "/v1/configurations:importCompose" and json.loads(sent.content) == {"compose": COMPOSE, "profiles": ["debug", "tools"]}
    assert out["confirmation_required"] is True and out["dry_run"] is True
    assert out["summary"].startswith("Not done yet: create configuration kiosk-new v1 with image registry.example.com/kiosk:2.0")
    assert "s3cret" not in json.dumps(out)
    assert out["next"]["arguments"]["name"] == "kiosk-new" and out["next"]["arguments"]["dry_run"] is False and out["next"]["arguments"]["profiles"] == ["debug", "tools"]
    assert out["warnings"] == ["services.app.restart: defaulted"]


def test_create_warns_about_an_existing_name(api):
    api.on("POST", "/configurations:importCompose", ok({"spec": {"version": 2, "services": {"app": {"image": "x:1"}}}, "name": "kiosk", "warnings": [], "errors": []}))
    out = call(server.import_configuration_compose, compose=COMPOSE)
    assert any("already exists" in w for w in out["warnings"])


def test_create_needs_a_name_only_when_saving(api):
    api.on("POST", "/configurations:importCompose", ok({"spec": {"version": 2, "services": {"app": {"image": "x:1"}}}, "warnings": [], "errors": []}))
    dry = call(server.import_configuration_compose, compose=COMPOSE)
    assert any("no name" in w for w in dry["warnings"])
    out = call(server.import_configuration_compose, compose=COMPOSE, dry_run=False, confirm=True)
    assert "needs a name" in out["error"] and not [r for r in posts(api) if r.url.path.endswith("/configurations")]


def test_create_save_posts_the_translated_spec_and_reads_back(api):
    built = {"version": 2, "services": {"app": {"image": "registry.example.com/kiosk:2.0"}}}
    api.on("POST", "/configurations:importCompose", ok({"spec": built, "warnings": [], "errors": []}))
    api.on("POST", "/configurations", ok({"id": CONFIG, "name": "kiosk-new", "status": "draft", "latest_version": 1}))
    api.on("GET", f"/configurations/{CONFIG}/details", ok({"configuration": cfg(1, name="kiosk-new", status="draft"), "latest_spec": built}))
    out = call(server.import_configuration_compose, compose=COMPOSE, name="kiosk-new", description="from compose", dry_run=False, confirm=True)
    created = next(r for r in posts(api) if r.url.path == "/v1/configurations")
    assert json.loads(created.content) == {"name": "kiosk-new", "spec": built, "description": "from compose"}
    assert out["saved"] is True and out["configuration"]["name"] == "kiosk-new" and out["read_back"]["confirmed"] is True
    assert out["next"][0]["tool"] == "assign_fleet_configuration"


def test_create_with_compose_errors_creates_nothing(api):
    api.on("POST", "/configurations:importCompose", ok({"spec": None, "warnings": [], "errors": [{"path": "services.web.build", "message": "build is not supported"}]}))
    out = call(server.import_configuration_compose, compose=COMPOSE, name="x", dry_run=False, confirm=True)
    assert out["errors"] == ["services.web.build: build is not supported"]
    assert len(posts(api)) == 1


# ------------------------------------------------------------- arguments ---


def test_argument_mixups_are_refused_before_any_request(api):
    out = call(server.import_configuration_compose, compose=COMPOSE, base_version=7)
    assert "only apply when updating" in out["error"]
    out = call(server.import_configuration_compose, compose=COMPOSE, configuration=CONFIG, name="other")
    assert "creating a new configuration" in out["error"]
    assert not api.seen


def test_empty_and_oversized_compose_are_refused_locally(api):
    assert call(server.import_configuration_compose, compose="  \n")["error"].endswith("compose is empty.")
    big = "x: " + "a" * (256 * 1024)
    assert "larger than 256 KiB" in call(server.import_configuration_compose, compose=big, configuration=CONFIG)["error"]
    assert not api.seen
