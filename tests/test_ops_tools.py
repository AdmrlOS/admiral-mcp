"""Operations tools: configuration editing, fleet assignment, fleet/device management, rollout planning.

Every test fakes the HTTP API (httpx.MockTransport via test_w1_tools.FakeAPI) and asserts method, path,
body and the read-back that follows each write. Ids are synthetic.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from admrl_mcp import server

from test_w1_tools import CONFIG, DEV, FLEET, FakeAPI, assert_pat, envelope, install

FLEET2 = "22222222-3333-4444-5555-666666666666"
CONFIG2 = "bbbbbbbb-cccc-dddd-eeee-ffffffffffff"
ROLLOUT = "99999999-8888-7777-6666-555555555555"


class Seq:
    """Route handler answering successive calls with successive responses (the last one repeats)."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp


def ok(data: Any, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json=envelope(data, **extra))


def page(rows: list[dict[str, Any]], total_pages: int = 1) -> httpx.Response:
    return httpx.Response(
        200,
        json={"code": 200, "msg": "Success", "data": rows, "pagination": {"page": 1, "limit": 200, "total": len(rows), "totalPages": total_pages}},
    )


def spec(**over: Any) -> dict[str, Any]:
    out = {
        "image": "registry.example.com/kiosk:1.0",
        "desiredState": "RUNNING",
        "command": [],
        "environment": {"MODE": "prod", "API_TOKEN": "s3cret"},
        "mounts": [],
        "options": {},
        "volumes": {},
        "ports": [{"port": 80, "protocol": "tcp"}],
    }
    out.update(over)
    return out


def cfg(latest: int = 7, **over: Any) -> dict[str, Any]:
    out = {"id": CONFIG, "name": "kiosk", "status": "active", "latest_version": latest, "updated_at": "2026-01-01T00:00:00Z"}
    out.update(over)
    return out


def details(latest: int = 7, the_spec: dict[str, Any] | None = None, **over: Any) -> httpx.Response:
    return ok(
        {
            "configuration": cfg(latest, **over),
            "latest_spec": the_spec or spec(),
            "version_info": {"version_number": latest, "changed_by": "u1", "change_reason": "init", "created_at": "2026-01-01T00:00:00Z"},
        }
    )


def fleet_row(fid: str = FLEET, name: str = "shop", cid: str | None = CONFIG, version: int | None = None, **over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": fid, "name": name, "devices": 4, "online": 3, "offline": 1, "tags": [{"key": "env", "value": "prod"}]}
    if cid:
        out["configuration_id"] = cid
    if version:
        out["config_version"] = version
    out.update(over)
    return out


def fleet_cfg(cid: str | None = CONFIG, name: str = "kiosk", version: int = 7, pinned: bool = False, the_spec: dict[str, Any] | None = None) -> httpx.Response:
    if not cid:
        return ok({"fleet_id": FLEET, "updated_at": "x"})
    return ok(
        {
            "fleet_id": FLEET,
            "configuration_id": cid,
            "config_version": version,
            "is_pinned": pinned,
            "configuration": {"id": cid, "name": name, "status": "active", "latest_version": 7},
            "configuration_spec": the_spec or spec(),
            "updated_at": "x",
        }
    )


def j(text: str) -> dict[str, Any]:
    return json.loads(text)


def bodies(api: FakeAPI, method: str, suffix: str) -> list[Any]:
    return [json.loads(r.content) for r in api.seen if r.method == method and r.url.path.endswith(suffix)]


def calls(api: FakeAPI) -> list[str]:
    return [f"{r.method} {r.url.path.removeprefix('/v1')}" for r in api.seen]


@pytest.fixture
def api(monkeypatch) -> FakeAPI:
    fake = FakeAPI()
    install(monkeypatch, fake)
    return fake


# ======================================================== configurations ===


def test_list_configurations_rows_and_fleet_usage(api):
    api.on("GET", "/configurations", page([cfg(7, description="Kiosk app", tags=["a"]), cfg(2, id=CONFIG2, name="signage", status="draft")]))
    api.on("GET", "/fleets", page([fleet_row(), fleet_row(FLEET2, "lab", CONFIG, 5)]))

    out = j(server.list_configurations(search="k", status="ACTIVE", limit=500))

    params = dict(api.seen[0].url.params)
    assert params["status"] == "active" and params["search"] == "k" and params["limit"] == "200"
    assert_pat(api.seen[0])
    assert out["summary"].startswith("2 of 2 configuration(s)")
    kiosk = out["configurations"][0]
    assert kiosk["fleets"] == [{"id": FLEET, "name": "shop", "version": "latest"}, {"id": FLEET2, "name": "lab", "version": 5}]
    assert out["configurations"][1]["fleets"] == []
    assert out["next"][0]["tool"] == "get_configuration"


def test_list_configurations_without_fleets_skips_the_fleet_call(api):
    api.on("GET", "/configurations", page([cfg()]))
    out = j(server.list_configurations(include_fleets=False))
    assert [r.url.path for r in api.seen] == ["/v1/configurations"] and "fleets" not in out["configurations"][0]


def test_list_configurations_rejects_unknown_status(api):
    out = j(server.list_configurations(status="live"))
    assert out["choices"] == ["draft", "active", "testing", "deprecated", "archived"] and api.seen == []


def test_get_configuration_latest_masks_secrets_and_lists_fleets(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on("GET", f"/configurations/{CONFIG}/details", details())
    api.on(
        "GET",
        f"/configurations/{CONFIG}/versions",
        ok([{"version_number": n, "changed_by": "u", "change_reason": f"r{n}", "created_at": f"2026-01-0{n}", "is_rollback": n == 3, "rolled_back_to": 1} for n in range(1, 8)]),
    )
    api.on("GET", "/fleets", page([fleet_row(), fleet_row(FLEET2, "lab", CONFIG, 5)]))

    out = j(server.get_configuration(CONFIG, history_limit=3))

    assert out["spec"]["environment"] == {"MODE": "prod", "API_TOKEN": "***"} and "masked" in out["note"]
    assert out["configuration"]["latest_version"] == 7 and out["is_latest"] is True
    assert [h["version"] for h in out["history"]] == [7, 6, 5]
    assert [f["name"] for f in out["fleets"]["follow_latest"]] == ["shop"]
    assert out["fleets"]["pinned"][0]["pinned_version"] == 5
    assert "1 fleet(s) follow latest, 1 pinned" in out["summary"]


def test_get_configuration_reveals_secrets_only_on_request(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on("GET", f"/configurations/{CONFIG}/details", details())
    api.on("GET", f"/configurations/{CONFIG}/versions", ok([]))
    api.on("GET", "/fleets", page([]))
    out = j(server.get_configuration(CONFIG, show_secret_env=True))
    assert out["spec"]["environment"]["API_TOKEN"] == "s3cret" and "note" not in out


def test_get_configuration_specific_version_and_spec_summary(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on(
        "GET",
        f"/configurations/{CONFIG}/versions/3",
        ok({"version_number": 3, "spec": spec(image="old:1"), "changed_by": "u", "change_reason": "x", "created_at": "t"}),
    )
    api.on("GET", f"/configurations/{CONFIG}/versions", ok([]))
    api.on("GET", "/fleets", page([]))

    out = j(server.get_configuration(CONFIG, version=3, include_spec=False))

    assert out["version"] == 3 and out["is_latest"] is False
    assert out["spec_summary"]["image"] == "old:1" and "spec" not in out


def test_get_configuration_partial_failure_becomes_a_warning(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on("GET", f"/configurations/{CONFIG}/details", details())
    api.on("GET", f"/configurations/{CONFIG}/versions", httpx.Response(500, json={"msg": "boom"}))
    api.on("GET", "/fleets", page([]))
    out = j(server.get_configuration(CONFIG))
    assert out["history"] == [] and "version history" in out["warnings"][0]


def test_get_configuration_by_name_ambiguous_returns_candidates(api):
    api.on("GET", "/configurations", ok([cfg(id=CONFIG, name="kiosk-a"), cfg(id=CONFIG2, name="kiosk-b")]))
    out = j(server.get_configuration("kiosk"))
    assert [c["id"] for c in out["candidates"]] == [CONFIG, CONFIG2] and "UUID" in out["error"]
    assert len(api.seen) == 1


def test_get_configuration_not_found_and_bad_version(api):
    api.on("GET", "/configurations", ok([]))
    out = j(server.get_configuration("ghost"))
    assert out["candidates"] == [] and "No configuration matches" in out["error"]
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    assert "1 or greater" in j(server.get_configuration(CONFIG, version=0))["error"]


def test_diff_configuration_versions_defaults_to_previous_and_latest(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7)))
    api.on("GET", f"/configurations/{CONFIG}/versions/6", ok({"version_number": 6, "spec": spec(), "changed_by": "u", "created_at": "a"}))
    api.on(
        "GET",
        f"/configurations/{CONFIG}/versions/7",
        ok({"version_number": 7, "spec": spec(image="registry.example.com/kiosk:2.0", environment={"MODE": "prod", "API_TOKEN": "new"}), "change_reason": "bump"}),
    )

    out = j(server.diff_configuration_versions(CONFIG))

    assert out["from"]["version"] == 6 and out["to"]["version"] == 7 and out["identical"] is False
    paths = {c["path"]: c for c in out["changes"]}
    assert paths["image"]["new"].endswith(":2.0") and paths["environment.API_TOKEN"]["new"] == "***"
    assert out["summary"].startswith("kiosk v6 → v7:") and "environment.API_TOKEN changed (value masked)" in out["summary"]


def test_diff_configuration_versions_edge_cases(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(1)))
    assert "nothing earlier" in j(server.diff_configuration_versions(CONFIG))["error"]
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(5)))
    assert "same" in j(server.diff_configuration_versions(CONFIG, from_version=4, to_version=4))["error"]
    api.on("GET", f"/configurations/{CONFIG}/versions/2", ok({"version_number": 2, "spec": spec()}))
    api.on("GET", f"/configurations/{CONFIG}/versions/4", ok({"version_number": 4, "spec": spec()}))
    out = j(server.diff_configuration_versions(CONFIG, from_version=2, to_version=4))
    assert out["identical"] is True and out["summary"].endswith("no changes")


def test_create_configuration_happy_path_with_read_back(api):
    api.on("POST", "/configurations", httpx.Response(201, json=envelope({"id": CONFIG, "name": "kiosk", "latest_version": 1, "status": "active"})))
    api.on("GET", f"/configurations/{CONFIG}/details", details(1, spec(image="web:1", environment={"A": "1"})))

    out = j(server.create_configuration(" kiosk ", image="web:1", env={"A": 1}, ports=["8080/udp", 80], command=["x"], description="d", tags=["t"]))

    body = bodies(api, "POST", "/configurations")[0]
    assert body["name"] == "kiosk" and body["description"] == "d" and body["tags"] == ["t"]
    assert body["spec"]["image"] == "web:1" and body["spec"]["environment"] == {"A": "1"} and body["spec"]["command"] == ["x"]
    assert body["spec"]["ports"] == [{"port": 8080, "protocol": "udp"}, {"port": 80, "protocol": "tcp"}]
    assert out["confirmed"] is True and out["configuration"]["latest_version"] == 1
    assert out["next"][0]["tool"] == "assign_fleet_configuration"


def test_create_configuration_validation_and_conflict(api):
    assert "name is required" in j(server.create_configuration(" ", image="x"))["error"]
    assert "needs an image" in j(server.create_configuration("n"))["error"]
    assert "not both" in j(server.create_configuration("n", image="x", spec={"image": "y"}))["error"]
    assert "RUNNING or STOPPED" in j(server.create_configuration("n", image="x", desired_state="PAUSED"))["error"]
    assert api.seen == []
    api.on("POST", "/configurations", httpx.Response(409, json={"msg": "Configuration already exists"}))
    out = j(server.create_configuration("n", spec={"image": "y"}))
    assert out["conflict"] is True and "409" in out["error"]


def edit_api(api: FakeAPI, latest: int = 7, after_spec: dict[str, Any] | None = None, after_latest: int | None = None, fleets: list | None = None) -> None:
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(latest)))
    api.on("GET", f"/configurations/{CONFIG}/details", Seq(details(latest), details(after_latest or latest + 1, after_spec)))
    api.on("GET", "/fleets", page(fleets if fleets is not None else [fleet_row(), fleet_row(FLEET2, "lab", CONFIG, 7)]))
    api.on("PUT", f"/configurations/{CONFIG}/spec", ok({"id": CONFIG, "latest_version": latest + 1}))


def test_edit_configuration_saves_new_version_and_explains_delivery(api):
    new = spec(image="registry.example.com/kiosk:2.0")
    edit_api(api, after_spec=new)

    out = j(server.edit_configuration(CONFIG, change_reason=" bump ", image_tag="2.0", base_version=7, confirm=True))

    put = bodies(api, "PUT", "/spec")[0]
    assert put["change_reason"] == "bump" and put["spec"]["image"].endswith(":2.0")
    assert out["old_version"] == 7 and out["new_version"] == 8 and out["read_back"] == {"latest_version": 8, "confirmed": True}
    assert out["diff"][0]["path"] == "image"
    assert [f["name"] for f in out["fleets"]["follow_latest"]] == ["shop"]
    assert out["fleets"]["pinned"][0]["pinned_version"] == 7
    assert any("without a canary" in w for w in out["warnings"])
    assert "does not push anything" in out["delivery"]
    assert out["next"][0]["tool"] == "preview_rollout"
    assert out["next"][0]["arguments"]["version"] == 8
    assert out["summary"].startswith("kiosk v7 → v8")
    # secrets never leak into the diff, but they were sent unmasked
    assert put["spec"]["environment"]["API_TOKEN"] == "s3cret"


def test_edit_configuration_dry_run_never_writes(api):
    edit_api(api)
    out = j(server.edit_configuration(CONFIG, env_set={"NEW": "1"}, dry_run=True))
    assert out["confirmation_required"] is True and out["summary"].startswith("Not done yet") and out["dry_run"] is True
    assert out["preview"]["diff"][0]["path"] == "environment.NEW"
    assert any("pin those fleets" in w for w in out["warnings"])
    assert any("change_reason is required" in w for w in out["warnings"])
    assert out["next"]["tool"] == "edit_configuration"
    assert out["next"]["arguments"] == {"configuration": CONFIG, "base_version": 7, "env_set": {"NEW": "1"}, "confirm": True, "organization_id": "org-1"}
    assert out["action"]["from_version"] == 7 and out["action"]["to_version"] == 8
    # dry_run wins over confirm=true; without confirm the call is a preview as well
    for kwargs in ({"dry_run": True, "confirm": True}, {}):
        again = j(server.edit_configuration(CONFIG, env_set={"NEW": "1"}, **kwargs))
        assert again["confirmation_required"] is True
    assert not [r for r in api.seen if r.method == "PUT"]


def test_edit_configuration_requires_reason_unless_dry_run(api):
    out = j(server.edit_configuration(CONFIG, image_tag="2", confirm=True))
    assert "change_reason is required" in out["error"] and api.seen == []


def test_edit_configuration_base_version_conflict_refuses(api):
    edit_api(api)
    out = j(server.edit_configuration(CONFIG, change_reason="x", image_tag="2", base_version=6, confirm=True))
    assert out["latest_version"] == 7 and out["base_version"] == 6 and "Nothing was written" in out["error"]
    assert not [r for r in api.seen if r.method == "PUT"]


def test_edit_configuration_refuses_no_op(api):
    edit_api(api)
    out = j(server.edit_configuration(CONFIG, change_reason="x", env_set={"MODE": "prod"}, env_unset=["ghost"], confirm=True))
    assert out["error"].startswith("No change") and out["notes"] == ["env_unset: ghost was not set (ignored)"]
    assert not [r for r in api.seen if r.method == "PUT"]


def test_edit_configuration_bad_edit_arguments_are_errors_not_writes(api):
    edit_api(api)
    out = j(server.edit_configuration(CONFIG, change_reason="x", spec={"image": "a"}, image_tag="2", confirm=True))
    assert "cannot be combined" in out["error"]
    out = j(server.edit_configuration(CONFIG, change_reason="x", image="a:1", image_tag="2", confirm=True))
    assert "not both" in out["error"]
    assert not [r for r in api.seen if r.method == "PUT"]


def test_edit_configuration_full_replacement_and_merge_patch(api):
    edit_api(api, after_spec=spec(image="other:1", environment={}, ports=[]))
    out = j(server.edit_configuration(CONFIG, change_reason="swap", spec={"image": "other:1"}, confirm=True))
    assert bodies(api, "PUT", "/spec")[0]["spec"]["image"] == "other:1"
    assert {c["path"] for c in out["diff"]} >= {"image", "environment.MODE", "ports[tcp/80]"}


def test_edit_configuration_merge_patch_deletes_and_sets(monkeypatch):
    fake = FakeAPI()
    install(monkeypatch, fake)
    edit_api(fake, after_spec=spec(environment={"MODE": "prod"}, options={"privileged": True}))
    out = j(server.edit_configuration(CONFIG, change_reason="p", merge_patch={"environment": {"API_TOKEN": None}, "options": {"privileged": True}}, confirm=True))
    sent = bodies(fake, "PUT", "/spec")[0]["spec"]
    assert sent["environment"] == {"MODE": "prod"} and sent["options"] == {"privileged": True}
    assert {c["path"] for c in out["diff"]} == {"environment.API_TOKEN", "options.privileged"}
    assert out["read_back"]["confirmed"] is True


def test_edit_configuration_402_signature_policy_is_surfaced_as_billing_gate(api):
    edit_api(api)
    api.on("PUT", f"/configurations/{CONFIG}/spec", httpx.Response(402, json={"msg": "signature_policy feature is not enabled"}))
    out = j(server.edit_configuration(CONFIG, change_reason="sign", merge_patch={"signaturePolicy": {"required": True}}, confirm=True))
    assert out["billing_gate"] is True and "402" in out["error"] and "entitlement" in out["hint"]


def test_edit_configuration_detects_concurrent_edit_and_unconfirmed_write(api):
    edit_api(api, after_spec=spec(image="registry.example.com/kiosk:3.0"), after_latest=9)
    out = j(server.edit_configuration(CONFIG, change_reason="x", image_tag="2.0", confirm=True))
    assert out["read_back"]["confirmed"] is False and out["new_version"] == 9
    assert any("another edit landed" in w for w in out["warnings"])
    assert any("differs from what was sent" in w for w in out["warnings"])


def test_edit_configuration_unused_config_suggests_assignment(api):
    edit_api(api, after_spec=spec(image="x:2"), fleets=[])
    out = j(server.edit_configuration(CONFIG, change_reason="x", image="x:2", confirm=True))
    assert out["next"][0]["tool"] == "assign_fleet_configuration" and "delivery" not in out and not out.get("warnings")


def test_edit_configuration_ambiguous_name_changes_nothing(api):
    api.on("GET", "/configurations", ok([cfg(id=CONFIG, name="kiosk-a"), cfg(id=CONFIG2, name="kiosk-b")]))
    out = j(server.edit_configuration("kiosk", change_reason="x", image_tag="2", confirm=True))
    assert len(out["candidates"]) == 2 and calls(api) == ["GET /configurations"]


def test_update_configuration_metadata_name_and_status(api):
    before = ok(cfg(7, description="old", tags=["a"]))
    after = ok(cfg(7, name="kiosk-2", status="testing", description="old", tags=["a"]))
    api.on("GET", f"/configurations/{CONFIG}", Seq(before, before, after))
    api.on("PUT", f"/configurations/{CONFIG}", ok({"id": CONFIG}))
    api.on("PUT", f"/configurations/{CONFIG}/status", ok({"id": CONFIG}))

    out = j(server.update_configuration_metadata(CONFIG, name=" kiosk-2 ", status="Testing"))

    assert bodies(api, "PUT", f"/configurations/{CONFIG}") == [{"name": "kiosk-2"}]
    assert bodies(api, "PUT", "/status") == [{"status": "testing"}]
    assert out["confirmed"] is True and out["before"]["name"] == "kiosk" and out["after"]["status"] == "testing"


def test_update_configuration_metadata_clears_tags_and_validates(api):
    api.on("GET", f"/configurations/{CONFIG}", Seq(ok(cfg(tags=["a"])), ok(cfg(tags=["a"])), ok(cfg())))
    api.on("PUT", f"/configurations/{CONFIG}", ok({}))
    out = j(server.update_configuration_metadata(CONFIG, tags=[]))
    assert bodies(api, "PUT", f"/configurations/{CONFIG}") == [{"tags": []}] and out["confirmed"] is True
    assert "cannot be empty" in j(server.update_configuration_metadata(CONFIG, name=" "))["error"]
    assert "Unknown status" in j(server.update_configuration_metadata(CONFIG, status="live"))["error"]
    assert "Nothing to change" in j(server.update_configuration_metadata(CONFIG))["error"]


def test_update_configuration_metadata_no_op(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(description="d", tags=["a"])))
    out = j(server.update_configuration_metadata(CONFIG, name="kiosk", description="d", tags=["a"], status="active"))
    assert out["error"].startswith("No change") and not [r for r in api.seen if r.method == "PUT"]


def rollback_api(api: FakeAPI, new_spec: dict[str, Any] | None = None) -> None:
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7)))
    api.on("GET", f"/configurations/{CONFIG}/versions/7", ok({"version_number": 7, "spec": spec(image="kiosk:7")}))
    api.on("GET", f"/configurations/{CONFIG}/versions/4", ok({"version_number": 4, "spec": spec(image="kiosk:4")}))
    api.on("GET", f"/configurations/{CONFIG}/details", details(8, new_spec or spec(image="kiosk:4")))
    api.on("GET", "/fleets", page([fleet_row()]))
    api.on("POST", f"/configurations/{CONFIG}/rollback", ok({"id": CONFIG, "latest_version": 8, "rolled_back_to": 4}))


def test_rollback_configuration_creates_new_version_and_reads_back(api):
    rollback_api(api)
    out = j(server.rollback_configuration(CONFIG, 4, reason=" bad build ", confirm=True))
    assert bodies(api, "POST", "/rollback") == [{"target_version": 4, "reason": "bad build"}]
    assert out["old_version"] == 7 and out["new_version"] == 8
    assert out["read_back"] == {"latest_version": 8, "matches_target": True, "confirmed": True}
    assert out["diff"][0]["new"] == "kiosk:4" and "copy of v4" in out["summary"]
    assert any("without a canary" in w for w in out["warnings"])


def test_rollback_configuration_dry_run_and_guards(api):
    rollback_api(api)
    out = j(server.rollback_configuration(CONFIG, 4, dry_run=True))
    assert out["dry_run"] is True and not [r for r in api.seen if r.method == "POST"]
    assert "already the latest" in j(server.rollback_configuration(CONFIG, 7, confirm=True))["error"]
    assert "between 1 and 7" in j(server.rollback_configuration(CONFIG, 9, confirm=True))["error"]
    assert "between 1 and 7" in j(server.rollback_configuration(CONFIG, 0, confirm=True))["error"]


def test_rollback_configuration_flags_identical_specs_and_drift(api):
    rollback_api(api, new_spec=spec(image="something-else"))
    api.on("GET", f"/configurations/{CONFIG}/versions/4", ok({"version_number": 4, "spec": spec(image="kiosk:7")}))
    out = j(server.rollback_configuration(CONFIG, 4, confirm=True))
    assert out["read_back"]["matches_target"] is False and out["read_back"]["confirmed"] is False
    assert any("identical version" in w for w in out["warnings"])


def test_delete_configuration_refuses_while_fleets_or_rollouts_use_it(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on("GET", "/fleets", page([fleet_row(), fleet_row(FLEET2, "lab", CONFIG, 3)]))
    api.on("GET", "/rollouts", page([{"id": ROLLOUT, "name": "r", "type": "config", "status": "in_progress", "config": CONFIG, "fleet_ids": [FLEET]}]))
    out = j(server.delete_configuration(CONFIG, confirm=True))
    assert out["error"].endswith("nothing was deleted.") and "2 fleet(s), 1 unfinished" in out["error"]
    assert len(out["fleets"]) == 2 and out["rollouts"][0]["id"] == ROLLOUT
    assert not [r for r in api.seen if r.method == "DELETE"]
    assert dict(next(r for r in api.seen if r.url.path.endswith("/rollouts")).url.params)["status"] == "pending,scheduled,in_progress,paused"


def test_delete_configuration_happy_path_confirms_with_404(api):
    api.on("GET", f"/configurations/{CONFIG}", Seq(ok(cfg()), httpx.Response(404, json={"msg": "not found"})))
    api.on("GET", "/fleets", page([fleet_row(cid=CONFIG2)]))
    api.on("GET", "/rollouts", page([]))
    api.on("DELETE", f"/configurations/{CONFIG}", ok(None))
    out = j(server.delete_configuration(CONFIG, confirm=True))
    assert out["deleted"] is True and out["read_back"]["confirmed"] is True


def test_delete_configuration_reports_when_it_still_reads_back(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    api.on("GET", "/fleets", page([]))
    api.on("GET", "/rollouts", page([]))
    api.on("DELETE", f"/configurations/{CONFIG}", ok(None))
    out = j(server.delete_configuration(CONFIG, confirm=True))
    assert out["deleted"] is False and "still reads back" in out["summary"]


# ================================================================= fleets ===


def test_get_fleet_assembles_a_complete_picture(api):
    detail = fleet_row(description="Shops", location="Sydney", update_window={"enabled": True, "days": ["Monday"], "start_time": "02:00", "end_time": "04:00", "timezone": "UTC"})
    api.on("GET", f"/fleets/{FLEET}", ok(detail))
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(version=7, pinned=False))
    api.on("GET", f"/fleets/{FLEET}/update-policy", ok({"mode": "latest"}))
    api.on(
        "GET",
        f"/fleets/{FLEET}/configuration/history",
        ok([{"configuration_id": CONFIG, "config_version": n, "assigned_by": "u", "assigned_at": f"2026-01-0{n}", "reason": "r"} for n in range(1, 8)]),
    )
    api.on("GET", "/rollouts", page([{"id": ROLLOUT, "name": "r", "type": "config", "status": "paused", "fleet_ids": [FLEET], "fleet_names": {FLEET: "shop"}, "pausedReason": "operator"}]))

    out = j(server.get_fleet(FLEET))

    assert out["summary"] == "shop: 4 device(s), 3 online; kiosk follows latest (v7)."
    assert out["configuration"]["assignment"] == "latest" and out["configuration"]["resolved_version"] == 7
    assert out["devices"] == {"total": 4, "online": 3, "offline": 1}
    assert out["update_window"]["enabled"] is True and out["update_policy"] == {"mode": "latest"}
    assert len(out["recent_assignments"]) == 5 and out["recent_assignments"][0]["config_version"] == 7
    assert out["active_rollouts"][0]["paused_reason"] == "operator"
    assert out["fleet"]["tags"] == [{"key": "env", "value": "prod"}]
    params = dict(next(r for r in api.seen if r.url.path == "/v1/rollouts").url.params)
    assert params["fleet_id"] == FLEET and params["status"] == "pending,scheduled,in_progress,paused"


def test_get_fleet_by_name_pinned_and_sub_failure_is_a_warning(api):
    api.on("GET", "/fleets", ok([fleet_row()]))
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(version=5, pinned=True))
    api.on("GET", f"/fleets/{FLEET}/update-policy", httpx.Response(403, json={"msg": "no"}))
    api.on("GET", f"/fleets/{FLEET}/configuration/history", ok([]))
    api.on("GET", "/rollouts", page([]))

    out = j(server.get_fleet("shop"))

    assert "pinned to v5" in out["summary"] and out["configuration"]["assignment"] == 5
    assert "update_policy" not in out and "update policy" in out["warnings"][0]


def test_get_fleet_without_configuration_and_ambiguity(api):
    api.on("GET", "/fleets", ok([fleet_row(cid=None)]))
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row(cid=None)))
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(cid=None))
    api.on("GET", f"/fleets/{FLEET}/update-policy", ok({"mode": "latest"}))
    api.on("GET", f"/fleets/{FLEET}/configuration/history", ok([]))
    api.on("GET", "/rollouts", page([]))
    assert "no configuration" in j(server.get_fleet("shop"))["summary"]
    api.on("GET", "/fleets", ok([fleet_row(name="shop"), fleet_row(FLEET2, "shop")]))
    out = j(server.get_fleet("shop"))
    assert [c["id"] for c in out["candidates"]] == [FLEET, FLEET2]


def assign_api(api: FakeAPI, before: httpx.Response, after: httpx.Response, rollouts: list | None = None) -> None:
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7)))
    api.on("GET", f"/fleets/{FLEET}/configuration", Seq(before, after))
    api.on("GET", "/rollouts", page(rollouts or []))
    api.on("POST", f"/fleets/{FLEET}/configuration", ok({"id": FLEET, "configuration_id": CONFIG}))


def test_assign_fleet_configuration_pins_a_version_and_returns_before_after(api):
    assign_api(
        api,
        fleet_cfg(CONFIG2, "signage", 2, False, spec(image="sign:1")),
        fleet_cfg(CONFIG, "kiosk", 5, True, spec(image="kiosk:5")),
    )
    out = j(server.assign_fleet_configuration(FLEET, CONFIG, 5, confirm=True))

    assert bodies(api, "POST", "/configuration") == [{"configuration_id": CONFIG, "version": 5}]
    assert out["before"]["configuration"]["name"] == "signage" and out["after"]["assignment"] == 5
    assert out["read_back"]["confirmed"] is True and out["diff"][0]["path"] == "image"
    assert out["summary"] == "shop: signage @ latest → kiosk @ 5 (confirmed)."
    assert any("no canary" in w for w in out["warnings"]) and any("create_rollout" in w for w in out["warnings"])
    assert "does not push" in out["delivery"]


def test_assign_fleet_configuration_latest_omits_version_and_warns_about_active_rollouts(api):
    assign_api(
        api,
        fleet_cfg(CONFIG, "kiosk", 5, True),
        fleet_cfg(CONFIG, "kiosk", 7, False),
        rollouts=[{"id": ROLLOUT, "name": "r", "type": "config", "status": "in_progress", "fleet_ids": [FLEET]}],
    )
    out = j(server.assign_fleet_configuration(FLEET, CONFIG, confirm=True))
    assert bodies(api, "POST", "/configuration") == [{"configuration_id": CONFIG}]
    assert any("unfinished rollout" in w for w in out["warnings"]) and out["after"]["assignment"] == "latest"


def test_assign_fleet_configuration_refuses_no_ops_and_bad_versions(api):
    assign_api(api, fleet_cfg(CONFIG, "kiosk", 7, False), fleet_cfg(CONFIG, "kiosk", 7, False))
    assert j(server.assign_fleet_configuration(FLEET, CONFIG, "latest", confirm=True))["error"].startswith("No change")
    api.on("GET", f"/fleets/{FLEET}/configuration", Seq(fleet_cfg(CONFIG, "kiosk", 5, True)))
    assert j(server.assign_fleet_configuration(FLEET, CONFIG, 5, confirm=True))["error"].startswith("No change")
    assert "between 1 and 7" in j(server.assign_fleet_configuration(FLEET, CONFIG, 99, confirm=True))["error"]
    assert not [r for r in api.seen if r.method == "POST"]


def test_assign_fleet_configuration_flags_an_unconfirmed_write(api):
    assign_api(api, fleet_cfg(CONFIG2, "signage", 2), fleet_cfg(CONFIG2, "signage", 2))
    out = j(server.assign_fleet_configuration(FLEET, CONFIG, confirm=True))
    assert out["read_back"]["confirmed"] is False and "NOT confirmed" in out["summary"]


def test_assign_fleet_configuration_ambiguity_sends_nothing(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", "/configurations", ok([cfg(id=CONFIG, name="kiosk-a"), cfg(id=CONFIG2, name="kiosk-b")]))
    out = j(server.assign_fleet_configuration(FLEET, "kiosk", confirm=True))
    assert len(out["candidates"]) == 2 and not [r for r in api.seen if r.method == "POST"]


def test_get_fleet_configuration_history_newest_first_with_names(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on(
        "GET",
        f"/fleets/{FLEET}/configuration/history",
        ok(
            [
                {"configuration_id": CONFIG2, "config_version": 2, "assigned_by": "u", "assigned_at": "2026-01-01", "reason": "first"},
                {"configuration_id": CONFIG, "assigned_by": "u", "assigned_at": "2026-02-01", "reason": "second"},
            ]
        ),
    )
    api.on("GET", "/configurations", page([cfg(), cfg(id=CONFIG2, name="signage")]))
    out = j(server.get_fleet_configuration_history(FLEET, limit=5))
    assert [h["configuration"]["name"] for h in out["history"]] == ["kiosk", "signage"]
    assert out["history"][0]["version"] == "latest" and out["history"][1]["version"] == 2
    assert out["summary"].endswith("latest: kiosk @ latest")


def test_get_fleet_configuration_history_survives_name_lookup_failure(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/configuration/history", ok([{"configuration_id": CONFIG, "config_version": 1, "assigned_at": "t"}]))
    api.on("GET", "/configurations", httpx.Response(403, json={"msg": "no"}))
    out = j(server.get_fleet_configuration_history(FLEET))
    assert out["history"][0]["configuration"] == {"id": CONFIG, "name": None}


def test_create_fleet_with_tags_and_configuration(api):
    api.on("GET", "/configurations", ok([cfg()]))
    api.on("GET", "/fleets", ok([fleet_row(name="shop")]))
    api.on("POST", "/fleets", httpx.Response(201, json=envelope({"id": FLEET2, "name": "shop"})))
    api.on("GET", f"/fleets/{FLEET2}", ok(fleet_row(FLEET2, "shop", CONFIG, description="d", location="Perth")))

    out = j(server.create_fleet(" shop ", description="d", location="Perth", tags={"env": "prod", "n": 3}, configuration="kiosk"))

    assert bodies(api, "POST", "/fleets") == [
        {"name": "shop", "description": "d", "location": "Perth", "tags": [{"key": "env", "value": "prod"}, {"key": "n", "value": "3"}], "configuration_id": CONFIG}
    ]
    assert out["fleet"]["id"] == FLEET2 and out["read_back"]["confirmed"] is True
    assert "already use this name" in out["warnings"][0]
    assert "with configuration kiosk" in out["summary"]


def test_create_fleet_minimal_and_validation(api):
    api.on("GET", "/fleets", ok([]))
    api.on("POST", "/fleets", httpx.Response(201, json=envelope({"id": FLEET2})))
    api.on("GET", f"/fleets/{FLEET2}", ok(fleet_row(FLEET2, "lab", None)))
    out = j(server.create_fleet("lab", tags=["a=b", {"key": "c", "value": "d"}]))
    assert bodies(api, "POST", "/fleets")[0] == {"name": "lab", "tags": [{"key": "a", "value": "b"}, {"key": "c", "value": "d"}]}
    assert "no configuration" in out["summary"] and not out["warnings"]
    assert "name is required" in j(server.create_fleet(""))["error"]
    assert "Cannot read tag" in j(server.create_fleet("x", tags=[3]))["error"]


def test_update_fleet_name_and_incremental_tags(api):
    after = fleet_row(name="shop-2", tags=[{"key": "tier", "value": "gold"}, {"key": "region", "value": "au"}])
    api.on("GET", f"/fleets/{FLEET}", Seq(ok(fleet_row()), ok(fleet_row()), ok(after)))
    api.on("PUT", f"/fleets/{FLEET}", ok({"id": FLEET}))
    api.on("POST", f"/fleets/{FLEET}/tags", ok({"id": FLEET, "tags": []}))
    api.on("DELETE", f"/fleets/{FLEET}/tags", ok({"id": FLEET, "tags": []}))

    out = j(server.update_fleet(FLEET, name="shop-2", tags_add={"tier": "gold", "region": "au"}, tags_remove=["env", "ghost"]))

    assert bodies(api, "PUT", f"/fleets/{FLEET}") == [{"name": "shop-2"}]
    assert bodies(api, "POST", "/tags") == [{"tags": [{"key": "tier", "value": "gold"}, {"key": "region", "value": "au"}]}]
    delete = next(r for r in api.seen if r.method == "DELETE")
    assert delete.url.params.get_list("key") == ["env"]
    assert out["ignored_tag_removals"] == ["ghost"] and out["read_back"]["confirmed"] is True
    assert out["before"]["tags"] == {"env": "prod"} and out["after"]["tags"] == {"tier": "gold", "region": "au"}


def test_update_fleet_replace_tags_and_validation(api):
    api.on("GET", f"/fleets/{FLEET}", Seq(ok(fleet_row()), ok(fleet_row()), ok(fleet_row(tags=[{"key": "a", "value": "1"}]))))
    api.on("PUT", f"/fleets/{FLEET}/tags", ok({"id": FLEET, "tags": []}))
    out = j(server.update_fleet(FLEET, tags_replace={"a": 1}))
    assert bodies(api, "PUT", "/tags") == [{"tags": [{"key": "a", "value": "1"}]}] and out["read_back"]["confirmed"] is True
    assert "tags_replace is empty" in j(server.update_fleet(FLEET, tags_replace={}))["error"]
    assert "cannot be combined" in j(server.update_fleet(FLEET, tags_replace={"a": "b"}, tags_add={"c": "d"}))["error"]
    assert "cannot be empty" in j(server.update_fleet(FLEET, description=" "))["error"]
    assert "Nothing to change" in j(server.update_fleet(FLEET))["error"]


def test_update_fleet_no_op(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    out = j(server.update_fleet(FLEET, name="shop", tags_add={"env": "prod"}, tags_remove=["ghost"]))
    assert out["error"].startswith("No change") and out["ignored_tag_removals"] == ["ghost"]
    assert not [r for r in api.seen if r.method != "GET"]
    out = j(server.update_fleet(FLEET, tags_replace={"env": "prod"}))
    assert out["error"].startswith("No change")


def policy_api(api: FakeAPI, before: dict[str, Any], after: dict[str, Any]) -> None:
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/update-policy", Seq(ok(before), ok(after)))
    api.on("PUT", f"/fleets/{FLEET}/update-policy", ok(after))
    api.on("PUT", f"/fleets/{FLEET}/update-window", ok({"id": FLEET}))


WINDOW = {"enabled": True, "days": ["monday", "Friday"], "start_time": "02:00", "end_time": "05:30", "timezone": "Australia/Sydney"}


def test_set_fleet_update_policy_window_only_uses_the_window_route(api):
    clean = {**WINDOW, "days": ["Monday", "Friday"]}
    policy_api(api, {"mode": "latest"}, {"mode": "latest", "update_window": clean})
    out = j(server.set_fleet_update_policy(FLEET, update_window=WINDOW, confirm=True))
    assert bodies(api, "PUT", "/update-window") == [clean]
    assert not bodies(api, "PUT", "/update-policy")
    assert out["read_back"]["confirmed"] is True and "window on" in out["summary"]


def test_set_fleet_update_policy_pinned_with_targets_and_window(api):
    target = {"architecture": "amd64", "system_version_id": "33333333-4444-5555-6666-777777777777"}
    clean = {**WINDOW, "days": ["Monday", "Friday"]}
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": [target], "update_window": clean})
    out = j(server.set_fleet_update_policy(FLEET, targets=[target], update_window=WINDOW, confirm=True))
    assert bodies(api, "PUT", "/update-policy") == [{"mode": "pinned", "targets": [target], "update_window": clean}]
    assert out["before"]["mode"] == "latest" and out["after"]["mode"] == "pinned" and out["read_back"]["confirmed"] is True


def test_set_fleet_update_policy_back_to_latest_drops_targets(api):
    t = {"architecture": "amd64"}
    policy_api(api, {"mode": "pinned", "targets": [t]}, {"mode": "latest"})
    j(server.set_fleet_update_policy(FLEET, mode="LATEST", confirm=True))
    assert bodies(api, "PUT", "/update-policy") == [{"mode": "latest"}]


def test_set_fleet_update_policy_keeps_existing_targets_when_only_window_changes_with_mode(api):
    t = {"architecture": "arm64", "board": "b1"}
    policy_api(api, {"mode": "pinned", "targets": [t]}, {"mode": "pinned", "targets": [t], "update_window": {**WINDOW, "days": ["Monday", "Friday"]}})
    j(server.set_fleet_update_policy(FLEET, mode="pinned", update_window=WINDOW, confirm=True))
    assert bodies(api, "PUT", "/update-policy")[0]["targets"] == [t]


def test_set_fleet_update_policy_disable_window_keeps_the_old_values(api):
    old = {**WINDOW, "days": ["Monday"]}
    off = {**old, "enabled": False}
    policy_api(api, {"mode": "latest", "update_window": old}, {"mode": "latest", "update_window": off})
    out = j(server.set_fleet_update_policy(FLEET, disable_update_window=True, confirm=True))
    assert bodies(api, "PUT", "/update-window") == [off] and out["read_back"]["confirmed"] is True


def test_set_fleet_update_policy_refusals_and_validation(api):
    policy_api(api, {"mode": "latest"}, {"mode": "latest"})
    assert j(server.set_fleet_update_policy(FLEET, mode="latest", confirm=True))["error"].startswith("No change")
    assert "needs targets" in j(server.set_fleet_update_policy(FLEET, mode="pinned", confirm=True))["error"]
    assert "needs an architecture" in j(server.set_fleet_update_policy(FLEET, targets=[{"board": "x"}], confirm=True))["error"]
    assert "Unknown mode" in j(server.set_fleet_update_policy(FLEET, mode="fast", confirm=True))["error"]
    assert "only apply to mode 'pinned'" in j(server.set_fleet_update_policy(FLEET, mode="latest", targets=[{"architecture": "a"}], confirm=True))["error"]
    assert "Nothing to change" in j(server.set_fleet_update_policy(FLEET, confirm=True))["error"]
    assert "not both" in j(server.set_fleet_update_policy(FLEET, update_window=WINDOW, disable_update_window=True, confirm=True))["error"]
    for bad in ({"enabled": True, "days": [], "start_time": "02:00", "end_time": "03:00", "timezone": "UTC"},
                {"enabled": True, "days": ["Funday"], "start_time": "02:00", "end_time": "03:00", "timezone": "UTC"},
                {"enabled": True, "days": ["Monday"], "start_time": "2am", "end_time": "03:00", "timezone": "UTC"},
                {"enabled": True, "days": ["Monday"], "start_time": "02:00", "end_time": "03:00"},
                {"enabled": True, "days": ["Monday"], "start_time": "02:00", "end_time": "03:00", "timezone": "UTC", "extra": 1}):
        assert "error" in j(server.set_fleet_update_policy(FLEET, update_window=bad, confirm=True))
    assert "update_window must be an object" in j(server.set_fleet_update_policy(FLEET, update_window=["x"], confirm=True))["error"]
    assert not [r for r in api.seen if r.method == "PUT"]


def test_set_fleet_update_policy_window_no_op(api):
    clean = {**WINDOW, "days": ["Monday", "Friday"]}
    policy_api(api, {"mode": "latest", "update_window": clean}, {"mode": "latest", "update_window": clean})
    assert j(server.set_fleet_update_policy(FLEET, update_window=WINDOW, confirm=True))["error"].startswith("No change")


# =============================================================== devices ===


def dev_detail(name: str = "dan-qemu-3", fleet_id: str = FLEET, fleet_name: str = "shop", tags: list | None = None, notes: str | None = None, **over: Any) -> dict[str, Any]:
    out = {"id": DEV, "name": name, "fleet": {"id": fleet_id, "name": fleet_name}, "tags": tags if tags is not None else [{"key": "role", "value": "kiosk"}]}
    if notes is not None:
        out["notes"] = notes
    out.update(over)
    return out


def test_update_device_name_notes_location_and_tags(api):
    before = dev_detail()
    after = dev_detail("front-desk", notes="lobby", tags=[{"key": "role", "value": "kiosk"}, {"key": "site", "value": "syd"}], location={"latitude": -33.8, "longitude": 151.2})
    api.on("GET", f"/devices/{DEV}", Seq(ok(before), ok(after)))
    api.on("PUT", f"/devices/{DEV}", ok({"id": DEV}))

    out = j(server.update_device("dan-qemu-3", name=" front-desk ", notes="lobby", latitude=-33.8, longitude=151.2, tags_add={"site": "syd"}, screenshots_disabled=True))

    body = bodies(api, "PUT", f"/devices/{DEV}")[0]
    assert body == {
        "name": "front-desk",
        "notes": "lobby",
        "tags": [{"key": "role", "value": "kiosk"}, {"key": "site", "value": "syd"}],
        "location": {"latitude": -33.8, "longitude": 151.2},
        "screenshots_disabled": True,
    }
    assert out["read_back"]["confirmed"] is True and out["before"]["name"] == "dan-qemu-3" and out["after"]["name"] == "front-desk"
    assert_pat(api.seen[0])


def test_update_device_replace_remove_and_unconfirmed_read_back(api):
    api.on("GET", f"/devices/{DEV}", Seq(ok(dev_detail()), ok(dev_detail())))
    api.on("PUT", f"/devices/{DEV}", ok({}))
    out = j(server.update_device("x", tags={"a": "b"}))
    assert bodies(api, "PUT", f"/devices/{DEV}")[0]["tags"] == [{"key": "a", "value": "b"}]
    assert out["read_back"]["confirmed"] is False and "NOT fully confirmed" in out["summary"]

    api.on("GET", f"/devices/{DEV}", Seq(ok(dev_detail()), ok(dev_detail(tags=[]))))
    j(server.update_device("x", tags_remove=["role"]))
    assert bodies(api, "PUT", f"/devices/{DEV}")[-1]["tags"] == []


def test_update_device_validation_and_no_op(api):
    assert "name cannot be empty" in j(server.update_device("x", name=" "))["error"]
    assert "together" in j(server.update_device("x", latitude=1.0))["error"]
    assert "cannot be combined" in j(server.update_device("x", tags={"a": "b"}, tags_add={"c": "d"}))["error"]
    assert "latitude must be" in j(server.update_device("x", latitude=95.0, longitude=0.0))["error"]
    assert api.seen == []
    api.on("GET", f"/devices/{DEV}", ok(dev_detail(notes="n")))
    assert j(server.update_device("x", name="dan-qemu-3", notes="n", tags_add={"role": "kiosk"}))["error"].startswith("Nothing to change")
    assert not [r for r in api.seen if r.method == "PUT"]


def test_update_device_unresolved_returns_candidates_and_sends_nothing(api, monkeypatch):
    cands = {"match": None, "candidates": [{"id": DEV, "name": "a"}, {"id": "other", "name": "a"}], "organization_id": "org-1"}
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: cands)
    out = j(server.update_device("a", name="b"))
    assert len(out["candidates"]) == 2 and api.seen == []
    assert len(j(server.move_device_to_fleet("a", FLEET, confirm=True))["candidates"]) == 2
    assert len(j(server.get_device_configuration("a"))["candidates"]) == 2
    assert len(j(server.set_device_configuration_override("a", {"image": "x"}, confirm=True))["candidates"]) == 2
    assert len(j(server.clear_device_configuration_override("a", confirm=True))["candidates"]) == 2
    assert api.seen == []


def test_move_device_to_fleet_reads_back_and_reports_new_configuration(api):
    api.on("GET", f"/fleets/{FLEET2}", ok(fleet_row(FLEET2, "lab")))
    api.on("GET", f"/devices/{DEV}", Seq(ok(dev_detail()), ok(dev_detail(fleet_id=FLEET2, fleet_name="lab"))))
    api.on("GET", f"/fleets/{FLEET2}/configuration", fleet_cfg(CONFIG2, "signage", 2, True))
    api.on("PUT", f"/devices/{DEV}/fleet", ok({"id": DEV, "flotilla_id": FLEET2}))

    out = j(server.move_device_to_fleet("dan-qemu-3", FLEET2, confirm=True))

    assert bodies(api, "PUT", "/fleet") == [{"flotilla_id": FLEET2}]
    assert out["from_fleet"]["name"] == "shop" and out["to_fleet"]["name"] == "lab"
    assert out["read_back"]["confirmed"] is True and "now runs signage @ 2" in out["summary"]
    assert out["next"][0]["tool"] == "get_device_workload"


def test_move_device_to_fleet_refuses_same_fleet_and_flags_unconfirmed(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/devices/{DEV}", ok(dev_detail()))
    assert j(server.move_device_to_fleet("x", FLEET, confirm=True))["error"].startswith("No change")
    api.on("GET", f"/fleets/{FLEET2}", ok(fleet_row(FLEET2, "lab", None)))
    api.on("GET", f"/fleets/{FLEET2}/configuration", fleet_cfg(None))
    api.on("PUT", f"/devices/{DEV}/fleet", ok({}))
    out = j(server.move_device_to_fleet("x", FLEET2, confirm=True))
    assert out["read_back"]["confirmed"] is False and "the new fleet has no configuration" in out["summary"]


def test_move_device_to_fleet_ambiguous_fleet(api):
    api.on("GET", "/fleets", ok([fleet_row(name="shop-a"), fleet_row(FLEET2, "shop-b")]))
    out = j(server.move_device_to_fleet("x", "shop", confirm=True))
    assert len(out["candidates"]) == 2 and not [r for r in api.seen if r.method == "PUT"]


def dev_cfg(override: dict[str, Any] | None = None, cid: str | None = CONFIG, pinned: bool = False) -> httpx.Response:
    base = spec()
    merged = {**base, **({"image": override["image"]} if override and "image" in override else {})}
    data: dict[str, Any] = {"host_id": DEV, "fleet_id": FLEET, "is_pinned": pinned, "has_override": bool(override), "base_config": base, "merged_config": merged, "host_version": 1}
    if cid:
        data.update({"configuration_id": cid, "config_version": 7, "configuration_info": {"id": cid, "name": "kiosk"}})
    if override:
        data["host_override"] = override
    return ok(data)


def test_get_device_configuration_shows_override_effect(api):
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg({"image": "kiosk:debug"}, pinned=True))
    out = j(server.get_device_configuration("dan-qemu-3"))
    assert out["has_override"] is True and out["override"] == {"image": "kiosk:debug"}
    assert out["overridden_fields"][0]["path"] == "image" and out["merged"]["environment"]["API_TOKEN"] == "***"
    assert out["assignment"] == 7 and out["configuration"]["name"] == "kiosk"
    assert "override on 1 field(s)" in out["summary"]


def test_get_device_configuration_without_fleet_configuration(api):
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg(None, cid=None))
    out = j(server.get_device_configuration("x"))
    assert out["configuration"] is None and "no configuration" in out["summary"]
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg(None))
    assert "no override" in j(server.get_device_configuration("x"))["summary"]


def test_set_device_configuration_override_merges_by_default(api):
    existing = {"image": "kiosk:debug", "environment": {"A": "1", "B": "2"}}
    api.on("GET", f"/devices/{DEV}/configuration", Seq(dev_cfg(existing), dev_cfg({"image": "kiosk:debug", "environment": {"A": "9"}})))
    api.on("PUT", f"/devices/{DEV}/configuration", ok({"id": DEV, "has_override": True}))

    out = j(server.set_device_configuration_override("x", {"environment": {"A": "9", "B": None}}, reason=" test ", confirm=True))

    assert bodies(api, "PUT", f"/devices/{DEV}/configuration") == [{"override": {"image": "kiosk:debug", "environment": {"A": "9"}}, "reason": "test"}]
    assert out["after"]["has_override"] is True and out["read_back"]["confirmed"] is True
    assert "reaches the device when it reconnects" in out["delivery"]


def test_set_device_configuration_override_replace_and_state_toggle(api):
    api.on("GET", f"/devices/{DEV}/configuration", Seq(dev_cfg({"image": "old"}), dev_cfg({"desiredState": "STOPPED"})))
    api.on("PUT", f"/devices/{DEV}/configuration", ok({}))
    j(server.set_device_configuration_override("x", {"desiredState": "STOPPED"}, replace=True, confirm=True))
    assert bodies(api, "PUT", f"/devices/{DEV}/configuration")[0] == {"override": {"desiredState": "STOPPED"}}


def test_set_device_configuration_override_guards(api):
    assert "Unknown override field" in j(server.set_device_configuration_override("x", {"imagee": "x"}, confirm=True))["error"]
    assert "non-empty" in j(server.set_device_configuration_override("x", {}, confirm=True))["error"]
    assert "RUNNING or STOPPED" in j(server.set_device_configuration_override("x", {"desiredState": "PAUSED"}, confirm=True))["error"]
    assert api.seen == []
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg({"image": "a"}))
    assert j(server.set_device_configuration_override("x", {"image": "a"}, confirm=True))["error"].startswith("No change")
    assert "empty" in j(server.set_device_configuration_override("x", {"image": None}, confirm=True))["error"]
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg(None, cid=None))
    assert "no configuration to override" in j(server.set_device_configuration_override("x", {"image": "a"}, confirm=True))["error"]
    assert not [r for r in api.seen if r.method == "PUT"]


def test_clear_device_configuration_override(api):
    api.on("GET", f"/devices/{DEV}/configuration", Seq(dev_cfg({"image": "a"}), dev_cfg(None)))
    api.on("DELETE", f"/devices/{DEV}/configuration/override", ok({"id": DEV, "has_override": False}))
    out = j(server.clear_device_configuration_override("x", reason=" done ", confirm=True))
    delete = next(r for r in api.seen if r.method == "DELETE")
    assert dict(delete.url.params) == {"reason": "done"}
    assert out["read_back"] == {"confirmed": True, "has_override": False} and out["removed_override"] == {"image": "a"}


def test_clear_device_configuration_override_refuses_when_none_and_flags_leftover(api):
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg(None))
    assert j(server.clear_device_configuration_override("x", confirm=True))["error"].startswith("No change")
    api.on("GET", f"/devices/{DEV}/configuration", dev_cfg({"image": "a"}))
    api.on("DELETE", f"/devices/{DEV}/configuration/override", ok({}))
    out = j(server.clear_device_configuration_override("x", confirm=True))
    assert out["read_back"]["confirmed"] is False and "NOT confirmed" in out["summary"]
    assert "reason" not in dict(next(r for r in api.seen if r.method == "DELETE").url.params)


# ============================================================== rollouts ===


def test_list_rollouts_filters_and_compacts(api):
    rows = [
        {"id": ROLLOUT, "name": "a", "type": "reboot", "status": "completed", "progress_pct": 100, "fleet_ids": [FLEET], "fleet_names": {FLEET: "shop"}, "is_disruptive": True, "created_at": "t", "stats": {"total": 1}},
        {"id": "r2", "name": "b", "type": "config", "status": "paused", "pausedReason": "operator", "config_name": "kiosk", "version": 3, "fleet_ids": [FLEET2]},
    ]
    api.on("GET", "/fleets", ok([fleet_row(name="shop")]))
    api.on("GET", "/rollouts", page(rows))

    out = j(server.list_rollouts(fleet="shop", status="Paused, completed", type="reboot", search="a", limit=5, page=2))

    params = dict(next(r for r in api.seen if r.url.path == "/v1/rollouts").url.params)
    assert params == {"status": "paused,completed", "fleet_id": FLEET, "search": "a", "limit": "5", "page": "2"}
    assert [r["id"] for r in out["rollouts"]] == [ROLLOUT]
    assert out["rollouts"][0]["fleets"] == ["shop"] and "stats" not in out["rollouts"][0]
    assert "type reboot" in out["summary"]


def test_list_rollouts_active_only_and_rejects_bad_type(api):
    api.on("GET", "/rollouts", page([]))
    j(server.list_rollouts(active_only=True))
    assert dict(api.seen[0].url.params)["status"] == "pending,scheduled,in_progress,paused"
    assert "Unsupported rollout type" in j(server.list_rollouts(type="nuke"))["error"]


def preview_api(api: FakeAPI, current_version: int = 5, current_cfg: str | None = CONFIG, impact: dict | None = None, rollouts: list | None = None, target_spec: dict | None = None, **fleet_over: Any) -> None:
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7)))
    api.on("GET", f"/configurations/{CONFIG}/versions/7", ok({"version_number": 7, "spec": target_spec or spec(image="registry.example.com/kiosk:2.0")}))
    api.on("GET", f"/configurations/{CONFIG}/versions/5", ok({"version_number": 5, "spec": spec()}))
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row(**fleet_over)))
    api.on("GET", f"/fleets/{FLEET2}", ok(fleet_row(FLEET2, "lab", online=0, offline=2, devices=2)))
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(current_cfg, version=current_version, pinned=True))
    api.on("GET", f"/fleets/{FLEET2}/configuration", fleet_cfg(current_cfg, version=7, pinned=False))
    api.on("POST", "/rollouts/impact", ok(impact or {"fleet_ids": [FLEET], "total_devices": 4, "online_count": 3, "is_disruptive": False}))
    api.on("GET", "/rollouts", page(rollouts or []))


def test_preview_rollout_config_is_read_only_and_ends_with_create_arguments(api):
    preview_api(api)
    out = j(server.preview_rollout(FLEET, CONFIG, strategy={"canary": 2, "max_in_flight": 5}))

    assert [r.method for r in api.seen if r.method != "GET"] == ["POST"]
    assert api.seen[[r.url.path for r in api.seen].index("/v1/rollouts/impact")].method == "POST"
    assert json.loads(next(r for r in api.seen if r.url.path.endswith("/impact")).content) == {"fleet_ids": [FLEET], "type": "config"}
    assert out["read_only"] is True and out["ready"] is True
    plan = out["fleets"][0]
    assert plan["current"]["version"] == 5 and plan["target"]["version"] == 7 and plan["same_version_noop"] is False
    assert plan["diff"][0]["path"] == "image"
    assert out["strategy"] == {"canary": 2, "maxInFlight": 5, "maxUnavailable": 2, "failureThreshold": 0.1, "progressDeadline": "30m"}
    assert out["impact"]["online_count"] == 3
    assert out["create_rollout"] == {
        "tool": "create_rollout",
        "arguments": {"fleet": FLEET, "type": "config", "configuration": CONFIG, "version": 7, "strategy": {"canary": 2, "max_in_flight": 5}},
    }
    assert out["warnings"] == [f"shop: 1 device(s) offline; they are only admitted when online, so the rollout stays in_progress until they return."]
    assert "Nothing was created" in out["summary"]


def test_preview_rollout_flags_same_version_noop(api):
    preview_api(api, current_version=7)
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(CONFIG, "kiosk", 7, True, spec(image="registry.example.com/kiosk:2.0")))
    out = j(server.preview_rollout([FLEET], CONFIG, 7))
    assert out["fleets"][0]["same_version_noop"] is True and out["fleets"][0]["diff"] == []
    assert out["ready"] is False and any("never converges" in w for w in out["warnings"])


def test_preview_rollout_can_switch_configuration_and_multiple_fleets(api):
    preview_api(api, current_cfg=CONFIG2, impact={"fleet_ids": [FLEET, FLEET2], "total_devices": 6, "online_count": 3})
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg(CONFIG2, "signage", 2, True, spec(image="sign:1")))
    api.on("GET", f"/fleets/{FLEET2}/configuration", fleet_cfg(CONFIG2, "signage", 2, True, spec(image="sign:1")))
    out = j(server.preview_rollout([FLEET, FLEET2], CONFIG))
    assert [p["current"]["configuration"]["name"] for p in out["fleets"]] == ["signage", "signage"]
    assert out["fleets"][0]["target"]["configuration"]["name"] == "kiosk"
    assert out["create_rollout"]["arguments"]["fleet"] == [FLEET, FLEET2]
    assert any("lab: 2 device(s) offline" in w for w in out["warnings"])
    assert "6 device(s) (3 online)" in out["summary"]


def test_preview_rollout_warns_about_active_rollouts_status_and_signature_policy(api):
    preview_api(
        api,
        rollouts=[{"id": ROLLOUT, "name": "other", "type": "config", "status": "in_progress", "fleet_ids": [FLEET]}],
        target_spec=spec(signaturePolicy={"required": True}),
    )
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7, status="deprecated")))
    out = j(server.preview_rollout(FLEET, CONFIG))
    text = " | ".join(out["warnings"])
    assert "unfinished rollout" in text and "deprecated" in text and "signature" in text
    assert out["active_rollouts"][0]["id"] == ROLLOUT


def test_preview_rollout_empty_fleet_and_no_online_devices(api):
    preview_api(api, devices=0, online=0, offline=0, impact={"total_devices": 0, "online_count": 0})
    out = j(server.preview_rollout(FLEET, CONFIG))
    assert out["ready"] is False and any("no devices" in w for w in out["warnings"])


def test_preview_rollout_reboot_and_system_update_and_restart(api):
    preview_api(api, impact={"fleet_ids": [FLEET], "total_devices": 4, "online_count": 3, "is_disruptive": True, "incompatible_count": 1})
    out = j(server.preview_rollout(FLEET, type="reboot", force=True))
    assert out["create_rollout"]["arguments"] == {"fleet": FLEET, "type": "reboot", "force": True}
    assert json.loads(next(r for r in api.seen if r.url.path.endswith("/impact")).content)["type"] == "reboot"
    assert any("disruptive" in w for w in out["warnings"]) and "fleets" in out and "diff" not in out["fleets"][0]

    out = j(server.preview_rollout(FLEET, type="system_update", use_current_versions=True, final_action="none"))
    assert out["create_rollout"]["arguments"] == {"fleet": FLEET, "type": "system_update", "final_action": "none", "use_current_versions": True}
    assert any("no compatible version" in w for w in out["warnings"])

    out = j(server.preview_rollout(FLEET, type="restart_workload", container_target="default"))
    assert out["create_rollout"]["arguments"]["container_target"] == "default"


def test_preview_rollout_validation(api):
    preview_api(api)
    assert "configuration is required" in j(server.preview_rollout(FLEET))["error"]
    assert "between 1 and 7" in j(server.preview_rollout(FLEET, CONFIG, 99))["error"]
    assert "Unsupported rollout type" in j(server.preview_rollout(FLEET, CONFIG, type="wipe"))["error"]
    assert "version_ids" in j(server.preview_rollout(FLEET, type="system_update"))["error"]
    assert "Unknown strategy field" in j(server.preview_rollout(FLEET, CONFIG, strategy={"batch": 1}))["error"]
    assert "listed twice" in j(server.preview_rollout([FLEET, FLEET.upper()], CONFIG))["error"]
    assert "fleet is required" in j(server.preview_rollout([], CONFIG))["error"]
    assert not [r for r in api.seen if r.method == "POST"]


def test_preview_rollout_ambiguous_fleet_returns_candidates(api):
    api.on("GET", "/fleets", ok([fleet_row(name="shop-a"), fleet_row(FLEET2, "shop-b")]))
    out = j(server.preview_rollout("shop", CONFIG))
    assert len(out["candidates"]) == 2 and calls(api) == ["GET /fleets"]


def test_preview_rollout_tolerates_a_failing_impact_call(api):
    preview_api(api)
    api.on("POST", "/rollouts/impact", httpx.Response(500, json={"msg": "down"}))
    out = j(server.preview_rollout(FLEET, CONFIG))
    assert out["impact"] is None and any(w.startswith("impact:") for w in out["warnings"])
    assert "4 device(s) (3 online)" in out["summary"]  # falls back to fleet counts


# ----------------------------------------------- create_rollout extension ---


def test_create_rollout_multiple_fleets(api):
    api.on("GET", "/fleets", Seq(ok([fleet_row(name="shop")])))
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))
    out = j(server.create_rollout(["shop", FLEET2], CONFIG, 3, confirm=True))
    body = bodies(api, "POST", "/rollouts")[0]
    assert body["fleet_ids"] == [FLEET, FLEET2] and body["name"] == f"{CONFIG} v3 → 2 fleets"
    assert out["rollout"]["id"] == ROLLOUT


def test_create_rollout_other_types(api):
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))
    j(server.create_rollout(FLEET, type="reboot", force=True, confirm=True))
    j(server.create_rollout(FLEET, type="restart_workload", container_target="default", strategy={"canary": 0}, confirm=True))
    j(server.create_rollout(FLEET, type="system_update", version_ids=["33333333-4444-5555-6666-777777777777"], final_action="restart", name="os", confirm=True))
    j(server.create_rollout(FLEET, type="system_update", use_current_versions=True, confirm=True))
    j(server.create_rollout(FLEET, type="reboot", confirm=True))
    posted = bodies(api, "POST", "/rollouts")
    assert posted[0] == {"name": f"reboot → {FLEET}", "type": "reboot", "fleet_ids": [FLEET], "reboot_spec": {"force": True}}
    assert posted[1]["restart_workload_spec"] == {"container_target": "default"} and posted[1]["strategy"] == {"canary": 0}
    assert posted[2]["system_update_spec"] == {"final_action": "restart", "version_ids": ["33333333-4444-5555-6666-777777777777"]} and posted[2]["name"] == "os"
    assert posted[3]["system_update_spec"] == {"final_action": "reboot", "use_current_versions": True}
    assert "reboot_spec" not in posted[4] and "config_spec" not in posted[4]


def test_create_rollout_validation_never_posts(api):
    assert "configuration is required" in j(server.create_rollout(FLEET, confirm=True))["error"]
    assert "Unsupported rollout type" in j(server.create_rollout(FLEET, type="wipe", confirm=True))["error"]
    assert "version_ids" in j(server.create_rollout(FLEET, type="system_update", confirm=True))["error"]
    assert "not both" in j(server.create_rollout(FLEET, type="system_update", use_current_versions=True, version_ids=[FLEET], confirm=True))["error"]
    assert "final_action" in j(server.create_rollout(FLEET, type="system_update", use_current_versions=True, final_action="halt", confirm=True))["error"]
    assert "must be UUIDs" in j(server.create_rollout(FLEET, type="system_update", version_ids=["v1"], confirm=True))["error"]
    assert "listed twice" in j(server.create_rollout([FLEET, FLEET], CONFIG, 1, confirm=True))["error"]
    assert "fleet is required" in j(server.create_rollout([], CONFIG, 1, confirm=True))["error"]
    assert api.seen == []


def test_create_rollout_ambiguous_fleet_or_configuration_returns_candidates(api):
    api.on("GET", "/fleets", ok([fleet_row(name="shop-a"), fleet_row(FLEET2, "shop-b")]))
    out = j(server.create_rollout("shop", CONFIG, 1, confirm=True))
    assert out["candidates"][0]["id"] == FLEET and not [r for r in api.seen if r.method == "POST"]
    api.on("GET", "/configurations", ok([cfg(id=CONFIG, name="kiosk-a"), cfg(id=CONFIG2, name="kiosk-b")]))
    out = j(server.create_rollout(FLEET, "kiosk", 1, confirm=True))
    assert len(out["candidates"]) == 2 and not [r for r in api.seen if r.method == "POST"]


def test_create_rollout_latest_for_uuid_configuration_and_unresolvable_latest(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(9)))
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))
    j(server.create_rollout(FLEET, CONFIG, confirm=True))
    assert bodies(api, "POST", "/rollouts")[0]["config_spec"] == {"config_id": CONFIG, "config_version": 9}
    api.on("GET", f"/configurations/{CONFIG}", ok({"id": CONFIG, "name": "kiosk"}))
    assert "Could not resolve" in j(server.create_rollout(FLEET, CONFIG, confirm=True))["error"]


# ===================================================== errors & metadata ===


def test_err_maps_billing_and_conflict_statuses():
    from admrl_mcp.client import AdmiralAPIError

    billing = j(server._err(AdmiralAPIError(402, "feature off")))
    assert billing["billing_gate"] is True and "entitlement" in billing["hint"]
    assert j(server._err(AdmiralAPIError(409, "exists")))["conflict"] is True
    assert j(server._err(AdmiralAPIError(404, "gone"))) == {"error": "Admiral API 404: gone"}
    assert "ValueError" in j(server._err(ValueError("x")))["error"]


NEW_READ_ONLY = {
    "list_configurations", "get_configuration", "diff_configuration_versions", "get_fleet", "get_fleet_configuration_history",
    "get_device_configuration", "list_rollouts", "preview_rollout",
}
NEW_MUTATING = {
    "create_configuration": False, "edit_configuration": True, "update_configuration_metadata": False, "rollback_configuration": True,
    "delete_configuration": True, "assign_fleet_configuration": True, "create_fleet": False, "update_fleet": False,
    "set_fleet_update_policy": True, "update_device": False, "move_device_to_fleet": True,
    "set_device_configuration_override": True, "clear_device_configuration_override": True,
}


def _tools():
    import asyncio

    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_new_tools_have_the_right_annotations():
    tools = _tools()
    for name in NEW_READ_ONLY:
        assert tools[name].annotations.readOnlyHint is True, name
    for name, destructive in NEW_MUTATING.items():
        ann = tools[name].annotations
        assert ann.readOnlyHint is False and ann.destructiveHint is destructive, name
        text = tools[name].description
        assert ("confirm=true" in text) if destructive else ("explicit request" in text.lower()), name
    assert tools["create_rollout"].annotations.destructiveHint is True


def test_new_tool_descriptions_are_public_safe():
    import re

    text = "\n".join(t.description or "" for n, t in _tools().items() if n in NEW_READ_ONLY | set(NEW_MUTATING))
    assert not re.search(r"ad" + r"min|\.internal|svc\.cluster", text, re.IGNORECASE)
    assert "Nothing is pushed" in _tools()["assign_fleet_configuration"].description
    assert "APPLIED DIRECTLY" in _tools()["assign_fleet_configuration"].description
    assert "create_rollout" in _tools()["assign_fleet_configuration"].description
    assert "READ-ONLY" in _tools()["preview_rollout"].description


def test_create_rollout_keeps_its_original_leading_parameters():
    import inspect

    names = list(inspect.signature(server.create_rollout).parameters)
    assert names[:7] == ["fleet", "configuration", "version", "strategy", "name", "description", "organization_id"]
