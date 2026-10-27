"""Fleet selectors must survive confirmation, transport and read-back unchanged."""

import copy

import httpx
import pytest

from admrl_mcp import server
from test_ops_tools import WINDOW, bodies, fleet_cfg, fleet_row, j, ok, page, policy_api
from test_w1_tools import FLEET, assert_pat

# Reuse the existing isolated HTTP fixture, never a live fleet.
from test_ops_tools import api  # noqa: F401


SELECTION = {"mode": "follow", "kernel_track": "linux-7.2", "features": {"nvidia": {"track": "nvidia-595"}}}
TARGETS = [
    {"architecture": "amd64", "target": "x86"},
    {"architecture": "amd64", "target": "x86-nvidia", "system_selection": SELECTION},
    {"architecture": "arm64", "target": "rk3568", "board": "board-a", "system_version_id": "33333333-4444-5555-6666-777777777777"},
]


def test_mixed_targets_confirmation_and_roundtrip(api):
    after = {"mode": "pinned", "targets": list(reversed(TARGETS))}
    policy_api(api, {"mode": "latest"}, after)
    preview = j(server.set_fleet_update_policy(FLEET, targets=TARGETS))
    assert preview["confirmation_required"] is True
    assert preview["preview"]["request"] == {"mode": "pinned", "targets": TARGETS}
    args = preview["next"]["arguments"]
    assert args == {"fleet": FLEET, "targets": TARGETS, "organization_id": "org-1", "base_revision": "", "confirm": True}
    assert not [r for r in api.seen if r.method != "GET"]
    # Preview consumed one GET; reset the mock sequence for the confirmed invocation.
    policy_api(api, {"mode": "latest"}, after)
    out = j(server.set_fleet_update_policy(**args))
    assert out["read_back"]["confirmed"] is True
    assert bodies(api, "PUT", "/update-policy") == [{"mode": "pinned", "targets": TARGETS}]
    assert out["after"]["targets"] == server._policy_targets(TARGETS)


@pytest.mark.parametrize("field", ["system_selection", "system_version_id", "target", "board"])
def test_dropped_target_fields_refuse_success(api, field):
    dropped = copy.deepcopy(TARGETS)
    index = 1 if field == "system_selection" else 2
    dropped[index].pop(field)
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": dropped})
    out = j(server.set_fleet_update_policy(FLEET, targets=TARGETS, base_revision="", confirm=True))
    assert "read-back mismatch" in out["error"]
    assert out["read_back"]["confirmed"] is False
    assert out["expected"]["targets"] == server._policy_targets(TARGETS)


def test_changed_nested_track_refuses_success(api):
    changed = copy.deepcopy(TARGETS)
    changed[1]["system_selection"]["features"]["nvidia"]["track"] = "nvidia-615"
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": changed})
    assert j(server.set_fleet_update_policy(FLEET, targets=TARGETS, base_revision="", confirm=True))["read_back"]["confirmed"] is False


def test_window_only_detects_lost_pins(api):
    clean = {**WINDOW, "days": ["Monday", "Friday"]}
    policy_api(api, {"mode": "pinned", "targets": TARGETS}, {"mode": "pinned", "update_window": clean})
    out = j(server.set_fleet_update_policy(FLEET, update_window=WINDOW, base_revision="", confirm=True))
    assert out["read_back"]["confirmed"] is False
    assert not bodies(api, "PUT", "/update-policy")


def test_optional_empty_fields_and_order_normalize(api):
    normalized = [{**t, "board": t.get("board", ""), "initram_version_id": None} for t in reversed(TARGETS)]
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": normalized})
    assert j(server.set_fleet_update_policy(FLEET, targets=TARGETS, base_revision="", confirm=True))["read_back"]["confirmed"] is True


@pytest.mark.parametrize("target", [
    {**TARGETS[1], "system_version_id": "33333333-4444-5555-6666-777777777777"},
    {**TARGETS[1], "system_selection": {}},
    {**TARGETS[1], "system_selection": {"mode": "pin", "kernel_track": "linux-7.2"}},
    {**TARGETS[1], "system_selection": {"mode": "follow", "kernel_track": "linux-7.2", "unknown": True}},
    {**TARGETS[1], "system_selection": {**SELECTION, "features": {"nvidia": {"track": ""}}}},
    {**TARGETS[1], "target": "x86"},
    {**TARGETS[1], "unknown": True},
    {**TARGETS[1], "target": ""},
    {**TARGETS[1], "system_selection": {"mode": "follow", "kernel_track": "linux-next"}},
])
def test_invalid_selectors_never_write(api, target):
    policy_api(api, {"mode": "latest"}, {"mode": "latest"})
    assert "error" in j(server.set_fleet_update_policy(FLEET, targets=[target], base_revision="", confirm=True))
    assert not [r for r in api.seen if r.method != "GET"]


def test_get_fleet_keeps_complete_selectors(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg())
    api.on("GET", f"/fleets/{FLEET}/update-policy", ok({"mode": "pinned", "revision": "policy-1", "targets": TARGETS}))
    api.on("GET", f"/fleets/{FLEET}/configuration/history", ok([]))
    api.on("GET", "/rollouts", page([]))
    policy = j(server.get_fleet(FLEET))["update_policy"]
    assert policy["targets"] == TARGETS and policy["revision"] == "policy-1"


def test_update_options_passthrough_is_org_scoped_and_read_only(api):
    catalogue = {"groups": [{"architecture": "amd64", "target": "x86-nvidia", "board": "", "options": [{"slot_id": "33333333-4444-5555-6666-777777777777", "head_revision": "head-1", "release_selection": {"kernel_track": "linux-7.2", "features": SELECTION["features"], "kernel_compatibility_id": "sha256:" + "a" * 64}}], "resolved_slot_id": "33333333-4444-5555-6666-777777777777", "blocked_reason": "mandatory floor blocks selected track"}]}
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/update-options", ok(catalogue))
    out = j(server.get_fleet_update_options(FLEET))
    assert out["update_options"] == catalogue
    assert all(r.method == "GET" for r in api.seen)
    for request in api.seen:
        assert_pat(request)


def test_update_options_errors_not_replaced_with_legacy_catalogue(api):
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/update-options", httpx.Response(403, json={"code": 403, "msg": "Forbidden"}))
    assert "error" in j(server.get_fleet_update_options(FLEET))
    assert all(r.method == "GET" for r in api.seen)


@pytest.mark.parametrize("revision", ["", "policy-1"])
def test_confirm_requires_preview_revision(api, revision):
    policy_api(api, {"mode": "latest", "revision": revision}, {})
    out = j(server.set_fleet_update_policy(FLEET, targets=TARGETS, confirm=True))
    assert "approved base_revision" in out["error"]
    assert all(r.method == "GET" for r in api.seen)


def test_revision_change_after_preview_never_writes(api):
    policy_api(api, {"mode": "latest", "revision": "policy-1"}, {})
    preview = j(server.set_fleet_update_policy(FLEET, targets=TARGETS))
    args = preview["next"]["arguments"]
    assert args["base_revision"] == "policy-1"
    policy_api(api, {"mode": "latest", "revision": "policy-2"}, {})
    out = j(server.set_fleet_update_policy(**args))
    assert "fresh preview and approval" in out["error"]
    assert all(r.method == "GET" for r in api.seen)


@pytest.mark.parametrize("status", [409, 412])
def test_cas_conflict_never_retries(api, status):
    policy_api(api, {"mode": "latest", "revision": "policy-1"}, {})
    api.on("PUT", f"/fleets/{FLEET}/update-policy", httpx.Response(status, json={"msg": "changed"}))
    out = j(server.set_fleet_update_policy(FLEET, targets=TARGETS, base_revision="policy-1", confirm=True))
    assert "Nothing retried" in out["error"]
    writes = [r for r in api.seen if r.method != "GET"]
    assert len(writes) == 1 and writes[0].headers["If-Match"] == "policy-1"


def test_revisioned_window_uses_cas_preserves_selectors(api):
    before = {"mode": "pinned", "revision": "policy-1", "targets": TARGETS}
    clean = {**WINDOW, "days": ["Monday", "Friday"]}
    policy_api(api, before, {**before, "revision": "policy-2", "update_window": clean})
    out = j(server.set_fleet_update_policy(FLEET, update_window=WINDOW, base_revision="policy-1", confirm=True))
    assert out["read_back"]["confirmed"] is True
    writes = [r for r in api.seen if r.method != "GET"]
    assert len(writes) == 1 and writes[0].url.path.endswith("/update-policy")
    assert writes[0].headers["If-Match"] == "policy-1"
    assert bodies(api, "PUT", "/update-policy")[0]["targets"] == server._policy_targets(TARGETS)


@pytest.mark.parametrize("tool", ["get_fleet_update_options", "set_fleet_update_policy"])
def test_other_org_denial_stops_before_options_or_write(api, tool):
    api.on("GET", f"/fleets/{FLEET}", httpx.Response(404, json={"msg": "Not found"}))
    args = {"fleet": FLEET, "organization_id": "org-other"}
    if tool == "set_fleet_update_policy":
        args.update(targets=TARGETS, base_revision="policy-1", confirm=True)
    assert "error" in j(getattr(server, tool)(**args))
    assert len(api.seen) == 1
    assert api.seen[0].headers["X-Organization-ID"] == "org-other"


def test_selector_tools_available_in_browser_and_hosted():
    import asyncio
    from admrl_mcp import browser, hosted
    from admrl_mcp.hosted import HostedConfig
    for runtime in (browser.build_browser_server(), hosted.build_hosted_server(HostedConfig())):
        tools = {t.name: t for t in asyncio.run(runtime.list_tools())}
        assert tools["get_fleet_update_options"].annotations.readOnlyHint is True
        setter = tools["set_fleet_update_policy"]
        assert setter.annotations.destructiveHint is True
        assert setter.inputSchema["properties"]["confirm"]["default"] is False
        assert "base_revision" in setter.inputSchema["properties"]


def test_feature_only_selection_revision_roundtrip(api):
    target = {"architecture": "amd64", "target": "x86-nvidia", "system_selection": {"mode": "follow", "features": {"nvidia": {"track": "nvidia-595"}}}}
    policy_api(api, {"mode": "latest", "revision": "policy-1"}, {"mode": "pinned", "revision": "policy-2", "targets": [target]})
    out = j(server.set_fleet_update_policy(FLEET, targets=[target], base_revision="policy-1", confirm=True))
    assert out["read_back"]["confirmed"] is True
    writes = [r for r in api.seen if r.method != "GET"]
    assert len(writes) == 1 and writes[0].headers["If-Match"] == "policy-1"


FR_TARGETS = [
    {"architecture": "amd64", "target": "x86-nvidia", "follow": "tag:lts", "ring": "soak"},
    {"architecture": "arm64", "target": "rk3568", "board": "board-a", "follow": "track:linux-7.2"},
]


def test_follow_and_ring_roundtrip_with_defaults_ignored(api):
    served = [{**FR_TARGETS[0], "board": ""}, {**FR_TARGETS[1], "ring": "general"}]
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": served})
    out = j(server.set_fleet_update_policy(FLEET, targets=FR_TARGETS, base_revision="", confirm=True))
    assert out["read_back"]["confirmed"] is True
    assert bodies(api, "PUT", "/update-policy") == [{"mode": "pinned", "targets": FR_TARGETS}]


@pytest.mark.parametrize("field,value", [("follow", "tag:stable"), ("ring", "general"), ("ring", None)])
def test_dropped_or_changed_follow_ring_refuses_success(api, field, value):
    served = copy.deepcopy(FR_TARGETS)
    if value is None:
        served[0].pop(field)
    else:
        served[0][field] = value
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": served})
    out = j(server.set_fleet_update_policy(FLEET, targets=FR_TARGETS, base_revision="", confirm=True))
    assert out["read_back"]["confirmed"] is False


def test_legacy_complete_selector_matches_folded_follow(api):
    folded = [{"architecture": "amd64", "target": "x86-nvidia", "follow": "track:linux-7.2/nvidia-595"}]
    policy_api(api, {"mode": "latest"}, {"mode": "pinned", "targets": folded})
    legacy = [{"architecture": "amd64", "target": "x86-nvidia", "system_selection": SELECTION}]
    out = j(server.set_fleet_update_policy(FLEET, targets=legacy, base_revision="", confirm=True))
    assert out["read_back"]["confirmed"] is True
    assert bodies(api, "PUT", "/update-policy") == [{"mode": "pinned", "targets": legacy}]


@pytest.mark.parametrize("extra", [
    {"follow": "tag:beta"}, {"follow": "track:7.2"}, {"follow": "version:"}, {"ring": "fast"},
    {"follow": "tag:lts", "system_version_id": "33333333-4444-5555-6666-777777777777"},
    {"follow": "tag:lts", "system_selection": SELECTION},
    {"follow": "tag:lts", "target": ""},
])
def test_invalid_follow_ring_never_writes(api, extra):
    policy_api(api, {"mode": "latest"}, {})
    target = {"architecture": "amd64", "target": "x86-nvidia", **extra}
    assert "error" in j(server.set_fleet_update_policy(FLEET, targets=[target], base_revision="", confirm=True))
    assert all(r.method == "GET" for r in api.seen)


def test_update_options_resolution_summary(api):
    group = {"architecture": "amd64", "target": "x86-nvidia", "board": "", "follow": "tag:lts", "ring": "soak", "resolved_track": "linux-7.2/nvidia-595", "served_ring": "current", "resolved_slot_id": "s1", "blocked_reason": "", "options": []}
    api.on("GET", f"/fleets/{FLEET}", ok(fleet_row()))
    api.on("GET", f"/fleets/{FLEET}/update-options", ok({"groups": [group]}))
    out = j(server.get_fleet_update_options(FLEET))
    assert out["resolution"][0]["follow"] == "tag:lts" and out["resolution"][0]["served_ring"] == "current"
    assert out["update_options"]["groups"] == [group]


def test_rollout_use_fleet_policy_and_legacy_alias(api):
    from test_ops_tools import ROLLOUT, envelope
    api.on("POST", "/rollouts", httpx.Response(201, json=envelope({"id": ROLLOUT})))
    j(server.create_rollout(FLEET, type="system_update", use_fleet_policy=True, confirm=True))
    j(server.create_rollout(FLEET, type="system_update", use_current_versions=True, confirm=True))
    posted = bodies(api, "POST", "/rollouts")
    assert posted[0]["system_update_spec"] == {"final_action": "reboot", "use_fleet_policy": True}
    assert posted[1]["system_update_spec"] == {"final_action": "reboot", "use_current_versions": True}
    out = j(server.create_rollout(FLEET, type="system_update", use_fleet_policy=True, version_ids=[FLEET], confirm=True))
    assert "not both" in out["error"]
    assert len(bodies(api, "POST", "/rollouts")) == 2
