"""The confirmation gate: every destructive tool previews first and only mutates with confirm=true.

* schema test: every registered destructive tool has a boolean ``confirm`` parameter defaulting to false;
* table test: every destructive tool, called without confirm against a transport that records any
  non-GET request, answers ``confirmation_required`` and sent nothing but reads;
* the preview's ``next`` call carries resolved UUIDs and ``confirm: true``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from admrl_mcp import ops, server

from test_ops_tools import CONFIG, CONFIG2, FLEET, FLEET2, ROLLOUT, cfg, details, fleet_cfg, fleet_row, ok, page, spec
from test_w1_tools import DEV, FakeAPI, install

CRED = "cccccccc-1111-2222-3333-444444444444"
RULE = "dddddddd-1111-2222-3333-444444444444"
FILE = "eeeeeeee-1111-2222-3333-444444444444"

# POSTs that only read (also what the hosted read scope has to allow).
READ_ONLY_POSTS = ("/rollouts/impact", "/alerting/rules/preview", "/metrics/logs/query")


class GuardedAPI(FakeAPI):
    """FakeAPI that answers unknown GETs with an empty body and records every request that could write."""

    def __init__(self) -> None:
        super().__init__()
        self.violations: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        if request.method != "GET":
            read_only = (
                path in READ_ONLY_POSTS
                or path.startswith("/telemetry/")
                or path.endswith("/diagnostics/probe")
                or (path.endswith("/document:render") and request.url.params.get("dryRun") == "1")
            )
            if not read_only:
                self.violations.append(f"{request.method} {path}")
        if request.method == "GET" and (request.method, path) not in self.routes:
            self.seen.append(request)
            return ok({})
        return super().__call__(request)


def device_cfg(**over: Any) -> dict[str, Any]:
    out = {
        "fleet_id": FLEET,
        "configuration_id": CONFIG,
        "config_version": 7,
        "is_pinned": False,
        "configuration_info": {"name": "kiosk"},
        "base_config": spec(),
        "merged_config": spec(),
        "has_override": True,
        "host_override": {"image": "registry.example.com/kiosk:2.0"},
    }
    out.update(over)
    return out


def net_cfg(**over: Any) -> dict[str, Any]:
    out = {
        "id": "n-1",
        "scope": "device",
        "host_id": DEV,
        "version": 3,
        "interfaces": [{"match_mac": "aa:bb:cc:00:00:01", "enabled": True, "type": "ethernet", "ethernet": {"ip": {"dhcp_v4": True}}}],
        "client_networks": [{"ssid": "shop-wifi", "has_psk": True, "priority": 1}],
        "nameservers": ["192.0.2.53"],
        "updated_at": "2026-01-01T00:00:00Z",
    }
    out.update(over)
    return out


def world() -> GuardedAPI:
    api = GuardedAPI()
    on = api.on
    on("GET", f"/devices/{DEV}", ok({"id": DEV, "name": "dan-qemu-3", "fleet": {"id": FLEET, "name": "shop"}, "isOnline": {"isOnline": True}}))
    on("GET", f"/devices/{DEV}/workload", ok({"state": "running"}))
    on("GET", f"/devices/{DEV}/memory-test", ok({"supported": True, "status": {}}))
    on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"spec": {"system": {"screenshots": "enabled"}}, "status": {}}, headers={"ETag": '"v1"'}))
    on("POST", f"/devices/{DEV}/document:render", ok({"diff": [{"path": "spec.x", "from": 1, "to": 2}], "revision": "r1", "converged": False}))
    on("GET", f"/devices/{DEV}/configuration", ok(device_cfg()))
    on("GET", "/fleets", page([fleet_row(), fleet_row(FLEET2, "lab", cid=None)]))
    for fid, name in ((FLEET, "shop"), (FLEET2, "lab")):
        on("GET", f"/fleets/{fid}", ok(fleet_row(fid, name)))
        on("GET", f"/fleets/{fid}/update-policy", ok({"mode": "latest", "targets": []}))
    on("GET", f"/fleets/{FLEET}/configuration", fleet_cfg())
    on("GET", f"/fleets/{FLEET2}/configuration", fleet_cfg(None))
    on("GET", "/configurations", page([cfg(), cfg(5, id=CONFIG2, name="signage")]))
    on("GET", f"/configurations/{CONFIG}", ok(cfg()))
    on("GET", f"/configurations/{CONFIG}/details", details())
    on("GET", f"/configurations/{CONFIG2}", ok(cfg(5, id=CONFIG2, name="signage")))
    for cid in (CONFIG, CONFIG2):
        for v in range(1, 8):
            on("GET", f"/configurations/{cid}/versions/{v}", ok({"spec": spec(image=f"registry.example.com/kiosk:{v}")}))
    on("GET", "/rollouts", page([]))
    on("GET", f"/rollouts/{ROLLOUT}", ok({"id": ROLLOUT, "name": "r1", "status": "in_progress", "counts": {"targets": 4}}))
    on("POST", "/rollouts/impact", ok({"total_devices": 4, "online_count": 3}))
    on("GET", f"/devices/{DEV}/network/configuration", ok(net_cfg()))
    on("GET", f"/devices/{DEV}/network/effective-configuration", ok({"configuration": net_cfg(scope="fleet"), "source": "fleet"}))
    on("GET", f"/fleets/{FLEET}/network/configuration", ok(net_cfg(scope="fleet", interfaces=[{"match_mac": "aa:bb:cc:00:00:01", "enabled": True, "type": "ethernet", "ethernet": {"ip": {"proxy": {"server": "http://proxy.example.com:3128", "ignore_tls": False}}}}], client_networks=[])))
    on("GET", f"/fleets/{FLEET}/ssh", ok({"enabled": True}))
    on("GET", f"/fleets/{FLEET}/image-proxy", ok({"fleet_id": FLEET, "enabled": False}))
    on("GET", f"/fleets/{FLEET}/custom-metrics", ok({"enabled": False}))
    on("GET", f"/fleets/{FLEET}/security-policy", ok({"fleet_id": FLEET, "require_tpm": False, "require_secure_boot": False, "require_disk_encryption": False, "require_recovery_key_escrow": False, "enforced": False, "devices_total": 4, "devices_noncompliant": 0, "devices_not_reported": 0, "can_manage": True}))
    on("GET", f"/fleets/{FLEET}/usb-policy", ok({"fleet_id": FLEET, "policy": None}))
    on("GET", f"/devices/{DEV}/usb-policy", ok({"deviceId": DEV, "fleet_id": FLEET, "device_override": {"mode": "off", "allow": []}, "fleet_policy": None, "effective": {"mode": "off", "allow": []}, "source": "device"}))
    rule = {"id": RULE, "name": "cpu high", "severity": "warning", "datasource": "metrics", "expr": "up == 0", "enabled": True, "interval_seconds": 60, "for_seconds": 0}
    on("GET", "/alerting/rules", ok([rule]))
    on("GET", f"/alerting/rules/{RULE}", ok(rule))
    on("GET", f"/alerting/rules/{RULE}/status", ok({"synced": True, "health": "ok", "active_alerts": 0}))
    cred = {"id": CRED, "name": "ghcr", "registry": "registry.example.com", "type": "basic", "username": "svc", "has_password": True}
    on("GET", "/registry-credentials", page([cred]))
    on("GET", f"/registry-credentials/{CRED}", ok(cred))
    files = {"enabled": True, "files": [{"id": FILE, "path": "/admrl/secrets/api.key", "mode": "0600", "uid": 0, "gid": 0, "size": 12, "sha256": "ab" * 32, "updated_at": "2026-01-01T00:00:00Z"}], "limits": {"max_file_bytes": 65536, "max_files_per_scope": 32, "max_total_bytes": 524288}}
    on("GET", f"/devices/{DEV}/secret-files", ok(files))
    on("GET", f"/configurations/{CONFIG}/secret-files", ok(files))
    return api


@pytest.fixture
def api(monkeypatch) -> GuardedAPI:
    fake = world()
    install(monkeypatch, fake)
    return fake


def _destructive_tools() -> dict[str, Any]:
    tools = asyncio.run(server.mcp.list_tools())
    return {t.name: t for t in tools if t.annotations and t.annotations.destructiveHint is True}


# --------------------------------------------------------------- schema ---


def test_every_destructive_tool_takes_confirm_defaulting_to_false():
    tools = _destructive_tools()
    assert len(tools) >= 30
    for name, tool in tools.items():
        props = tool.inputSchema["properties"]
        assert "confirm" in props, name
        assert props["confirm"]["type"] == "boolean" and props["confirm"]["default"] is False, name
        assert "confirm" not in tool.inputSchema.get("required", []), name
        assert list(props)[-1] == "confirm", f"{name}: confirm must be the last parameter"


def test_destructive_descriptions_mention_the_confirmation():
    for name, tool in _destructive_tools().items():
        assert "confirm" in (tool.description or "").lower(), name


# ---------------------------------------------------------------- table ---


def _cases(tmp_path) -> dict[str, dict[str, Any]]:
    secret = tmp_path / "token.txt"
    secret.write_text("S3CR3T-registry-token\n")
    payload = tmp_path / "key.pem"
    payload.write_text("-----BEGIN KEY-----\nabc\n")
    return {
        "reboot_device": {"device": "dan-qemu-3"},
        "change_workload_status": {"device": "dan-qemu-3", "action": "restart"},
        "patch_device_document": {"device": "dan-qemu-3", "merge_patch": {"spec": {"system": {"screenshots": "disabled"}}}},
        "render_device_document": {"device": "dan-qemu-3", "push": True},
        "adopt_local_override": {"device": "dan-qemu-3"},
        "discard_local_override": {"device": "dan-qemu-3"},
        "start_memory_test": {"device": "dan-qemu-3", "mode": "full_online"},
        "cancel_memory_test": {"device": "dan-qemu-3"},
        "create_rollout": {"fleet": FLEET, "configuration": CONFIG, "version": 6},
        "rollout_control": {"rollout_id": ROLLOUT, "action": "cancel"},
        "edit_configuration": {"configuration": CONFIG, "image_tag": "9", "change_reason": "bump"},
        "rollback_configuration": {"configuration": CONFIG, "target_version": 5},
        "delete_configuration": {"configuration": CONFIG2},
        "assign_fleet_configuration": {"fleet": FLEET2, "configuration": CONFIG2, "version": 3},
        "set_fleet_update_policy": {"fleet": FLEET, "mode": "pinned", "targets": [{"architecture": "arm64"}]},
        "move_device_to_fleet": {"device": "dan-qemu-3", "fleet": FLEET2},
        "set_device_configuration_override": {"device": "dan-qemu-3", "override": {"image": "registry.example.com/kiosk:3"}},
        "clear_device_configuration_override": {"device": "dan-qemu-3"},
        "set_network_configuration": {"device": "dan-qemu-3", "configuration": {"nameservers": ["192.0.2.54"]}},
        "clear_network_configuration": {"fleet": FLEET},
        "set_fleet_ssh_access": {"fleet": FLEET, "enabled": False},
        "set_fleet_image_proxy": {"fleet": FLEET, "enabled": True},
        "set_fleet_custom_metrics": {"fleet": FLEET, "enabled": True},
        "set_fleet_security_policy": {"fleet": FLEET, "require_tpm": True},
        "set_usb_policy": {"fleet": FLEET, "mode": "block_storage_hid"},
        "clear_usb_policy": {"device": "dan-qemu-3"},
        "delete_alert_rule": {"rule": "cpu high"},
        "update_registry_credential": {"credential": "ghcr", "secret_file": str(secret)},
        "delete_registry_credential": {"credential": CRED},
        "delete_secret_file": {"file": "/admrl/secrets/api.key", "device": "dan-qemu-3"},
        "upload_secret_file": {"source_path": str(payload), "path": "/admrl/secrets/tls.key", "device": "dan-qemu-3"},
    }


CASE_NAMES = sorted(
    """reboot_device change_workload_status patch_device_document render_device_document adopt_local_override
    discard_local_override start_memory_test cancel_memory_test create_rollout rollout_control edit_configuration
    rollback_configuration delete_configuration assign_fleet_configuration set_fleet_update_policy move_device_to_fleet
    set_device_configuration_override clear_device_configuration_override set_network_configuration
    clear_network_configuration set_fleet_ssh_access set_fleet_image_proxy set_fleet_custom_metrics
    set_fleet_security_policy set_usb_policy clear_usb_policy delete_alert_rule update_registry_credential
    delete_registry_credential delete_secret_file upload_secret_file""".split()
)
IRREVERSIBLE = {"delete_configuration", "delete_alert_rule", "delete_registry_credential", "delete_secret_file", "rollout_control"}


def test_the_table_covers_every_destructive_tool(tmp_path):
    assert set(_cases(tmp_path)) == set(CASE_NAMES) == set(_destructive_tools()), "add new destructive tools to the confirmation table"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_destructive_tool_without_confirm_only_reads(name, api, tmp_path):
    kwargs = _cases(tmp_path)[name]
    if name == "delete_configuration":
        api.on("GET", "/fleets", page([fleet_row(cid=None)]))
    out = json.loads(getattr(server, name)(**kwargs))
    assert out.get("confirmation_required") is True, out
    assert not api.violations, api.violations
    assert out["summary"].startswith("Not done yet: ")
    assert out["action"]["tool"] == name
    assert out["irreversible"] is (name in IRREVERSIBLE)
    assert out["instruction"] == ops.CONFIRM_INSTRUCTION
    nxt = out["next"]
    assert nxt["tool"] == name and nxt["arguments"]["confirm"] is True
    # the confirmed call is built from resolved UUIDs, not from the names the caller typed
    text = json.dumps(nxt["arguments"])
    for typed in ("dan-qemu-3", "cpu high", "ghcr"):
        assert typed not in text, (typed, nxt)
    # the confirmed call only uses parameters the tool really has
    props = _destructive_tools()[name].inputSchema["properties"]
    assert set(nxt["arguments"]) <= set(props), set(nxt["arguments"]) - set(props)


def test_confirm_must_be_literally_true(api):
    for bogus in (None, 0, "yes", "true"):
        out = json.loads(server.reboot_device("dan-qemu-3", confirm=bogus))  # type: ignore[arg-type]
        assert out["confirmation_required"] is True
    assert not api.violations


def test_nothing_is_sent_for_an_ambiguous_target(monkeypatch, api):
    candidates = {"match": None, "candidates": [{"id": DEV, "name": "a"}, {"id": "x", "name": "a"}], "organization_id": "org-1"}
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: dict(candidates))
    for fn, kw in (
        (server.reboot_device, {"device": "a"}),
        (server.set_network_configuration, {"device": "a", "configuration": {"nameservers": []}}),
        (server.upload_secret_file, {"source_path": "/nonexistent", "path": "/admrl/secrets/x", "device": "a"}),
    ):
        out = json.loads(fn(**kw, confirm=True))
        assert "confirmation_required" not in out
    assert not [r for r in api.seen if r.method != "GET"]


# ------------------------------------------------- targeted gate behaviour ---


def test_patch_document_preview_shows_the_diff_and_pins_the_version(api):
    out = json.loads(server.patch_device_document("dan-qemu-3", {"spec": {"system": {"screenshots": "disabled"}}}))
    assert out["preview"]["diff"] == [{"path": "spec.system.screenshots", "change": "changed", "old": "enabled", "new": "disabled"}]
    assert out["next"]["arguments"]["if_match"] == "v1" and not api.violations
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(200, json={"spec": {}}, headers={"ETag": '"v2"', "X-Admrl-Push": "sent", "X-Admrl-Changed": "spec.system.screenshots"}))
    done = json.loads(server.patch_device_document(**out["next"]["arguments"]))
    patch = [r for r in api.seen if r.method == "PATCH"][0]
    assert patch.headers["If-Match"] == '"v1"' and done["push"] == "sent"


def test_every_memory_test_mode_is_gated(api):
    live = json.loads(server.start_memory_test("dan-qemu-3", mode="live"))
    assert live["confirmation_required"] is True and live["preview"]["workload_stopped"] is False
    assert "keeps running" in live["summary"] and not api.violations


def test_rollout_control_irreversibility_depends_on_the_action(api):
    flags = {a: json.loads(server.rollout_control(ROLLOUT, a))["irreversible"] for a in ("pause", "resume", "cancel", "rollback")}
    assert flags == {"pause": False, "resume": False, "cancel": True, "rollback": True}
    assert json.loads(server.rollout_control(ROLLOUT, "explode"))["error"].startswith("Unsupported action")


def test_create_rollout_preview_carries_impact_and_the_exact_request(api):
    out = json.loads(server.create_rollout(fleet="shop", configuration=CONFIG, version="latest", strategy={"canary": 2}))
    assert out["preview"]["impact"] == {"total_devices": 4, "online_count": 3}
    assert out["preview"]["request"]["config_spec"] == {"config_id": CONFIG, "config_version": 7}
    assert out["next"]["arguments"]["configuration"] == CONFIG and out["next"]["arguments"]["version"] == 7
    assert out["next"]["arguments"]["fleet"] == FLEET and out["next"]["arguments"]["strategy"] == {"canary": 2}
    assert [r.url.path for r in api.seen if r.method == "POST"] == ["/v1/rollouts/impact"]


def test_move_with_wipe_states_the_data_loss_and_sends_the_wipe_options(api):
    moved = {"v": False}
    api.on("PUT", f"/devices/{DEV}/fleet", lambda r: (moved.update(v=True), ok({"id": DEV}))[1])
    api.on("GET", f"/devices/{DEV}", lambda r: ok({"id": DEV, "name": "dan-qemu-3", "fleet": {"id": FLEET2 if moved["v"] else FLEET, "name": "lab" if moved["v"] else "shop"}, "isOnline": {"isOnline": True}}))
    out = json.loads(server.move_device_to_fleet("dan-qemu-3", "lab", wipe=True, wipe_secure=True))
    assert out["irreversible"] is True and out["action"]["wipe"] == {"images": True, "volumes": True, "secure": True}
    assert "WIPE" in out["summary"] and "persistent volumes" in out["summary"]
    assert any("IRREVERSIBLE DATA LOSS" in w and "secure erase" in w for w in out["warnings"])
    assert any("higher device permission" in w for w in out["warnings"])
    assert not [r for r in api.seen if r.method == "PUT"]
    done = json.loads(server.move_device_to_fleet(**out["next"]["arguments"]))
    body = json.loads([r for r in api.seen if r.method == "PUT"][0].content)
    assert body == {"flotilla_id": FLEET2, "wipe": True, "wipe_options": {"images": True, "volumes": True, "secure": True}}
    assert done["read_back"]["confirmed"] is True and done["wiped"] == ["cached container images", "persistent volumes (application data)"]


def test_move_without_wipe_does_not_mention_or_send_a_wipe(api):
    api.on("PUT", f"/devices/{DEV}/fleet", ok({"id": DEV}))
    out = json.loads(server.move_device_to_fleet("dan-qemu-3", "lab"))
    assert out["irreversible"] is False and out["action"]["wipe"] is False and "wipe" not in out["next"]["arguments"]
    json.loads(server.move_device_to_fleet(**out["next"]["arguments"]))
    assert json.loads([r for r in api.seen if r.method == "PUT"][0].content) == {"flotilla_id": FLEET2}


def test_move_with_wipe_needs_something_to_wipe_and_surfaces_403_504(api):
    assert "wipe needs" in json.loads(server.move_device_to_fleet("dan-qemu-3", "lab", wipe=True, wipe_images=False, wipe_volumes=False))["error"]
    api.on("PUT", f"/devices/{DEV}/fleet", httpx.Response(403, json={"msg": "not allowed to wipe this device"}))
    assert "not allowed" in json.loads(server.move_device_to_fleet("dan-qemu-3", "lab", wipe=True, confirm=True))["error"]
    api.on("PUT", f"/devices/{DEV}/fleet", httpx.Response(504, json={"msg": "Device is not contactable"}))
    assert "not contactable" in json.loads(server.move_device_to_fleet("dan-qemu-3", "lab", wipe=True, confirm=True))["error"]


def test_confirmed_calls_still_surface_409_and_402(api):
    api.on("POST", f"/fleets/{FLEET2}/configuration", httpx.Response(402, json={"msg": "feature not enabled"}))
    out = json.loads(server.assign_fleet_configuration(fleet=FLEET2, configuration=CONFIG2, version=3, confirm=True))
    assert out["billing_gate"] is True
    api.on("PUT", f"/devices/{DEV}/document", httpx.Response(409, json={"msg": "stale"}))
    api.on("PATCH", f"/devices/{DEV}/document", httpx.Response(409, json={"msg": "precondition failed"}))
    out = json.loads(server.patch_device_document("dan-qemu-3", {"spec": {"a": 1}}, confirm=True))
    assert out["conflict"] is True


def test_the_server_instructions_describe_the_two_step_flow():
    text = server.INSTRUCTIONS
    assert "Two-step confirmation" in text and "confirm=true" in text and "next.arguments" in text
    assert "Never set" in text
