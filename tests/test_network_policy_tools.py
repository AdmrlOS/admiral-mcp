"""Network configuration and fleet policy tools (mocked transport, synthetic ids)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from admrl_mcp import ops, server

from test_confirmation import FLEET, FLEET2, GuardedAPI, device_cfg, net_cfg, world
from test_ops_tools import fleet_row, ok, page
from test_w1_tools import DEV, install


def j(text: str) -> dict[str, Any]:
    return json.loads(text)


@pytest.fixture
def api(monkeypatch) -> GuardedAPI:
    fake = world()
    install(monkeypatch, fake)
    return fake


def writes(api: GuardedAPI, method: str | None = None) -> list[httpx.Request]:
    return [r for r in api.seen if r.method != "GET" and (method is None or r.method == method)]


# ------------------------------------------------------------- ops helpers ---


def test_clean_network_config_strips_read_only_fields_and_validates():
    cleaned = ops.clean_network_config(net_cfg(), fleet=False)
    assert set(cleaned) == {"interfaces", "client_networks", "nameservers"}
    assert cleaned["client_networks"] == [{"ssid": "shop-wifi", "priority": 1}]  # has_psk dropped, no psk invented
    with pytest.raises(ValueError, match="unknown field"):
        ops.clean_network_config({"interfaces": [{"match_mac": "aa", "type": "ethernet", "bogus": 1}]}, fleet=False)
    with pytest.raises(ValueError, match="interfaces is required"):
        ops.clean_network_config({"nameservers": ["192.0.2.1"]}, fleet=False)
    # a stored configuration may legitimately have no interfaces (only Wi-Fi networks)
    assert ops.clean_network_config({"interfaces": [], "client_networks": [{"ssid": "a", "has_psk": False}]}, fleet=False) == {"interfaces": [], "client_networks": [{"ssid": "a"}]}
    with pytest.raises(ValueError, match="type must be one of"):
        ops.clean_network_config({"interfaces": [{"match_mac": "aa", "type": "modem"}]}, fleet=False)
    # a fleet configuration has no static addressing
    static = {"interfaces": [{"match_mac": "aa", "type": "ethernet", "ethernet": {"ip": {"dhcp_v4": False, "ipv4": {"address": "192.0.2.5/24"}}}}]}
    with pytest.raises(ValueError, match="unknown field"):
        ops.clean_network_config(static, fleet=True)
    assert ops.clean_network_config(static, fleet=False)["interfaces"][0]["ethernet"]["ip"]["ipv4"]["address"] == "192.0.2.5/24"


def test_merge_network_upserts_keyed_lists_and_keeps_the_rest():
    base = ops.clean_network_config(net_cfg(), fleet=False)
    merged = ops.merge_network(base, {"client_networks": [{"ssid": "SHOP-wifi", "priority": 5}, {"ssid": "guest", "psk": "pw"}], "nameservers": None})
    assert [c["ssid"] for c in merged["client_networks"]] == ["shop-wifi", "guest"]
    assert merged["client_networks"][0]["priority"] == 5 and "nameservers" not in merged
    assert merged["interfaces"] == base["interfaces"]
    with pytest.raises(ValueError, match="need a ssid"):
        ops.merge_network(base, {"client_networks": [{"priority": 1}]})


def test_network_diff_masks_passwords_and_keys_list_items():
    old = {"interfaces": [{"match_mac": "aa", "type": "ethernet", "enabled": True}], "client_networks": [{"ssid": "a", "psk": "old-secret"}]}
    new = {"interfaces": [{"match_mac": "aa", "type": "ethernet", "enabled": False}], "client_networks": [{"ssid": "a", "psk": "new-secret"}]}
    rows = ops.network_diff(old, new)
    assert {r["path"] for r in rows} == {"interfaces[aa].enabled"}  # both masked to *** so the password change is not a diff row
    assert "secret" not in json.dumps(rows)
    assert ops.network_psk_supplied(new) == ["a"] and ops.network_psk_supplied(ops.mask_network_secrets(new)) == []


# ------------------------------------------------------------------- get ---


def test_get_network_configuration_for_a_device_shows_own_and_effective(api):
    out = j(server.get_network_configuration(device="dan-qemu-3"))
    assert out["target"] == {"kind": "device", "id": DEV, "name": "dan-qemu-3"}
    assert out["own"]["client_networks"] == [{"ssid": "shop-wifi", "has_psk": True, "priority": 1}]
    assert out["effective"]["source"] == "fleet"
    assert "id" not in out["own_configuration"] and "version" not in out["own_configuration"]
    assert "psk" not in json.dumps(out).replace("has_psk", "")
    assert not writes(api)


def test_get_network_configuration_for_a_fleet_and_for_none_set(api):
    out = j(server.get_network_configuration(fleet=FLEET))
    assert out["target"]["kind"] == "fleet" and "effective" not in out
    api.on("GET", f"/fleets/{FLEET}/network/configuration", ok(None))
    out = j(server.get_network_configuration(fleet=FLEET))
    assert "no own network configuration" in out["summary"] and "own_configuration" not in out


def test_get_network_configuration_needs_exactly_one_target(api):
    assert "exactly one of device or fleet" in j(server.get_network_configuration())["error"]
    assert "exactly one of device or fleet" in j(server.get_network_configuration(device="a", fleet=FLEET))["error"]


def test_network_tools_return_candidates_for_an_ambiguous_device(monkeypatch, api):
    found = {"match": None, "candidates": [{"id": DEV, "name": "kiosk"}, {"id": "x", "name": "kiosk"}], "organization_id": "org-1"}
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: dict(found))
    for fn, kw in ((server.get_network_configuration, {}), (server.set_network_configuration, {"configuration": {"nameservers": []}}), (server.clear_network_configuration, {})):
        out = j(fn(device="kiosk", **kw))
        assert len(out["candidates"]) == 2
    assert not writes(api)


# ------------------------------------------------------------------- set ---


def test_set_network_configuration_preview_shows_diff_risks_and_masks_passwords(api):
    out = j(server.set_network_configuration(configuration={"client_networks": [{"ssid": "guest", "psk": "hunter2-secret"}]}, device="dan-qemu-3"))
    assert out["confirmation_required"] is True and not writes(api)
    assert [r["path"] for r in out["preview"]["diff"]] == ["client_networks[guest].psk", "client_networks[guest].ssid"]
    assert any("strand" in w for w in out["warnings"]) and any("stops following the fleet" in w for w in out["warnings"])
    assert "hunter2-secret" not in json.dumps(out)
    assert out["next"]["arguments"]["mode"] == "replace" and out["next"]["arguments"]["device"] == DEV
    assert out["next"]["arguments"]["configuration"]["client_networks"][1] == {"ssid": "guest", "psk": "***"}
    assert out["preview"]["before"]["source"] == "fleet"


def test_set_network_configuration_warns_when_nothing_stays_enabled(api):
    out = j(
        server.set_network_configuration(
            configuration={"interfaces": [{"match_mac": "aa:bb:cc:00:00:01", "type": "ethernet", "enabled": False}]}, device="dan-qemu-3"
        )
    )
    assert any("No interface is enabled" in w for w in out["warnings"])


def test_set_network_configuration_merge_writes_and_reads_back(api):
    stored = net_cfg()
    api.on("PUT", f"/devices/{DEV}/network/configuration", ok({"saved": True, "applied": True}))
    api.on("GET", f"/devices/{DEV}/network/configuration", lambda r: ok(stored if len(writes(api)) else net_cfg(nameservers=["192.0.2.53"])))
    # after the PUT the API returns the merged configuration (a stored password reads back as has_psk)
    new_cn = [{"ssid": "shop-wifi", "has_psk": True, "priority": 1}, {"ssid": "guest", "has_psk": True}]
    stored["client_networks"], stored["nameservers"] = new_cn, ["192.0.2.54"]
    out = j(server.set_network_configuration(configuration={"nameservers": ["192.0.2.54"], "client_networks": [{"ssid": "guest", "psk": "pw-1"}]}, device=DEV, confirm=True))
    put = writes(api, "PUT")[0]
    body = json.loads(put.content)
    assert put.url.path.endswith(f"/devices/{DEV}/network/configuration")
    assert body["nameservers"] == ["192.0.2.54"]
    assert body["client_networks"] == [{"ssid": "shop-wifi", "priority": 1}, {"ssid": "guest", "psk": "pw-1"}]  # stored networks kept, no psk invented
    assert "id" not in body and "version" not in body
    assert out["saved"] is True and out["applied"] is True and out["read_back"]["confirmed"] is True
    assert "pw-1" not in json.dumps(out) and "saved and applied" in out["summary"]


def test_set_network_configuration_offline_device_saves_and_defers(api):
    api.on("PUT", f"/devices/{DEV}/network/configuration", httpx.Response(202, json={"code": 202, "msg": "saved", "data": {"saved": True, "applied": False}}))
    out = j(server.set_network_configuration(configuration={"nameservers": ["192.0.2.60"]}, device=DEV, confirm=True))
    assert out["saved"] is True and out["applied"] is False and "not applied yet" in out["summary"]


def test_set_network_configuration_device_rejecting_it_is_reported_as_saved(api):
    api.on("PUT", f"/devices/{DEV}/network/configuration", httpx.Response(502, json={"msg": "Network configuration saved, but the device did not accept it"}))
    out = j(server.set_network_configuration(configuration={"nameservers": ["192.0.2.60"]}, device=DEV, confirm=True))
    assert out["saved"] is True and out["applied"] is False and "did not accept" in out["error"]


def test_set_network_configuration_validation_errors_send_nothing(api):
    out = j(server.set_network_configuration(configuration={"interfaces": [{"match_mac": "aa", "type": "ethernet", "mtu": 9000}]}, device=DEV, mode="replace", confirm=True))
    assert "unknown field" in out["error"] and "mtu" in out["error"]
    assert "Unknown mode" in j(server.set_network_configuration(configuration={"a": 1}, device=DEV, mode="patch"))["error"]
    assert "non-empty" in j(server.set_network_configuration(configuration={}, device=DEV))["error"]
    assert not writes(api)


def test_set_network_configuration_refuses_a_no_op(api):
    out = j(server.set_network_configuration(configuration={"nameservers": ["192.0.2.53"]}, device=DEV, confirm=True))
    assert out["error"].startswith("No change") and not writes(api)


def test_set_fleet_network_configuration_uses_the_fleet_route_and_fleet_model(api):
    api.on("PUT", f"/fleets/{FLEET}/network/configuration", ok(None))
    out = j(
        server.set_network_configuration(
            configuration={"interfaces": [{"match_mac": "aa:bb:cc:00:00:01", "type": "ethernet", "enabled": True, "ethernet": {"ip": {"proxy": {"server": "http://proxy.example.com:3128"}}}}]},
            fleet="shop",
            mode="replace",
        )
    )
    assert any("every device of the fleet" in w for w in out["warnings"])
    done = j(server.set_network_configuration(configuration=out["next"]["arguments"]["configuration"], fleet=FLEET, mode="replace", confirm=True))
    assert writes(api, "PUT")[0].url.path.endswith(f"/fleets/{FLEET}/network/configuration")
    assert done["target"]["kind"] == "fleet"
    static = {"interfaces": [{"match_mac": "aa", "type": "ethernet", "ethernet": {"ip": {"ipv4": {"address": "192.0.2.9/24"}}}}]}
    assert "unknown field" in j(server.set_network_configuration(configuration=static, fleet=FLEET, mode="replace"))["error"]


def test_set_network_configuration_surfaces_403_and_404(api):
    api.on("PUT", f"/devices/{DEV}/network/configuration", httpx.Response(403, json={"msg": "access denied"}))
    out = j(server.set_network_configuration(configuration={"nameservers": ["192.0.2.60"]}, device=DEV, confirm=True))
    assert "access denied" in out["error"]
    api.on("GET", f"/fleets/{FLEET2}", httpx.Response(404, json={"msg": "Fleet not found"}))
    assert "not found" in j(server.get_network_configuration(fleet=FLEET2))["error"].lower()


# ----------------------------------------------------------------- clear ---


def test_clear_network_configuration_device_preview_and_delete(api):
    out = j(server.clear_network_configuration(device="dan-qemu-3"))
    assert out["confirmation_required"] is True and not writes(api)
    assert out["preview"]["removed"]["client_networks"][0]["ssid"] == "shop-wifi"
    assert any("strand" in w for w in out["warnings"])
    gone = {"n": 0}
    api.on("DELETE", f"/devices/{DEV}/network/configuration", lambda r: (gone.update(n=1), ok(None))[1])
    api.on("GET", f"/devices/{DEV}/network/configuration", lambda r: ok(None if gone["n"] else net_cfg()))
    done = j(server.clear_network_configuration(device=DEV, confirm=True))
    assert done["read_back"]["confirmed"] is True and writes(api, "DELETE")


def test_clear_network_configuration_refuses_when_there_is_none(api):
    api.on("GET", f"/fleets/{FLEET}/network/configuration", ok(None))
    out = j(server.clear_network_configuration(fleet=FLEET, confirm=True))
    assert out["error"].startswith("No change") and not writes(api)


# -------------------------------------------------------- fleet policies ---


def test_get_fleet_policies_collects_all_five_and_flags_missing_entitlements(api):
    api.on("GET", f"/fleets/{FLEET}/usb-policy", httpx.Response(402, json={"msg": "feature not enabled"}))
    out = j(server.get_fleet_policies("shop"))
    assert out["ssh"] == {"enabled": True} and out["image_proxy"]["enabled"] is False and out["custom_metrics"] == {"enabled": False}
    assert out["security"]["enforced"] is False and out["security"]["devices"]["total"] == 4
    assert "usb" not in out and any("usb" in w and "402" in w for w in out["warnings"])
    assert "ssh on" in out["summary"] and not writes(api)


def test_get_fleet_policies_reports_a_usb_policy(api):
    api.on("GET", f"/fleets/{FLEET}/usb-policy", ok({"fleet_id": FLEET, "policy": {"mode": "allowlist", "allow": [{"vendor_id": "0bda"}]}}))
    out = j(server.get_fleet_policies(FLEET))
    assert out["usb"]["mode"] == "allowlist" and out["usb"]["allow"] == [{"vendor_id": "0bda"}]


@pytest.mark.parametrize(
    "tool,leaf,args,before,body",
    [
        ("set_fleet_ssh_access", "ssh", {"enabled": False}, True, {"enabled": False}),
        ("set_fleet_image_proxy", "image-proxy", {"enabled": True}, False, {"enabled": True}),
        ("set_fleet_custom_metrics", "custom-metrics", {"enabled": True}, False, {"enabled": True}),
    ],
)
def test_fleet_toggles_write_and_read_back(api, tool, leaf, args, before, body):
    state = {"v": before}
    api.on("PUT", f"/fleets/{FLEET}/{leaf}", lambda r: (state.update(v=json.loads(r.content)["enabled"]), ok({"enabled": state["v"]}))[1])
    api.on("GET", f"/fleets/{FLEET}/{leaf}", lambda r: ok({"enabled": state["v"]}))
    fn = getattr(server, tool)
    preview = j(fn(fleet="shop", **args))
    assert preview["confirmation_required"] is True and state["v"] is before
    assert preview["next"]["arguments"] == {"fleet": FLEET, **args, "organization_id": "org-1", "confirm": True}
    done = j(fn(**preview["next"]["arguments"]))
    assert json.loads(writes(api, "PUT")[0].content) == body
    assert done["read_back"]["confirmed"] is True and done["after"] == {"enabled": args["enabled"]}
    again = j(fn(fleet=FLEET, confirm=True, **args))
    assert again["error"].startswith("No change")


def test_image_proxy_surfaces_billing_gate_and_permission(api):
    api.on("PUT", f"/fleets/{FLEET}/image-proxy", httpx.Response(402, json={"msg": "feature not enabled"}))
    out = j(server.set_fleet_image_proxy(fleet=FLEET, enabled=True, confirm=True))
    assert out["billing_gate"] is True
    api.on("PUT", f"/fleets/{FLEET}/image-proxy", httpx.Response(403, json={"msg": "billing permission required"}))
    assert "billing permission" in j(server.set_fleet_image_proxy(fleet=FLEET, enabled=True, confirm=True))["error"]


def test_image_proxy_usage(api):
    api.on("GET", f"/fleets/{FLEET}/image-proxy/usage", ok({"fleet_id": FLEET, "period_start": "2026-10-01T00:00:00Z", "period_end": "2026-10-09T00:00:00Z", "totals": {"bytes": 1024}, "daily": [], "devices": []}))
    out = j(server.get_image_proxy_usage("shop", start="2026-10-01"))
    assert out["totals"] == {"bytes": 1024}
    usage = [r for r in api.seen if r.url.path.endswith("/usage")][0]
    assert usage.url.params["start"] == "2026-10-01" and "end" not in usage.url.params


def test_security_policy_resends_unchanged_flags(api):
    state = {"require_tpm": False, "require_secure_boot": True, "require_disk_encryption": False, "require_recovery_key_escrow": False}
    base = {"fleet_id": FLEET, "enforced": True, "devices_total": 4, "devices_noncompliant": 1, "devices_not_reported": 0, "can_manage": True}
    api.on("GET", f"/fleets/{FLEET}/security-policy", lambda r: ok({**base, **state}))
    api.on("PUT", f"/fleets/{FLEET}/security-policy", lambda r: (state.update(json.loads(r.content)), ok({**base, **state}))[1])
    preview = j(server.set_fleet_security_policy(fleet="shop", require_tpm=True))
    assert preview["preview"]["after"] == {**state, "require_tpm": True} and preview["preview"]["devices"]["non_compliant_now"] == 1
    done = j(server.set_fleet_security_policy(**preview["next"]["arguments"]))
    assert json.loads(writes(api, "PUT")[0].content) == {"require_tpm": True, "require_secure_boot": True, "require_disk_encryption": False, "require_recovery_key_escrow": False}
    assert done["read_back"]["confirmed"] is True
    assert "Nothing to change" in j(server.set_fleet_security_policy(fleet=FLEET))["error"]
    assert j(server.set_fleet_security_policy(fleet=FLEET, require_tpm=True, confirm=True))["error"].startswith("No change")


def test_security_policy_respects_can_manage(api):
    api.on("GET", f"/fleets/{FLEET}/security-policy", ok({"fleet_id": FLEET, "require_tpm": False, "can_manage": False}))
    assert "cannot manage" in j(server.set_fleet_security_policy(fleet=FLEET, require_tpm=True, confirm=True))["error"]
    assert not writes(api)


# ------------------------------------------------------------------- USB ---


def test_usb_rule_validation_is_client_side(api):
    for rules, text in (
        ([{"vendor_id": "xyz1"}], "4 hex digits"),
        ([{"product_id": "0001"}], "requires vendor_id"),
        ([{"comment": "x"}], "vendor_id or interface_class"),
        ([{"vendor_id": "0bda", "colour": "red"}], "unknown field"),
    ):
        assert text in j(server.set_usb_policy(mode="allowlist", allow=rules, fleet=FLEET))["error"]
    assert "only apply to mode 'allowlist'" in j(server.set_usb_policy(mode="off", allow=[{"vendor_id": "0bda"}], fleet=FLEET))["error"]
    assert "Unknown mode" in j(server.set_usb_policy(mode="deny", fleet=FLEET))["error"]
    assert not writes(api)


def test_set_usb_policy_fleet_writes_normalised_rules(api):
    state: dict[str, Any] = {"policy": None}
    api.on("GET", f"/fleets/{FLEET}/usb-policy", lambda r: ok({"fleet_id": FLEET, "policy": state["policy"]}))
    api.on("PUT", f"/fleets/{FLEET}/usb-policy", lambda r: (state.update(policy=json.loads(r.content)), ok({"fleet_id": FLEET, "policy": state["policy"], "devices_notified": 4}))[1])
    preview = j(server.set_usb_policy(mode="allowlist", allow=[{"vendor_id": "0BDA", "product_id": "8153", "comment": "NIC"}, {"vendor_id": "0bda", "product_id": "8153"}], fleet="shop"))
    assert preview["preview"]["after"]["allow"] == [{"vendor_id": "0bda", "product_id": "8153", "comment": "NIC"}]  # lower-cased, duplicate dropped
    done = j(server.set_usb_policy(**preview["next"]["arguments"]))
    assert json.loads(writes(api, "PUT")[0].content) == {"mode": "allowlist", "allow": [{"vendor_id": "0bda", "product_id": "8153", "comment": "NIC"}]}
    assert done["read_back"]["confirmed"] is True and done["devices_notified"] == 4
    assert j(server.set_usb_policy(mode="allowlist", allow=preview["next"]["arguments"]["allow"], fleet=FLEET, confirm=True))["error"].startswith("No change")


def test_set_usb_policy_empty_allowlist_warns_and_device_override_uses_device_route(api):
    out = j(server.set_usb_policy(mode="allowlist", device="dan-qemu-3"))
    assert any("blocks every USB device" in w for w in out["warnings"])
    assert out["preview"]["before"]["source"] == "device"
    api.on("PUT", f"/devices/{DEV}/usb-policy", ok({"delivered": True}))
    j(server.set_usb_policy(mode="allowlist", device=DEV, confirm=True))
    assert writes(api, "PUT")[0].url.path.endswith(f"/devices/{DEV}/usb-policy")


def test_usb_policy_is_an_enterprise_feature(api):
    api.on("PUT", f"/fleets/{FLEET}/usb-policy", httpx.Response(402, json={"msg": "usb_policy not enabled"}))
    out = j(server.set_usb_policy(mode="block_storage_hid", fleet=FLEET, confirm=True))
    assert out["billing_gate"] is True


def test_get_and_clear_usb_policy(api):
    out = j(server.get_usb_policy(device="dan-qemu-3"))
    assert out["source"] == "device" and out["effective"]["mode"] == "off"
    api.on("GET", f"/fleets/{FLEET}/usb-policy", ok({"fleet_id": FLEET, "policy": None}))
    assert j(server.clear_usb_policy(fleet=FLEET, confirm=True))["error"].startswith("No change")
    api.on("GET", f"/fleets/{FLEET}/usb-policy", ok({"fleet_id": FLEET, "policy": {"mode": "block_storage_hid", "allow": []}}))
    preview = j(server.clear_usb_policy(fleet=FLEET))
    assert preview["preview"]["removed"]["mode"] == "block_storage_hid" and not writes(api)


def test_fleet_row_used_by_world_is_synthetic():
    assert fleet_row()["id"] == FLEET and device_cfg()["fleet_id"] == FLEET and page([]).status_code == 200


def test_effective_configuration_500_means_none_and_a_first_config_can_be_merged_in(api):
    # the API answers 500 (not an empty 200) when neither the device nor its fleet has a configuration
    api.on("GET", f"/devices/{DEV}/network/configuration", httpx.Response(200, json={"success": True, "msg": "No device network configuration found", "code": 200}))
    api.on("GET", f"/devices/{DEV}/network/effective-configuration", httpx.Response(500, json={"msg": "Failed to retrieve configuration"}))
    out = j(server.get_network_configuration(device=DEV))
    assert "no own network configuration" in out["summary"] and "effective" not in out
    preview = j(server.set_network_configuration(configuration={"client_networks": [{"ssid": "QDynamics", "priority": 1}]}, device=DEV))
    assert preview["next"]["arguments"]["configuration"] == {"interfaces": [], "client_networks": [{"ssid": "QDynamics", "priority": 1}]}
    assert not any("No interface is enabled" in w for w in preview["warnings"])
    assert not writes(api)
