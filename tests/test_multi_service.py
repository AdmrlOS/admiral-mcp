"""Multi-service workloads (service sets): version-2 configuration specs, per-service workload state, commands,
logs, edits and failure explanations. Every test fakes the HTTP API (httpx.MockTransport) with synthetic ids and
covers the version-2 shape and the legacy single-image / single-workload shape older backends and agents return."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from test_ops_tools import bodies, cfg, details, edit_api, fleet_row, j, ok, page
from test_ops_tools import spec as legacy_spec
from test_troubleshoot import build_api, by_code, codes, good_state, run
from test_troubleshoot import install as ts_install
from test_w1_tools import CONFIG, DEV, FakeAPI, envelope, install

from admrl_mcp import ops, server, workloadset
from admrl_mcp import troubleshoot as ts
from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings

CRED = "cccccccc-1111-2222-3333-444444444444"


# ------------------------------------------------------------------ fixtures ---


def svc(**over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image": "registry.example.com/web:1.0",
        "command": [],
        "environment": {},
        "ports": [],
        "mounts": [],
        "volumes": [],
        "secrets": [],
        "restart": "always",
        "tty": False,
        "readOnlyRoot": False,
        "ephemeralRoot": False,
    }
    out.update(over)
    return out


def v2(services: dict[str, dict[str, Any]] | None = None, **over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "version": 2,
        "desiredState": "RUNNING",
        "services": services if services is not None else {"app": svc(environment={"MODE": "prod", "API_TOKEN": "s3cret"}, ports=[{"port": 80, "protocol": "tcp"}])},
        "volumes": {},
        "options": {"hostResolv": True},
    }
    out.update(over)
    return out


def two_services() -> dict[str, dict[str, Any]]:
    return {
        "web": svc(image="registry.example.com/web:1.4", environment={"DB_HOST": "db", "SESSION_TOKEN": "abc"}, ports=[{"port": 8080, "protocol": "tcp", "hostPort": 80}], dependsOn={"db": {"condition": "service_started"}}),
        "db": svc(image="docker.example.com/postgres:16", volumes=[{"name": "data", "destination": "/var/lib/postgresql/data"}], secrets=["/admrl/secrets/pg"]),
    }


# ------------------------------------------------------------------------ ops ---


def test_norm_spec_v2_fills_defaults_per_service_and_keeps_unknown_keys():
    raw = {"version": 2, "services": {"web": {"image": " web:1 ", "command": None, "environment": None, "ports": [{"port": 80, "protocol": " TCP "}, {"port": 53}], "x": 1}}, "volumes": None, "extra": True}
    out = ops.norm_spec(raw)
    web = out["services"]["web"]
    assert web["image"] == "web:1" and web["command"] == [] and web["environment"] == {} and web["volumes"] == [] and web["secrets"] == []
    assert [p["protocol"] for p in web["ports"]] == ["tcp", "tcp"] and web["x"] == 1
    assert out["desiredState"] == "RUNNING" and out["volumes"] == {} and out["options"] == {} and out["extra"] is True
    assert "image" not in out  # nothing legacy is invented at the top level


def test_is_v2_distinguishes_shapes_and_legacy_still_normalises_as_before():
    assert ops.is_v2({"version": 2}) and ops.is_v2({"services": {}}) and not ops.is_v2({"image": "a"}) and not ops.is_v2(None)
    legacy = ops.norm_spec({"image": " a "})
    assert legacy["image"] == "a" and legacy["ports"] == [] and legacy["environment"] == {} and "services" not in legacy


def test_mask_spec_masks_every_service_environment():
    masked = ops.mask_spec(v2(two_services()))
    assert masked["services"]["web"]["environment"] == {"DB_HOST": "db", "SESSION_TOKEN": ops.MASK}
    assert ops.mask_spec(legacy_spec())["environment"]["API_TOKEN"] == ops.MASK


def test_spec_diff_flattens_services_and_masks_service_secrets():
    old = v2({"app": svc(environment={"MODE": "prod", "API_TOKEN": "old"}, ports=[{"port": 80, "protocol": "tcp"}])})
    new = v2(
        {
            "app": svc(image="registry.example.com/web:2.0", environment={"MODE": "prod", "API_TOKEN": "new", "EXTRA": "1"}, ports=[{"port": 80, "protocol": "tcp"}, {"port": 53, "protocol": "udp"}], volumes=[{"name": "data", "destination": "/d"}]),
            "worker": svc(image="registry.example.com/worker:1"),
        },
        network={"mode": "host"},
    )
    rows = {r["path"]: r for r in ops.spec_diff(old, new)}
    assert rows["services.app.image"]["new"].endswith(":2.0")
    assert rows["services.app.environment.API_TOKEN"] == {"path": "services.app.environment.API_TOKEN", "change": "changed", "old": "***", "new": "***"}
    assert rows["services.app.environment.EXTRA"]["change"] == "added"
    assert rows["services.app.ports[udp/53]"]["change"] == "added" and "services.app.ports[tcp/80]" not in rows
    assert rows["services.app.volumes[data]"]["new"] == {"name": "data", "destination": "/d"}
    assert rows["services.worker.image"]["change"] == "added" and rows["network.mode"]["new"] == "host"
    assert not [p for p in rows if p.startswith("environment.") or p == "image"]
    unmasked = {r["path"]: r for r in ops.spec_diff(old, new, mask=False)}
    assert unmasked["services.app.environment.API_TOKEN"]["new"] == "new"


def test_spec_diff_removed_service_and_identical_sets():
    sets = v2(two_services())
    assert ops.spec_diff(sets, sets) == []
    gone = v2({"web": two_services()["web"]})
    rows = ops.spec_diff(sets, gone)
    assert {r["change"] for r in rows} == {"removed"} and any(r["path"] == "services.db.image" for r in rows)
    assert any(r["path"] == "services.db.volumes[data]" for r in rows)


def test_spec_diff_legacy_paths_unchanged():
    rows = {r["path"] for r in ops.spec_diff(legacy_spec(), legacy_spec(image="x:2", ports=[{"port": 81, "protocol": "tcp"}]))}
    assert rows == {"image", "ports[tcp/80]", "ports[tcp/81]"}


def test_spec_summary_v2_lists_services_and_single_image():
    one = ops.spec_summary(v2())
    assert one["services"] == {"app": "registry.example.com/web:1.0"} and one["image"] == "registry.example.com/web:1.0" and one["env_vars"] == 2
    many = ops.spec_summary(v2(two_services()))
    assert many["services"] == {"db": "docker.example.com/postgres:16", "web": "registry.example.com/web:1.4"} and "image" not in many
    assert many["env_vars"] == 2 and many["ports"] == 1
    assert ops.spec_summary(legacy_spec())["image"] == "registry.example.com/kiosk:1.0"


def test_credential_ids_reads_both_shapes():
    assert ops.credential_ids({"registry_credential_id": CRED.upper()}) == {CRED}
    assert ops.credential_ids(v2({"a": svc(registryCredentialId=CRED), "b": svc()})) == {CRED}
    assert ops.credential_ids(None) == set()


def test_apply_edits_default_service_and_service_selection():
    sole = v2()
    new, notes = ops.apply_spec_edits(sole, image_tag="2.0", env_set={"NEW": 1}, env_unset=["MODE"])
    app = new["services"]["app"]
    assert app["image"].endswith("web:2.0") and app["environment"] == {"API_TOKEN": "s3cret", "NEW": "1"} and notes == []

    sets = v2(two_services())
    with pytest.raises(ValueError, match=r"2 services \(db, web\): pass service="):
        ops.apply_spec_edits(sets, image_tag="2")
    with pytest.raises(ValueError, match="Unknown service 'cache'; this configuration has: db, web"):
        ops.apply_spec_edits(sets, image_tag="2", service="cache")
    new, notes = ops.apply_spec_edits(sets, image="docker.example.com/postgres:17", service="db")
    assert new["services"]["db"]["image"].endswith("postgres:17") and new["services"]["web"] == ops.norm_spec(sets)["services"]["web"]
    assert notes == ["edited service db"]
    new, notes = ops.apply_spec_edits(sets, env_unset=["ghost"], env_set={"A": "1"}, service="web")
    assert new["services"]["web"]["environment"]["A"] == "1" and "env_unset: ghost was not set (ignored)" in notes


def test_apply_edits_service_argument_rules():
    sets = v2(two_services())
    with pytest.raises(ValueError, match="service selects the target of image"):
        ops.apply_spec_edits(sets, patch={"desiredState": "STOPPED"}, service="web")
    with pytest.raises(ValueError, match="cannot be combined"):
        ops.apply_spec_edits(sets, replace=v2(), service="web")
    _, notes = ops.apply_spec_edits(legacy_spec(), image_tag="3", service="app")
    assert notes == ["service ignored: this configuration is a single image, not a service set"]


def test_apply_edits_merge_patch_addresses_services_and_rejects_single_image_keys():
    sets = v2(two_services())
    new, _ = ops.apply_spec_edits(sets, patch={"services": {"web": {"restart": "no"}, "db": None}, "network": {"mode": "host"}})
    assert new["services"]["web"]["restart"] == "no" and "db" not in new["services"] and new["network"] == {"mode": "host"}
    with pytest.raises(ValueError, match=r"\['environment', 'image'\] are single-image fields.*services: db, web"):
        ops.apply_spec_edits(sets, patch={"image": "x", "environment": {"A": "1"}})


def test_apply_edits_replacement_must_be_a_service_set():
    new, _ = ops.apply_spec_edits(legacy_spec(), replace={"services": {"a": {"image": "a:1"}}})
    assert new["version"] == 2 and new["services"]["a"]["environment"] == {}
    with pytest.raises(ValueError, match="version-2 spec.*Top-level image must move inside a service"):
        ops.apply_spec_edits(v2(), replace={"image": "a:1"})
    with pytest.raises(ValueError, match="JSON object"):
        ops.apply_spec_edits(v2(), replace=["x"])  # type: ignore[arg-type]


def test_legacy_override_preview_maps_onto_the_only_service_or_app():
    base = v2()
    merged, skipped = ops.legacy_override_preview(
        base, {"image": "x:9", "environment": {"A": "1"}, "options": {"tty": True, "read_only_root": True, "host_resolvconf": False, "exposeDBus": True, "image_cleaning": {}}, "volumes": {"d": {}}}
    )
    app = merged["services"]["app"]
    assert app["image"] == "x:9" and app["environment"]["A"] == "1" and app["tty"] is True and app["readOnlyRoot"] is True
    assert merged["options"]["hostResolv"] is False and merged["options"]["exposeDBus"] is True
    assert skipped == ["options.image_cleaning", "volumes"]
    # several services and none called app: nothing is guessed
    sets = v2(two_services())
    merged, skipped = ops.legacy_override_preview(sets, {"image": "x:9", "desiredState": "STOPPED"})
    assert merged["services"]["web"]["image"].endswith("web:1.4") and merged["desiredState"] == "STOPPED" and skipped == ["image"]
    named = v2({"app": svc(), "db": svc()})
    assert ops.legacy_override_preview(named, {"image": "x:9"})[0]["services"]["app"]["image"] == "x:9"
    # a single-image base keeps the plain merge
    assert ops.legacy_override_preview(legacy_spec(), {"image": "x:9"})[0]["image"] == "x:9"


# --------------------------------------------------------------- workloadset ---


STATE_DOC = {
    "workload": {
        "state": "DEGRADED",
        "configurationName": "shop",
        "version": 3,
        "services": [
            {"name": "db", "state": "RUNNING", "image": "docker.example.com/postgres:16", "imageDigest": "sha256:db", "restarts": 0, "health": {"status": "healthy", "failingStreak": 0}, "startedAt": "2026-11-04T01:00:00Z"},
            {"name": "web", "state": "STOPPED", "image": "registry.example.com/web:1.4", "imageDigest": "sha256:web", "restarts": 2, "exitCode": 1, "error": "exit status 1 password=hunter2", "startedAt": "0001-01-01T00:00:00Z"},
        ],
    },
    "runtime": {
        "services": [
            {"name": "db", "containerId": "default-db", "pid": 411, "ip": "10.10.0.4", "crunState": "running", "failureCount": 0},
            {"name": "web", "containerId": "default-web", "pid": 0, "ip": "10.10.0.5", "crunState": "stopped", "exitCode": 1, "failureCount": 2, "oomKilled": True},
        ]
    },
}


def test_service_rows_merge_status_and_runtime_by_name():
    rows = {r["name"]: r for r in workloadset.service_rows(None, STATE_DOC)}
    assert list(rows) == ["db", "web"]
    assert rows["db"] == {
        "name": "db", "state": "RUNNING", "image": "docker.example.com/postgres:16", "digest": "sha256:db", "restarts": 0,
        "health": {"status": "healthy", "failing_streak": 0}, "ip": "10.10.0.4", "pid": 411, "container_id": "default-db", "started_at": "2026-11-04T01:00:00Z",
    }
    assert rows["web"]["exit_code"] == 1 and rows["web"]["oom_killed"] is True and rows["web"]["restarts"] == 2 and "started_at" not in rows["web"]


def test_service_rows_prefers_the_workload_report_and_falls_back_to_runtime_only():
    wl = {"latestReport": {"status": {"state": "RUNNING", "services": [{"name": "a", "state": "RUNNING", "image": "a:1"}]}}}
    assert [r["name"] for r in workloadset.service_rows(wl, STATE_DOC)] == ["a", "db", "web"]
    only_rt = {"runtime": {"services": [{"name": "a", "crunState": "running", "pid": 5}]}}
    assert workloadset.service_rows(None, only_rt) == [{"name": "a", "state": "RUNNING", "pid": 5}]


def test_set_view_is_none_for_single_workload_agents():
    legacy = {"state": "RUNNING", "running": {"configurationName": "kiosk", "image": "a:1", "version": 4}}
    assert workloadset.set_view(legacy, {"workload": {"state": "RUNNING"}, "runtime": {"failureCount": 0}}) is None
    assert workloadset.set_view(None, None) is None


def test_set_view_summarises_the_set():
    view = workloadset.set_view({"running": {"configurationName": "shop", "version": 3}}, STATE_DOC)
    assert view is not None
    assert view["set"] == {"state": "DEGRADED", "configuration": "shop", "version": 3, "service_count": 2, "not_running": ["web"]}


def test_check_service_name_and_log_matching():
    assert workloadset.check_service_name(None) is None and workloadset.check_service_name("  ") is None
    assert workloadset.check_service_name(" web ") == "web"
    with pytest.raises(ValueError, match="Invalid service name"):
        workloadset.check_service_name("Web App")
    assert workloadset.log_matches_service({"source": "workload.default-web.stdout"}, "web")
    assert workloadset.log_matches_service({"source": "workload.default-my-web.stderr"}, "my-web")
    assert not workloadset.log_matches_service({"source": "workload.default-my-web.stderr"}, "web")
    assert not workloadset.log_matches_service({"source": "workload.default-webby.stdout"}, "web")
    assert workloadset.log_matches_service({"source": "system", "labels": {"service": "db"}}, "db")
    assert not workloadset.log_matches_service({"source": "workload.default-web.stdout", "service": "db"}, "web")
    assert not workloadset.log_matches_service({"source": "kernel"}, "web")


# ------------------------------------------------------ get_device_workload ---


def test_get_device_workload_returns_set_and_services(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "DEGRADED", "running": {"configurationName": "shop", "image": "registry.example.com/web:1.4", "version": 3},
                                                  "latestReport": {"status": {"state": "DEGRADED", "services": STATE_DOC["workload"]["services"]}}}))
    api.on("GET", f"/devices/{DEV}/state", ok({"deviceId": DEV, "state": STATE_DOC, "source": "stored"}))
    install(monkeypatch, api)

    out = j(server.get_device_workload("dan-qemu-3"))

    assert out["set"] == {"state": "DEGRADED", "configuration": "shop", "version": 3, "service_count": 2, "not_running": ["web"]}
    web = next(s for s in out["services"] if s["name"] == "web")
    assert web["ip"] == "10.10.0.5" and web["digest"] == "sha256:web" and web["restarts"] == 2 and "hunter2" not in json.dumps(out["services"])
    assert "password=[redacted]" in web["error"]
    assert out["workload"]["state"] == "DEGRADED"  # the raw payload is still there
    assert [r.url.path.removeprefix("/v1") for r in api.seen] == [f"/devices/{DEV}/workload", f"/devices/{DEV}/state"]


def test_get_device_workload_set_without_state_document_still_answers(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "RUNNING", "running": {"configurationName": "shop", "version": 3}, "services": [{"name": "web", "state": "RUNNING", "image": "w:1", "restarts": 0, "health": {"status": "healthy", "last": "ok"}}]}))
    api.on("GET", f"/devices/{DEV}/state", httpx.Response(404, json={"msg": "no state"}))
    install(monkeypatch, api)

    out = j(server.get_device_workload("dan-qemu-3"))

    assert out["set"]["state"] == "RUNNING" and out["set"]["service_count"] == 1
    assert out["services"] == [{"name": "web", "state": "RUNNING", "image": "w:1", "restarts": 0, "health": {"status": "healthy", "last": "ok"}}]


def test_get_device_workload_legacy_shape_is_unchanged_and_makes_one_call(monkeypatch):
    payload = {"state": "RUNNING", "running": {"configurationName": "kiosk", "image": "a:1", "version": 4}}
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/workload", ok(payload))
    install(monkeypatch, api)

    out = j(server.get_device_workload("dan-qemu-3"))

    assert set(out) == {"device", "workload"} and out["workload"] == payload
    assert len(api.seen) == 1


# ------------------------------------------------------ change_workload_status ---


def _settings() -> Settings:
    return Settings(api_base="https://api.test/v1", token_id="tok", secret_key="sec", org_id="org-1")


def test_client_workload_command_sends_service_only_when_given():
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content)})
        return httpx.Response(200, json=envelope({"success": True}))

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    client.workload_command("d1", action="restart", service="web", org_id="org-1")
    client.workload_command("d1", action="stop", org_id="org-1")
    client.workload_command("d1", action="start", service="", org_id="org-1")
    assert [s["body"] for s in seen] == [{"action": "RESTART", "service": "web"}, {"action": "STOP"}, {"action": "START"}]
    assert seen[0]["path"].endswith("/devices/d1/workload/cmd")


def test_change_workload_status_service_preview_then_confirmed_command(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "DEGRADED", "services": STATE_DOC["workload"]["services"]}))
    api.on("PUT", f"/devices/{DEV}/workload/cmd", ok({"deviceId": DEV, "action": "RESTART", "success": True}))
    install(monkeypatch, api)

    preview = j(server.change_workload_status("dan-qemu-3", "restart", service="web"))

    assert preview["confirmation_required"] is True and "service web of the workload" in preview["summary"]
    assert preview["action"]["service"] == "web"
    assert {k: v for k, v in preview["next"]["arguments"].items() if k != "organization_id"} == {"device": DEV, "action": "restart", "service": "web", "confirm": True}
    assert not [r for r in api.seen if r.method == "PUT"]

    out = j(server.change_workload_status(**preview["next"]["arguments"]))

    assert bodies(api, "PUT", "/workload/cmd") == [{"action": "RESTART", "service": "web"}]
    assert out["service"] == "web" and out["operation"] == "workload_restart" and out["result"]["success"] is True


def test_change_workload_status_whole_set_sends_no_service(monkeypatch):
    api = FakeAPI()
    api.on("PUT", f"/devices/{DEV}/workload/cmd", ok({"success": True}))
    install(monkeypatch, api)
    out = j(server.change_workload_status("dan-qemu-3", "stop", confirm=True))
    assert bodies(api, "PUT", "/workload/cmd") == [{"action": "STOP"}] and "service" not in out


def test_change_workload_status_unknown_or_malformed_service_sends_nothing(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/workload", ok({"state": "RUNNING", "services": STATE_DOC["workload"]["services"]}))
    install(monkeypatch, api)
    out = j(server.change_workload_status("dan-qemu-3", "stop", service="cache"))
    assert "Unknown service 'cache'" in out["error"] and out["services"] == ["db", "web"]
    assert "Invalid service name" in j(server.change_workload_status("dan-qemu-3", "stop", service="Web App", confirm=True))["error"]
    assert not [r for r in api.seen if r.method == "PUT"]


# ----------------------------------------------------------- get_device_logs ---


def _log(ts_: str, source: str, msg: str, **extra: Any) -> dict[str, Any]:
    return {"timestamp": ts_, "level": "info", "source": source, "message": msg, **extra}


def test_get_device_logs_service_filter_fetches_deep_and_filters(monkeypatch):
    captured: dict[str, Any] = {}
    rows = [
        _log("2026-11-04T01:05:00Z", "workload.default-web.stdout", "web 3"),
        _log("2026-11-04T01:04:00Z", "workload.default-db.stdout", "db 2"),
        _log("2026-11-04T01:03:00Z", "workload.default-web.stderr", "web 1"),
        _log("2026-11-04T01:02:00Z", "kernel", "kernel"),
        _log("2026-11-04T01:01:00Z", "supervisor", "labelled", labels={"service": "web"}),
    ]

    class StubClient:
        def fetch_logs(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"logs": rows, "has_more": True, "source": "query", "query_error": None}

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    monkeypatch.setattr(server, "_resolve_device", lambda *a, **k: {"match": {"id": DEV, "name": "n"}, "candidates": [], "organization_id": "org-1"})

    out = j(server.get_device_logs("n", service="web", limit=2))

    assert captured["limit"] == 1000
    assert [e["message"] for e in out["logs"]] == ["web 3", "web 1"]
    assert out["query"]["service"] == "web"
    assert out["page"]["has_more"] is True and out["page"]["next_before"].startswith("2026-11-04T01:02:59")  # just older than the oldest kept line

    out = j(server.get_device_logs("n", service="web", limit=50))
    assert [e["message"] for e in out["logs"]] == ["web 3", "web 1", "labelled"]
    assert out["page"]["next_before"].startswith("2026-11-04T01:00:59")  # oldest fetched line: filtered-out lines are not re-read as matches


def test_get_device_logs_service_filter_note_and_no_service_unchanged(monkeypatch):
    calls: list[dict[str, Any]] = []

    class StubClient:
        def fetch_logs(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"logs": [_log("2026-11-04T01:05:00Z", "workload.default-workload.stdout", "old style")], "has_more": False, "source": "query", "query_error": None}

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    monkeypatch.setattr(server, "_resolve_device", lambda *a, **k: {"match": {"id": DEV, "name": "n"}, "candidates": [], "organization_id": "org-1"})

    out = j(server.get_device_logs("n", service="web"))
    assert out["logs"] == [] and "No lines for service web" in out["note"] and out["page"] == {"has_more": False, "next_before": None}
    plain = j(server.get_device_logs("n", limit=7))
    assert calls[1]["limit"] == 7 and len(plain["logs"]) == 1 and "service" not in plain["query"] and "note" not in plain
    assert "Invalid service name" in j(server.get_device_logs("n", service="a b"))["error"]


# ---------------------------------------------------------- create_configuration ---


def _created(api: FakeAPI, back_spec: dict[str, Any]) -> None:
    api.on("POST", "/configurations", httpx.Response(201, json=envelope({"id": CONFIG, "name": "shop", "latest_version": 1, "status": "active"})))
    api.on("GET", f"/configurations/{CONFIG}/details", details(1, back_spec, name="shop"))


@pytest.fixture
def api(monkeypatch) -> FakeAPI:
    fake = FakeAPI()
    install(monkeypatch, fake)
    return fake


def test_spec_from_args_builds_a_one_service_app_set():
    built = server._spec_from_args("web:1", {"A": 1}, ["8080/udp", 80, {"port": 443, "hostPort": 8443}], ["run"], "stopped".upper(), None)
    assert built["version"] == 2 and built["desiredState"] == "STOPPED" and list(built["services"]) == ["app"]
    app = built["services"]["app"]
    assert app["image"] == "web:1" and app["environment"] == {"A": "1"} and app["command"] == ["run"]
    assert app["ports"] == [{"port": 8080, "protocol": "udp"}, {"port": 80, "protocol": "tcp"}, {"port": 443, "protocol": "tcp", "hostPort": 8443}]
    assert "image" not in built and "environment" not in built and "ports" not in built


def test_spec_from_args_validation():
    with pytest.raises(ValueError, match="needs an image"):
        server._spec_from_args(None, None, None, None, None, None)
    with pytest.raises(ValueError, match="not both"):
        server._spec_from_args("x", None, None, None, None, {"services": {"a": {"image": "a"}}})
    with pytest.raises(ValueError, match="no image on service b"):
        server._spec_from_args(None, None, None, None, None, {"services": {"a": {"image": "a"}, "b": {}}})
    with pytest.raises(ValueError, match="RUNNING or STOPPED"):
        server._spec_from_args("x", None, None, None, "PAUSED", None)
    with pytest.raises(ValueError, match="version-2 spec"):
        server._spec_from_args(None, None, None, None, None, {"image": "x"})


def test_create_configuration_from_args_posts_v2(api):
    _created(api, v2({"app": svc(image="web:1", environment={"A": "1", "DB_PASSWORD": "pw"})}))

    out = j(server.create_configuration(" shop ", image="web:1", env={"A": 1, "DB_PASSWORD": "pw"}, ports=["8080/udp", 80], command=["x"], description="d", tags=["t"]))

    body = bodies(api, "POST", "/configurations")[0]
    assert body["name"] == "shop" and body["spec"]["version"] == 2 and "image" not in body["spec"]
    app = body["spec"]["services"]["app"]
    assert app["image"] == "web:1" and app["environment"] == {"A": "1", "DB_PASSWORD": "pw"} and app["command"] == ["x"]
    assert app["ports"] == [{"port": 8080, "protocol": "udp"}, {"port": 80, "protocol": "tcp"}]
    assert out["confirmed"] is True and out["summary"].endswith("with image web:1.")
    assert out["spec"]["services"]["app"]["environment"]["DB_PASSWORD"] == "***"
    assert "source" not in out


def test_create_configuration_from_a_multi_service_spec(api):
    given = {"version": 2, "services": two_services(), "volumes": {"data": {"backup": {"enabled": True}}}}
    _created(api, v2(two_services(), volumes=given["volumes"]))

    out = j(server.create_configuration("shop", spec=given))

    sent = bodies(api, "POST", "/configurations")[0]["spec"]
    assert set(sent["services"]) == {"web", "db"} and sent["volumes"] == given["volumes"]
    assert out["summary"].endswith("with services db (docker.example.com/postgres:16), web (registry.example.com/web:1.4).")


def test_create_configuration_refuses_a_single_image_spec_and_conflicts(api):
    out = j(server.create_configuration("n", spec={"image": "y"}))
    assert "version-2 spec" in out["error"] and "Top-level image must move inside a service" in out["error"]
    assert api.seen == []
    api.on("POST", "/configurations", httpx.Response(409, json={"msg": "Configuration already exists"}))
    out = j(server.create_configuration("n", spec={"services": {"a": {"image": "y"}}}))
    assert out["conflict"] is True and "409" in out["error"]


COMPOSE = "services:\n  web:\n    image: registry.example.com/web:1.4\n    ports: ['80:8080']\n  db:\n    image: docker.example.com/postgres:16\n"


def test_create_configuration_from_compose_creates_from_the_returned_spec_and_surfaces_warnings(api):
    imported = v2(two_services())
    api.on("POST", "/configurations:importCompose", ok({"spec": imported, "warnings": ["services.web.networks: custom networks are flattened to the bridge", {"path": "services.db.env_file", "message": "inlined"}], "errors": []}))
    _created(api, imported)

    out = j(server.create_configuration("shop", compose_yaml=COMPOSE))

    assert bodies(api, "POST", ":importCompose") == [{"compose": COMPOSE}]
    assert bodies(api, "POST", "/configurations")[0]["spec"] == imported
    assert out["source"] == "compose" and out["confirmed"] is True
    assert out["warnings"] == ["services.web.networks: custom networks are flattened to the bridge", "services.db.env_file: inlined"]
    assert [r.url.path.removeprefix("/v1") for r in api.seen if r.method == "POST"] == ["/configurations:importCompose", "/configurations"]


def test_create_configuration_compose_without_warnings_still_lists_them(api):
    imported = v2(two_services())
    api.on("POST", "/configurations:importCompose", ok({"spec": imported}))
    _created(api, imported)
    assert j(server.create_configuration("shop", compose_yaml=COMPOSE))["warnings"] == []


def test_create_configuration_compose_unsupported_backend_says_so_and_creates_nothing(api):
    for status in (404, 405, 501):
        api.on("POST", "/configurations:importCompose", httpx.Response(status, json={"msg": "not found"}))
        out = j(server.create_configuration("shop", compose_yaml=COMPOSE))
        assert "does not support compose import yet" in out["error"] and out["compose_import_supported"] is False
    assert not [r for r in api.seen if r.url.path.endswith("/configurations") and r.method == "POST"]


def test_create_configuration_compose_hard_errors_are_returned_with_their_paths(api):
    api.on("POST", "/configurations:importCompose", httpx.Response(422, json={"msg": "invalid", "data": {"errors": [{"path": "services.web.build", "message": "build is not supported"}, "services.db.privileged: not supported"]}}))
    out = j(server.create_configuration("shop", compose_yaml=COMPOSE))
    assert out["errors"] == ["services.web.build: build is not supported", "services.db.privileged: not supported"] and "nothing was created" in out["error"]
    api.on("POST", "/configurations:importCompose", ok({"spec": v2(), "errors": ["services.web.cap_add: not supported"]}))
    out = j(server.create_configuration("shop", compose_yaml=COMPOSE))
    assert out["errors"] == ["services.web.cap_add: not supported"]
    api.on("POST", "/configurations:importCompose", ok({"warnings": []}))
    assert "without a spec" in j(server.create_configuration("shop", compose_yaml=COMPOSE))["error"]
    assert not [r for r in api.seen if r.url.path.endswith("/configurations") and r.method == "POST"]


def test_create_configuration_compose_argument_rules(api):
    assert "not a mix" in j(server.create_configuration("n", compose_yaml=COMPOSE, image="x"))["error"]
    assert "not a mix" in j(server.create_configuration("n", compose_yaml=COMPOSE, spec={"services": {"a": {"image": "a"}}}))["error"]
    assert "empty" in j(server.create_configuration("n", compose_yaml="  \n"))["error"]
    assert "larger than 256 KiB" in j(server.create_configuration("n", compose_yaml="x" * (257 * 1024)))["error"]
    assert api.seen == []


# ------------------------------------------------------------ edit_configuration ---


def _edit_v2(api: FakeAPI, current: dict[str, Any], after: dict[str, Any], latest: int = 7) -> None:
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(latest)))
    def current_or_saved(request: httpx.Request) -> httpx.Response:
        written = any(r.method == "PUT" for r in api.seen)
        return details(latest + 1, after) if written else details(latest, current)

    api.on("GET", f"/configurations/{CONFIG}/details", current_or_saved)
    api.on("GET", "/fleets", page([fleet_row()]))
    api.on("PUT", f"/configurations/{CONFIG}/spec", ok({"id": CONFIG, "latest_version": latest + 1}))


def test_edit_configuration_single_service_defaults_and_writes_v2(api):
    after = v2({"app": svc(image="registry.example.com/web:2.0", environment={"MODE": "prod", "API_TOKEN": "s3cret"}, ports=[{"port": 80, "protocol": "tcp"}])})
    _edit_v2(api, v2(), after)

    out = j(server.edit_configuration(CONFIG, change_reason="bump", image_tag="2.0", base_version=7, confirm=True))

    sent = bodies(api, "PUT", "/spec")[0]["spec"]
    assert sent["services"]["app"]["image"].endswith(":2.0") and "image" not in sent
    assert [c["path"] for c in out["diff"]] == ["services.app.image"]
    assert out["read_back"] == {"latest_version": 8, "confirmed": True}
    assert out["summary"].startswith("kiosk v7 → v8: services.app.image:")


def test_edit_configuration_multi_service_requires_service_and_targets_it(api):
    current = v2(two_services())
    after = v2({**two_services(), "db": svc(image="docker.example.com/postgres:17", volumes=[{"name": "data", "destination": "/var/lib/postgresql/data"}], secrets=["/admrl/secrets/pg"])})
    _edit_v2(api, current, after)

    out = j(server.edit_configuration(CONFIG, change_reason="x", image_tag="17", confirm=True))
    assert "2 services (db, web): pass service=" in out["error"]
    assert not [r for r in api.seen if r.method == "PUT"]
    out = j(server.edit_configuration(CONFIG, change_reason="x", image_tag="2", service="cache", confirm=True))
    assert "Unknown service 'cache'" in out["error"] and not [r for r in api.seen if r.method == "PUT"]

    out = j(server.edit_configuration(CONFIG, change_reason="pg 17", image="docker.example.com/postgres:17", service="db", confirm=True))
    sent = bodies(api, "PUT", "/spec")[0]["spec"]
    assert sent["services"]["db"]["image"].endswith("postgres:17") and sent["services"]["web"]["image"].endswith("web:1.4")
    assert [c["path"] for c in out["diff"]] == ["services.db.image"] and out["notes"] == ["edited service db"]
    assert out["read_back"]["confirmed"] is True


def test_edit_configuration_preview_carries_service_in_next_and_masks_secrets(api):
    _edit_v2(api, v2(two_services()), v2(two_services()))

    out = j(server.edit_configuration(CONFIG, env_set={"SESSION_TOKEN": "new", "A": "1"}, service="web", change_reason="r"))

    assert out["confirmation_required"] is True
    assert out["next"]["arguments"]["service"] == "web" and out["next"]["arguments"]["confirm"] is True
    rows = {c["path"]: c for c in out["preview"]["diff"]}
    assert rows["services.web.environment.SESSION_TOKEN"]["new"] == "***" and rows["services.web.environment.A"]["new"] == "1"
    assert "abc" not in json.dumps(out["preview"]["diff"])  # the old secret value never appears
    assert not [r for r in api.seen if r.method == "PUT"]


def test_edit_configuration_merge_patch_on_a_set_and_legacy_key_refusal(api):
    after = v2({**two_services(), "web": {**two_services()["web"], "restart": "no"}})
    _edit_v2(api, v2(two_services()), after)

    out = j(server.edit_configuration(CONFIG, change_reason="p", merge_patch={"services": {"web": {"restart": "no"}}}, confirm=True))
    assert [c["path"] for c in out["diff"]] == ["services.web.restart"] and out["read_back"]["confirmed"] is True

    bad = j(server.edit_configuration(CONFIG, change_reason="p", merge_patch={"registry_credential_id": CRED}, confirm=True))
    assert "single-image fields" in bad["error"] and "services.<name>.<field>" in bad["error"]


def test_edit_configuration_whole_replacement_must_be_v2(api):
    _edit_v2(api, v2(), v2({"web": svc(image="other:1")}))
    out = j(server.edit_configuration(CONFIG, change_reason="swap", spec={"services": {"web": {"image": "other:1"}}}, confirm=True))
    assert set(bodies(api, "PUT", "/spec")[0]["spec"]["services"]) == {"web"}
    assert {"services.app.image", "services.web.image"} <= {c["path"] for c in out["diff"]}
    bad = j(server.edit_configuration(CONFIG, change_reason="swap", spec={"image": "other:1"}, confirm=True))
    assert "version-2 spec" in bad["error"]


def test_edit_configuration_service_without_a_per_service_edit_is_an_error(api):
    _edit_v2(api, v2(two_services()), v2(two_services()))
    out = j(server.edit_configuration(CONFIG, change_reason="x", merge_patch={"desiredState": "STOPPED"}, service="web", confirm=True))
    assert "service selects the target of image" in out["error"]


def test_edit_configuration_legacy_single_image_backend_still_edits(api):
    edit_api(api, after_spec=legacy_spec(image="registry.example.com/kiosk:2.0"))
    out = j(server.edit_configuration(CONFIG, change_reason="x", image_tag="2.0", confirm=True))
    assert bodies(api, "PUT", "/spec")[0]["spec"]["image"].endswith(":2.0") and out["diff"][0]["path"] == "image"


# ------------------------------------------------- read tools on the v2 shape ---


def test_diff_configuration_versions_flattens_services(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(3)))
    old = v2(two_services())
    new_services = two_services()
    new_services["web"] = {**new_services["web"], "image": "registry.example.com/web:1.5", "environment": {"DB_HOST": "db", "SESSION_TOKEN": "changed"}}
    new_services["cache"] = svc(image="docker.example.com/redis:7")
    api.on("GET", f"/configurations/{CONFIG}/versions/2", ok({"version_number": 2, "spec": old}))
    api.on("GET", f"/configurations/{CONFIG}/versions/3", ok({"version_number": 3, "spec": v2(new_services), "change_reason": "add cache"}))

    out = j(server.diff_configuration_versions(CONFIG))

    paths = {c["path"]: c for c in out["changes"]}
    assert paths["services.web.image"]["new"].endswith(":1.5")
    assert paths["services.web.environment.SESSION_TOKEN"]["new"] == "***" and paths["services.web.environment.SESSION_TOKEN"]["old"] == "***"
    assert paths["services.cache.image"]["change"] == "added"
    assert "changed" not in json.dumps([c.get("old") for c in out["changes"]] + [c.get("new") for c in out["changes"]])
    assert out["summary"].startswith("kiosk v2 → v3:") and out["identical"] is False


def test_get_configuration_masks_every_service_and_summarises(api):
    api.on("GET", f"/configurations/{CONFIG}", ok(cfg(7)))
    api.on("GET", f"/configurations/{CONFIG}/details", details(7, v2(two_services())))
    api.on("GET", f"/configurations/{CONFIG}/versions", ok([]))
    api.on("GET", "/fleets", page([]))

    out = j(server.get_configuration(CONFIG))
    assert out["spec"]["services"]["web"]["environment"]["SESSION_TOKEN"] == "***" and "credential-looking" in out["note"].lower()
    plain = j(server.get_configuration(CONFIG, include_spec=False))
    assert plain["spec_summary"]["services"] == {"db": "docker.example.com/postgres:16", "web": "registry.example.com/web:1.4"}
    shown = j(server.get_configuration(CONFIG, show_secret_env=True))
    assert shown["spec"]["services"]["web"]["environment"]["SESSION_TOKEN"] == "abc"


def test_delete_registry_credential_sees_per_service_credentials(monkeypatch):
    from test_alert_credentials_secrets import CONFIG as CFG2
    from test_alert_credentials_secrets import details as d2
    from test_alert_credentials_secrets import ok as ok2
    from test_alert_credentials_secrets import page as page2

    fake = FakeAPI()
    install(monkeypatch, fake)
    fake.on("GET", "/registry-credentials", page2([{"id": CRED, "name": "ghcr", "type": "basic", "registry": "ghcr.io"}]))
    fake.on("GET", f"/registry-credentials/{CRED}", ok2({"id": CRED, "name": "ghcr", "type": "basic", "registry": "ghcr.io"}))
    fake.on("GET", "/configurations", page2([{"id": CFG2, "name": "kiosk"}]))
    fake.on("GET", f"/configurations/{CFG2}/details", d2(the_spec=v2({"web": svc(registryCredentialId=CRED), "db": svc()})))

    out = j(server.delete_registry_credential("ghcr", confirm=True))

    assert out["error"].endswith("nothing was deleted.") and out["configurations"] == [{"id": CFG2, "name": "kiosk"}]
    assert not [r for r in fake.seen if r.method == "DELETE"]


# --------------------------------------------------------- render redaction ---


def test_render_redacts_secret_files_of_the_service_section():
    result = {"bundle": {"workload": {"secretFiles": [{"path": "/a", "content": "QQ=="}]}, "services": {"secretFiles": [{"path": "/b", "content": "Qg=="}], "services": [{"name": "web"}]}, "secrets": [{"path": "/c", "content": "Qw=="}]}}
    shown = json.dumps(server._redact_render(result))
    assert "QQ==" not in shown and "Qg==" not in shown and "Qw==" not in shown and shown.count("[redacted]") == 3


def test_render_note_only_when_neither_workload_nor_services_section(monkeypatch):
    api = FakeAPI()
    install(monkeypatch, api)
    api.on("POST", f"/devices/{DEV}/document:render", ok({"bundle": {"services": {"services": [{"name": "web"}]}}, "revision": "r", "diff": []}))
    assert "note" not in j(server.render_device_document("dan-qemu-3"))
    api.on("POST", f"/devices/{DEV}/document:render", ok({"bundle": {"network": {}}, "revision": "r", "diff": []}))
    assert "no workload section" in j(server.render_device_document("dan-qemu-3"))["note"]


# ------------------------------------------------------ device override preview ---


def test_set_device_override_preview_maps_onto_the_set(api):
    base = v2({"app": svc(image="registry.example.com/kiosk:1.0", environment={"MODE": "prod"})})
    merged = v2({"app": svc(image="registry.example.com/kiosk:1.0", environment={"MODE": "prod"})})
    api.on("GET", f"/devices/{DEV}/configuration", ok({"fleet_id": "f", "configuration_id": CONFIG, "configuration_info": {"name": "kiosk"}, "config_version": 3, "base_config": base, "merged_config": merged, "has_override": False}))

    out = j(server.set_device_configuration_override("dan-qemu-3", {"image": "registry.example.com/kiosk:9", "environment": {"A": "1"}, "volumes": {"d": {"destination": "/d"}}}))

    rows = {c["path"]: c for c in out["preview"]["effective_changes"]}
    assert set(rows) == {"services.app.image", "services.app.environment.A"}
    assert out["preview"]["not_previewed"] == ["volumes"]
    assert "services.app.image" in out["summary"]


def test_get_device_configuration_reads_v2_base_and_merged(api):
    base = v2({"app": svc(environment={"MODE": "prod"})})
    merged = v2({"app": svc(environment={"MODE": "dev", "API_KEY": "k"})})
    api.on("GET", f"/devices/{DEV}/configuration", ok({"fleet_id": "f", "configuration_id": CONFIG, "configuration_info": {"name": "kiosk"}, "config_version": 3, "base_config": base, "merged_config": merged, "has_override": True,
                                                       "configuration_override": {"environment": {"MODE": "dev", "API_KEY": "k"}}, "overridden_paths": ["environment.MODE", "environment.API_KEY"]}))
    out = j(server.get_device_configuration("dan-qemu-3"))
    paths = {c["path"]: c for c in out["overridden_fields"]}
    assert set(paths) == {"services.app.environment.MODE", "services.app.environment.API_KEY"} and paths["services.app.environment.API_KEY"]["new"] == "***"
    assert out["merged"]["services"]["app"]["environment"]["API_KEY"] == "***"


# ------------------------------------------------ explain_workload_failure & co ---


def set_state(**over: Any) -> dict[str, Any]:
    wl = {"state": "CRASHING", "configurationName": "shop", "version": 3, "image": "registry.example.com/web:1.4",
          "services": [
              {"name": "db", "state": "RUNNING", "image": "docker.example.com/postgres:16", "imageDigest": "sha256:db", "restarts": 0, "health": {"status": "healthy", "failingStreak": 0}},
              {"name": "web", "state": "CRASHING", "image": "registry.example.com/web:1.4", "imageDigest": "sha256:web", "restarts": 7, "exitCode": 137, "error": "container exited token=abc123secret",
               "health": {"status": "unhealthy", "failingStreak": 3, "last": "curl: (7) connection refused password=hunter2"}},
          ]}
    rt = {"failureCount": 0, "exitCode": 0, "services": [
        {"name": "db", "containerId": "default-db", "pid": 400, "ip": "10.10.0.4", "failureCount": 0},
        {"name": "web", "containerId": "default-web", "pid": 0, "ip": "10.10.0.5", "failureCount": 7, "exitCode": 137, "oomKilled": True},
    ]}
    return good_state(workload=wl, runtime=rt, **over)


def test_explain_workload_failure_per_service(monkeypatch):
    logs = [
        {"timestamp": "2026-10-07T23:58:00Z", "level": "error", "source": "workload.default-web.stderr", "message": "fatal: out of memory token=abc123secret"},
        {"timestamp": "2026-10-07T23:57:00Z", "level": "info", "source": "workload.default-db.stdout", "message": "checkpoint complete"},
        {"timestamp": "2026-10-07T23:56:00Z", "level": "warn", "source": "workload.default-db.stderr", "message": "slow fsync"},
    ]
    api = build_api(state=set_state(), logs=logs, events=[{"event": "WorkloadCrashed", "timestamp": "2026-10-07T23:58:00Z"}])
    ts_install(monkeypatch, api)

    text = server.explain_workload_failure("kiosk-7")
    out = json.loads(text)

    assert "abc123secret" not in text and "hunter2" not in text
    services = {s["name"]: s for s in out["workload"]["services"]}
    web, db = services["web"], services["db"]
    assert web["restarts"] == 7 and web["exit_code"] == 137 and "SIGKILL" in web["exit_meaning"] and web["oom_killed"] is True
    assert web["health"]["status"] == "unhealthy" and "connection refused" in web["health"]["last"] and "password=[redacted]" in web["health"]["last"]
    assert [e["message"] for e in web["log_excerpts"]] == ["fatal: out of memory token=[redacted]"]
    assert [e["message"] for e in db["log_excerpts"]] == ["checkpoint complete", "slow fsync"] and "exit_code" not in db
    assert {"failure_count", "last_exit_code", "oom_killed"}.isdisjoint(out["workload"])
    assert {e["service"] for e in out["log_excerpts"]} == {"web", "db"}
    crash = by_code(out, "workload_crashing")
    assert crash["finding"] == "Service web is crash-looping" and any("service web: crashing, 7 restarts, last exit code 137" in e for e in crash["evidence"])
    oom = by_code(out, "workload_oom")
    assert "Service web was killed" in oom["finding"]
    assert "Service web fails its healthcheck" in by_code(out, "service_unhealthy")["finding"]
    assert out["summary"].startswith("kiosk-7: Service web is crash-looping")


def test_explain_workload_failure_degraded_set_names_the_stopped_service(monkeypatch):
    st = set_state()
    st["workload"] = {**st["workload"], "state": "DEGRADED", "services": [
        {"name": "db", "state": "RUNNING", "restarts": 0},
        {"name": "migrate", "state": "STOPPED", "restarts": 0, "exitCode": 1},
    ]}
    st["runtime"] = {"services": [{"name": "migrate", "exitCode": 1, "crunState": "stopped"}]}
    api = build_api(state=st)
    ts_install(monkeypatch, api)

    out = run(server.explain_workload_failure, "kiosk-7")

    degraded = by_code(out, "workload_degraded")
    assert degraded["severity"] == "warning" and "migrate is down" in degraded["finding"]
    assert any("service migrate: stopped, last exit code 1: generic application error" in e for e in degraded["evidence"])
    assert "workload_crashing" not in codes(out) and "workload_oom" not in codes(out)
    assert degraded["what_you_can_do"] and degraded["needs_admiral_support"] is False


def test_explain_workload_failure_legacy_single_workload_keeps_singular_fields(monkeypatch):
    st = good_state(workload={"state": "CRASHING", "image": "ghcr.io/acme/kiosk:1"}, runtime={"failureCount": 7, "exitCode": 137, "oomKilled": True, "oomEvents": 3})
    api = build_api(state=st)
    ts_install(monkeypatch, api)
    out = run(server.explain_workload_failure, "kiosk-7")
    assert out["workload"]["failure_count"] == 7 and out["workload"]["last_exit_code"] == 137 and "services" not in out["workload"]
    assert {"workload_crashing", "workload_oom"} <= set(codes(out)) and "service_unhealthy" not in codes(out)


def test_troubleshoot_workload_findings_for_sets_directly():
    ev = {"state_doc": set_state(), "workload": None, "diagnose": None}
    found = ts._workload(ev)
    assert {f["code"] for f in found} == {"workload_crashing", "workload_oom", "service_unhealthy"}
    healthy = {"state_doc": good_state(workload={"state": "RUNNING", "services": [{"name": "web", "state": "RUNNING", "health": {"status": "healthy"}}]}, runtime={"failureCount": 0}), "workload": None, "diagnose": None}
    assert ts._workload(healthy) == []
    stopped = {"state_doc": good_state(workload={"state": "STOPPED", "services": [{"name": "web", "state": "STOPPED"}]}, runtime={"failureCount": 0}), "workload": None, "diagnose": None}
    assert ts._workload(stopped) == []  # a deliberately stopped set is not "degraded"


def test_state_summary_lists_services_for_a_set_and_is_unchanged_otherwise():
    from admrl_mcp.watch import summarise_state

    summary = summarise_state(STATE_DOC)
    assert summary["workload"]["state"] == "DEGRADED"
    assert summary["services"] == [
        {"name": "db", "state": "RUNNING", "restarts": 0, "ip": "10.10.0.4", "health": "healthy"},
        {"name": "web", "state": "STOPPED", "restarts": 2, "ip": "10.10.0.5"},
    ]
    legacy = summarise_state({"workload": {"state": "RUNNING", "image": "a:1"}, "runtime": {"failureCount": 0}})
    assert "services" not in legacy and legacy["workload"] == {"state": "RUNNING", "image": "a:1"}
