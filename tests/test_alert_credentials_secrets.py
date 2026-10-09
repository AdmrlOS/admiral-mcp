"""Alerting, registry credential and secret file tools.

Secret rule: the secret string read from a temp file or an environment variable must never appear in a tool
result, and in the request log only in the single outgoing body that carries it.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from admrl_mcp import server

from test_confirmation import CRED, FILE, RULE, GuardedAPI, world
from test_ops_tools import CONFIG, FLEET, FLEET2, details, ok, page, spec
from test_w1_tools import DEV, install

SECRET = "tok_live_9f8e7d6c5b4a-do-not-print"
FILE_BODY = b"-----BEGIN PRIVATE KEY-----\nTOPSECRETKEYMATERIAL\n"


def j(text: str) -> dict[str, Any]:
    return json.loads(text)


@pytest.fixture
def api(monkeypatch) -> GuardedAPI:
    fake = world()
    install(monkeypatch, fake)
    return fake


def writes(api: GuardedAPI, method: str | None = None) -> list[httpx.Request]:
    return [r for r in api.seen if r.method != "GET" and (method is None or r.method == method)]


def assert_secret_only_in_the_body(api: GuardedAPI, secret: str, outputs: list[str]) -> None:
    for out in outputs:
        assert secret not in out
    carrying = []
    for r in api.seen:
        hay = [str(r.url), json.dumps(dict(r.headers))]
        assert not any(secret in h for h in hay), f"secret in url/headers of {r.method} {r.url.path}"
        if secret.encode() in r.content:
            carrying.append(f"{r.method} {r.url.path}")
    assert len(carrying) == 1, carrying


# ---------------------------------------------------------------- alerting ---


def test_list_alert_rules_filters_by_name_and_reports_counts(api):
    api.on("GET", "/alerting/rules", ok([{"id": RULE, "name": "cpu high", "severity": "warning", "datasource": "metrics", "enabled": True}, {"id": "r2", "name": "disk full", "severity": "critical", "datasource": "metrics", "enabled": False}]))
    api.on("GET", "/alerting/alerts/summary", ok({"firing": 2, "critical": 1, "warning": 1, "info": 0}))
    out = j(server.list_alert_rules(search="CPU"))
    assert [r["name"] for r in out["rules"]] == ["cpu high"] and "2 alert(s) firing" in out["summary"]
    assert len(j(server.list_alert_rules())["rules"]) == 2


def test_get_alert_rule_by_name_includes_evaluation_health(api):
    out = j(server.get_alert_rule("cpu high"))
    assert out["rule"]["expr"] == "up == 0" and out["status"]["health"] == "ok" and "health ok" in out["summary"]


def test_alert_rule_name_ambiguity_returns_candidates(api):
    api.on("GET", "/alerting/rules", ok([{"id": RULE, "name": "cpu high"}, {"id": "r2", "name": "cpu high"}]))
    out = j(server.get_alert_rule("cpu high"))
    assert len(out["candidates"]) == 2 and "UUID" in out["next"]
    out = j(server.delete_alert_rule("cpu high", confirm=True))
    assert len(out["candidates"]) == 2 and not writes(api)


def test_list_alerts_passes_filters_and_validates(api):
    api.on("GET", "/alerting/alerts", ok({"alerts": [{"id": "a1", "alertname": "cpu high", "severity": "warning", "state": "firing", "device_id": DEV, "starts_at": "2026-10-01T00:00:00Z"}], "total": 1, "limit": 50, "offset": 0}))
    out = j(server.list_alerts(state="firing", severity="warning", rule="cpu high", since="2026-10-01T00:00:00Z", limit=9999))
    params = [r for r in api.seen if r.url.path.endswith("/alerting/alerts")][0].url.params
    assert dict(params) == {"state": "firing", "severity": "warning", "rule_id": RULE, "since": "2026-10-01T00:00:00Z", "limit": "500"}
    assert out["alerts"][0]["alertname"] == "cpu high"
    assert "state must be" in j(server.list_alerts(state="open"))["error"]
    assert "severity must be" in j(server.list_alerts(severity="urgent"))["error"]


def test_preview_alert_rule_posts_to_the_read_only_endpoint(api):
    api.on("POST", "/alerting/rules/preview", ok({"status": 200, "series": 3}))
    out = j(server.preview_alert_rule(expr="up == 0", fleets=["shop"]))
    req = writes(api, "POST")[0]
    assert req.url.path.endswith("/alerting/rules/preview") and json.loads(req.content)["fleet_ids"] == [FLEET]
    assert out["read_only"] is True and "3 series" in out["summary"] and "Nothing was saved" in out["summary"]
    assert not api.violations
    api.on("POST", "/alerting/rules/preview", ok({"status": 400, "series": 0, "error": "bad expr"}))
    assert "Query error: bad expr" in j(server.preview_alert_rule(expr="((", datasource="metrics"))["summary"]


def test_create_alert_rule_resolves_scope_and_reads_back(api):
    created = {"id": RULE, "name": "disk", "severity": "critical", "datasource": "metrics", "expr": "disk > 90", "enabled": True}
    api.on("POST", "/alerting/rules", httpx.Response(201, json={"code": 201, "msg": "created", "data": created}))
    api.on("GET", f"/alerting/rules/{RULE}", ok(created))
    out = j(server.create_alert_rule(name="disk", expr="disk > 90", severity="critical", for_seconds=300, fleets=["shop"], devices=["dan-qemu-3"], labels={"team": "ops"}))
    body = json.loads(writes(api, "POST")[0].content)
    assert body == {"name": "disk", "datasource": "metrics", "expr": "disk > 90", "for_seconds": 300, "severity": "critical", "labels": {"team": "ops"}, "fleet_ids": [FLEET], "device_ids": [DEV], "enabled": True, "notify_in_app": True, "notify_webhooks": True}
    assert out["read_back"]["confirmed"] is True and out["rule"]["id"] == RULE


def test_create_alert_rule_needs_the_telemetry_addon_and_validates(api):
    api.on("POST", "/alerting/rules", httpx.Response(402, json={"msg": "telemetry add-on required"}))
    assert j(server.create_alert_rule(name="x", expr="up"))["billing_gate"] is True
    api.on("POST", "/alerting/rules", httpx.Response(400, json={"msg": "a logs rule must aggregate with a stats pipe"}))
    assert "stats pipe" in j(server.create_alert_rule(name="x", expr="error", datasource="logs"))["error"]
    api.on("POST", "/alerting/rules", httpx.Response(409, json={"msg": "rule exists"}))
    assert j(server.create_alert_rule(name="x", expr="up"))["conflict"] is True
    assert "name and expr are required" in j(server.create_alert_rule(name=" ", expr="up"))["error"]


def test_update_alert_rule_resends_the_whole_rule_and_refuses_a_no_op(api):
    current = {"id": RULE, "name": "cpu high", "severity": "warning", "datasource": "metrics", "expr": "up == 0", "enabled": True, "interval_seconds": 60, "for_seconds": 0, "org_id": "o", "created_at": "t"}
    state = dict(current)
    api.on("GET", f"/alerting/rules/{RULE}", lambda r: ok(state))
    api.on("PUT", f"/alerting/rules/{RULE}", lambda r: (state.update(json.loads(r.content)), ok(state))[1])
    out = j(server.update_alert_rule("cpu high", enabled=False, severity="critical"))
    body = json.loads(writes(api, "PUT")[0].content)
    assert body["enabled"] is False and body["severity"] == "critical" and body["expr"] == "up == 0" and body["notify_in_app"] is True
    assert "org_id" not in body and "created_at" not in body
    assert {r["path"] for r in out["diff"]} >= {"enabled", "severity"} and out["read_back"]["confirmed"] is True
    again = j(server.update_alert_rule(RULE, enabled=False))
    assert again["error"].startswith("No change") and len(writes(api, "PUT")) == 1
    assert "Nothing to change" in j(server.update_alert_rule(RULE))["error"]


def test_delete_alert_rule_previews_then_deletes_and_reads_back_404(api):
    preview = j(server.delete_alert_rule("cpu high"))
    assert preview["irreversible"] is True and preview["preview"]["expr"] == "up == 0" and not writes(api)
    gone = {"v": False}
    api.on("DELETE", f"/alerting/rules/{RULE}", lambda r: (gone.update(v=True), ok(None))[1])
    api.on("GET", f"/alerting/rules/{RULE}", lambda r: httpx.Response(404, json={"msg": "Not found"}) if gone["v"] else ok({"id": RULE, "name": "cpu high"}))
    out = j(server.delete_alert_rule(**preview["next"]["arguments"]))
    assert out["deleted"] is True and out["read_back"]["confirmed"] is True


# ------------------------------------------------------ registry credentials ---


def test_credential_views_never_pass_secret_looking_fields_through(api):
    leaky = {"id": CRED, "name": "ghcr", "registry": "registry.example.com", "type": "basic", "username": "svc", "has_password": True, "password": SECRET, "encrypted_password": SECRET, "registry_token": SECRET}
    api.on("GET", "/registry-credentials", page([leaky]))
    api.on("GET", f"/registry-credentials/{CRED}", ok(leaky))
    for out in (server.list_registry_credentials(), server.get_registry_credential("ghcr"), server.get_registry_credential(CRED)):
        assert SECRET not in out
    assert j(server.get_registry_credential(CRED))["credential"]["has_password"] is True


def test_credential_name_ambiguity_returns_candidates(api):
    api.on("GET", "/registry-credentials", page([{"id": CRED, "name": "ghcr"}, {"id": "c2", "name": "ghcr"}]))
    out = j(server.get_registry_credential("ghcr"))
    assert len(out["candidates"]) == 2


def test_create_registry_credential_from_a_file_sends_the_secret_once(api, tmp_path):
    f = tmp_path / "pw.txt"
    f.write_text(SECRET + "\n")
    created = {"id": CRED, "name": "quay", "registry": "quay.example.com", "type": "basic", "username": "svc", "has_password": True}
    api.on("POST", "/registry-credentials", httpx.Response(201, json={"code": 201, "msg": "ok", "data": created}))
    api.on("GET", f"/registry-credentials/{CRED}", ok(created))
    out = server.create_registry_credential(name="quay", registry="quay.example.com", username="svc", secret_file=str(f))
    body = json.loads(writes(api, "POST")[0].content)
    assert body == {"type": "basic", "name": "quay", "registry": "quay.example.com", "password": SECRET, "username": "svc"}  # newline stripped
    assert j(out)["read_back"]["confirmed"] is True and j(out)["credential"]["has_password"] is True
    assert_secret_only_in_the_body(api, SECRET, [out])


def test_create_registry_credential_from_env_and_aws_ecr_field(api, monkeypatch):
    monkeypatch.setenv("QUAY_TOKEN", SECRET)
    created = {"id": CRED, "name": "ecr", "registry": "1.dkr.ecr.eu-west-1.amazonaws.com", "type": "aws_ecr", "has_aws_secret_access_key": True}
    api.on("POST", "/registry-credentials", httpx.Response(201, json={"code": 201, "msg": "ok", "data": created}))
    api.on("GET", f"/registry-credentials/{CRED}", ok(created))
    out = server.create_registry_credential(name="ecr", registry=created["registry"], type="aws_ecr", aws_region="eu-west-1", aws_access_key_id="AKIAEXAMPLE", secret_env="QUAY_TOKEN")
    body = json.loads(writes(api, "POST")[0].content)
    assert body["aws_secret_access_key"] == SECRET and body["type"] == "aws_ecr" and "password" not in body
    assert_secret_only_in_the_body(api, SECRET, [out])
    assert "secret_field" in j(server.create_registry_credential(name="x", registry="r", secret_env="QUAY_TOKEN", secret_field="password", type="aws_ecr"))["error"]


def test_create_registry_credential_secret_source_errors_never_leak(api, tmp_path, monkeypatch):
    f = tmp_path / "pw.txt"
    f.write_text(SECRET)
    both = server.create_registry_credential(name="x", registry="r", secret_file=str(f), secret_env="X")
    neither = server.create_registry_credential(name="x", registry="r")
    assert "exactly one of secret_file" in j(both)["error"] and "exactly one of secret_file" in j(neither)["error"]
    missing = server.create_registry_credential(name="x", registry="r", secret_file=str(tmp_path / "nope.txt"))
    assert "Cannot read" in j(missing)["error"]
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    assert "is empty" in j(server.create_registry_credential(name="x", registry="r", secret_file=str(empty)))["error"]
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert "not set" in j(server.create_registry_credential(name="x", registry="r", secret_env="NOT_SET_ANYWHERE"))["error"]
    assert "Unknown type" in j(server.create_registry_credential(name="x", registry="r", type="oauth", secret_file=str(f)))["error"]
    for out in (both, neither, missing):
        assert SECRET not in out
    assert not writes(api)


def test_create_registry_credential_api_failures_never_echo_the_secret(api, tmp_path):
    f = tmp_path / "pw.txt"
    f.write_text(SECRET)
    api.on("POST", "/registry-credentials", httpx.Response(409, json={"msg": "credential name already exists"}))
    out = server.create_registry_credential(name="x", registry="r", secret_file=str(f))
    assert j(out)["conflict"] is True
    api.on("POST", "/registry-credentials", httpx.Response(403, json={"msg": "Access denied"}))
    out2 = server.create_registry_credential(name="x", registry="r", secret_file=str(f))
    assert "Access denied" in j(out2)["error"] and SECRET not in out + out2


def test_update_registry_credential_preview_reads_the_secret_but_never_prints_it(api, tmp_path):
    f = tmp_path / "new.txt"
    f.write_text(SECRET)
    out = server.update_registry_credential("ghcr", username="svc2", secret_file=str(f), secret_field="registry_token")
    data = j(out)
    assert data["confirmation_required"] is True and not writes(api)
    assert data["preview"]["secret"] == {"replaced": True, "field": "registry_token", "source": "secret_file"}
    assert data["preview"]["field_changes"] == {"username": {"old": "svc", "new": "svc2"}}
    assert data["next"]["arguments"]["secret_file"] == str(f) and data["next"]["arguments"]["credential"] == CRED
    assert SECRET not in out and not any(SECRET.encode() in r.content for r in api.seen)


def test_update_registry_credential_confirmed_sends_only_changes(api, tmp_path):
    f = tmp_path / "new.txt"
    f.write_text(SECRET)
    state = {"has_registry_token": False}
    base = {"id": CRED, "name": "ghcr", "registry": "registry.example.com", "type": "basic", "username": "svc", "has_password": True}
    api.on("GET", f"/registry-credentials/{CRED}", lambda r: ok({**base, **state}))
    api.on("PUT", f"/registry-credentials/{CRED}", lambda r: (state.update(has_registry_token=True), ok({**base, **state}))[1])
    out = server.update_registry_credential(CRED, secret_file=str(f), secret_field="registry_token", confirm=True)
    assert json.loads(writes(api, "PUT")[0].content) == {"registry_token": SECRET}
    assert j(out)["read_back"]["confirmed"] is True
    assert_secret_only_in_the_body(api, SECRET, [out])
    nothing = j(server.update_registry_credential(CRED, username="svc", confirm=True))
    assert nothing["error"].startswith("No change")


def test_delete_registry_credential_refuses_while_a_configuration_uses_it(api):
    api.on("GET", f"/configurations/{CONFIG}/details", details(the_spec=spec(registry_credential_id=CRED)))
    out = j(server.delete_registry_credential("ghcr", confirm=True))
    assert out["error"].endswith("nothing was deleted.") and out["configurations"][0]["id"] == CONFIG
    assert not writes(api)


def test_delete_registry_credential_previews_then_deletes(api):
    preview = j(server.delete_registry_credential("ghcr"))
    assert preview["irreversible"] is True and preview["preview"]["configurations_checked"] == 2 and not writes(api)
    gone = {"v": False}
    cred = {"id": CRED, "name": "ghcr", "registry": "registry.example.com", "type": "basic"}
    api.on("DELETE", f"/registry-credentials/{CRED}", lambda r: (gone.update(v=True), ok(None))[1])
    api.on("GET", f"/registry-credentials/{CRED}", lambda r: httpx.Response(404, json={"msg": "nf"}) if gone["v"] else ok(cred))
    out = j(server.delete_registry_credential(**preview["next"]["arguments"]))
    assert out["deleted"] is True
    api.on("GET", f"/registry-credentials/{CRED}", ok(cred))
    api.on("DELETE", f"/registry-credentials/{CRED}", httpx.Response(409, json={"msg": "credential is in use by 1 configuration version(s) and cannot be deleted"}))
    assert j(server.delete_registry_credential(CRED, confirm=True))["conflict"] is True


# --------------------------------------------------------------- secret files ---


def test_list_secret_files_is_metadata_only(api):
    api.on("GET", f"/devices/{DEV}/secret-files", ok({"enabled": True, "files": [{"id": FILE, "path": "/admrl/secrets/a", "mode": "0600", "uid": 0, "gid": 0, "size": 5, "sha256": "ab" * 32, "updated_at": "t", "apply": {"state": "failed", "error": "read-only"}, "content_base64": "U0VDUkVU", "content": "SECRET"}], "inherited": [{"id": "i1", "path": "/admrl/secrets/b", "size": 3}], "limits": {"max_file_bytes": 65536}}))
    out = server.list_secret_files(device="dan-qemu-3")
    data = j(out)
    assert data["files"][0]["apply"]["state"] == "failed" and data["inherited"][0]["path"] == "/admrl/secrets/b"
    assert "U0VDUkVU" not in out and "SECRET" not in out
    assert "exactly one of configuration or device" in j(server.list_secret_files())["error"]
    assert j(server.list_secret_files(configuration=CONFIG))["target"]["kind"] == "configuration"


def test_delete_secret_file_by_path_and_by_id(api):
    preview = j(server.delete_secret_file("/admrl/secrets/api.key", device="dan-qemu-3"))
    assert preview["irreversible"] is True and preview["next"]["arguments"]["file"] == FILE and not writes(api)
    left = {"files": [{"id": FILE, "path": "/admrl/secrets/api.key"}]}
    api.on("DELETE", f"/devices/{DEV}/secret-files/{FILE}", lambda r: (left.update(files=[]), ok(None))[1])
    api.on("GET", f"/devices/{DEV}/secret-files", lambda r: ok({"enabled": True, "files": left["files"]}))
    out = j(server.delete_secret_file(**preview["next"]["arguments"]))
    assert out["deleted"] is True
    unknown = j(server.delete_secret_file("/admrl/secrets/nope", device=DEV, confirm=True))
    assert unknown["candidates"] == [] and "No secret_file matches" in unknown["error"]


def test_upload_secret_file_preview_pins_the_hash_and_never_prints_content(api, tmp_path):
    src = tmp_path / "tls.key"
    src.write_bytes(FILE_BODY)
    out = server.upload_secret_file(str(src), "/admrl/secrets/tls.key", device="dan-qemu-3")
    data = j(out)
    assert data["confirmation_required"] is True and not writes(api)
    import hashlib

    digest = hashlib.sha256(FILE_BODY).hexdigest()
    assert data["preview"]["upload"] == {"path": "/admrl/secrets/tls.key", "mode": "0600", "uid": 0, "gid": 0, "size": len(FILE_BODY), "sha256": digest}
    assert data["next"]["arguments"]["expected_sha256"] == digest and data["next"]["arguments"]["device"] == DEV
    assert "TOPSECRETKEYMATERIAL" not in out and "PRIVATE KEY" not in out
    assert not any(b"TOPSECRET" in r.content for r in api.seen)


def test_upload_secret_file_confirmed_posts_base64_and_reads_back(api, tmp_path):
    src = tmp_path / "tls.key"
    src.write_bytes(FILE_BODY)
    import hashlib

    digest = hashlib.sha256(FILE_BODY).hexdigest()
    state = {"files": []}
    meta = {"id": "ffffffff-1111-2222-3333-444444444444", "path": "/admrl/secrets/tls.key", "mode": "0640", "uid": 100, "gid": 100, "size": len(FILE_BODY), "sha256": digest}
    api.on("GET", f"/devices/{DEV}/secret-files", lambda r: ok({"enabled": True, "files": state["files"], "limits": {"max_file_bytes": 65536}}))
    api.on("POST", f"/devices/{DEV}/secret-files", lambda r: (state.update(files=[meta]), ok(meta))[1])
    out = server.upload_secret_file(str(src), "/admrl/secrets/tls.key", device="dan-qemu-3", mode="640", uid=100, gid=100, expected_sha256=digest, confirm=True)
    body = json.loads(writes(api, "POST")[0].content)
    assert set(body) == {"path", "mode", "uid", "gid", "content_base64"} and body["mode"] == "0640"
    assert base64.b64decode(body["content_base64"]) == FILE_BODY
    assert j(out)["read_back"]["confirmed"] is True
    b64 = base64.b64encode(FILE_BODY).decode()
    assert "TOPSECRETKEYMATERIAL" not in out and b64 not in out
    # the content travels once, base64 encoded, in the POST body and nowhere else
    carriers = [r for r in api.seen if b64 in r.content.decode(errors="ignore") or "TOPSECRETKEYMATERIAL" in r.content.decode(errors="ignore")]
    assert [r.method for r in carriers] == ["POST"]
    assert not any(b64 in str(r.url) or b64 in json.dumps(dict(r.headers)) for r in api.seen)


def test_upload_secret_file_refuses_a_file_that_changed_since_the_preview(api, tmp_path):
    src = tmp_path / "tls.key"
    src.write_bytes(FILE_BODY)
    out = j(server.upload_secret_file(str(src), "/admrl/secrets/tls.key", device=DEV, expected_sha256="0" * 64, confirm=True))
    assert "changed since the preview" in out["error"] and not writes(api)


def test_upload_secret_file_validates_locally(api, tmp_path):
    src = tmp_path / "f"
    src.write_bytes(b"x")
    for path, text in (("etc/x", "absolute"), ("/etc/passwd", "reserved"), ("/admrl/identity/key", "/admrl/secrets"), ("/admrl/secrets/../x", "canonical"), ("/a/", "absolute path naming a file"), ("/tmp/x", "reserved")):
        assert text in j(server.upload_secret_file(str(src), path, device=DEV))["error"], path
    for mode, text in (("0666", "group or world writable"), ("4755", "setuid"), ("0200", "owner read"), ("9", "octal")):
        assert text in j(server.upload_secret_file(str(src), "/admrl/secrets/x", device=DEV, mode=mode))["error"], mode
    big = tmp_path / "big"
    big.write_bytes(b"a" * (64 * 1024 + 1))
    assert "limit is 65536" in j(server.upload_secret_file(str(big), "/admrl/secrets/x", device=DEV))["error"]
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert "is empty" in j(server.upload_secret_file(str(empty), "/admrl/secrets/x", device=DEV))["error"]
    assert "not a readable file" in j(server.upload_secret_file(str(tmp_path), "/admrl/secrets/x", device=DEV))["error"]
    assert "exactly one of configuration or device" in j(server.upload_secret_file(str(src), "/admrl/secrets/x"))["error"]
    assert not writes(api)


def test_upload_secret_file_refuses_a_no_op_and_the_file_limit(api, tmp_path):
    src = tmp_path / "f"
    src.write_bytes(b"x")
    import hashlib

    same = {"id": FILE, "path": "/admrl/secrets/api.key", "mode": "0600", "uid": 0, "gid": 0, "size": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
    api.on("GET", f"/devices/{DEV}/secret-files", ok({"enabled": True, "files": [same], "limits": {}}))
    assert j(server.upload_secret_file(str(src), "/admrl/secrets/api.key", device=DEV, confirm=True))["error"].startswith("No change")
    full = [{"id": str(i), "path": f"/admrl/secrets/{i}", "size": 1} for i in range(32)]
    api.on("GET", f"/devices/{DEV}/secret-files", ok({"enabled": True, "files": full, "limits": {"max_files_per_scope": 32}}))
    assert "limit 32" in j(server.upload_secret_file(str(src), "/admrl/secrets/new", device=DEV, confirm=True))["error"]
    assert not writes(api)


def test_upload_secret_file_api_errors_413_and_409(api, tmp_path):
    src = tmp_path / "f"
    src.write_bytes(b"x")
    api.on("GET", f"/configurations/{CONFIG}/secret-files", ok({"enabled": True, "files": [], "limits": {}}))
    api.on("POST", f"/configurations/{CONFIG}/secret-files", httpx.Response(409, json={"msg": "secret file limit exceeded"}))
    out = j(server.upload_secret_file(str(src), "/admrl/secrets/x", configuration=CONFIG, confirm=True))
    assert out["conflict"] is True
    api.on("POST", f"/configurations/{CONFIG}/secret-files", httpx.Response(503, json={"msg": "Secret files are disabled on this server"}))
    assert "disabled" in j(server.upload_secret_file(str(src), "/admrl/secrets/x", configuration=CONFIG, confirm=True))["error"]


# ---------------------------------------------------------- stdio-only tools ---


STDIO_ONLY = {"create_registry_credential", "update_registry_credential", "upload_secret_file"}


def test_secret_tools_are_excluded_from_browser_and_hosted_builds():
    import asyncio

    from admrl_mcp import browser, hosted

    assert STDIO_ONLY <= set(browser.EXCLUDED_TOOLS) and STDIO_ONLY <= set(hosted.HOSTED_EXCLUDED_TOOLS)
    stdio = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert STDIO_ONLY <= stdio
    shown = {t.name for t in asyncio.run(browser._server().list_tools())}
    assert not (STDIO_ONLY & shown) and {"list_registry_credentials", "delete_registry_credential", "list_secret_files"} <= shown


def test_secret_tool_schemas_have_no_secret_or_content_parameter():
    import asyncio

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in STDIO_ONLY:
        props = set(tools[name].inputSchema["properties"])
        assert not props & {"password", "secret", "token", "content", "content_base64", "value", "psk_value"}, name
    assert {"secret_file", "secret_env"} <= set(tools["create_registry_credential"].inputSchema["properties"])
    assert "source_path" in tools["upload_secret_file"].inputSchema["properties"]
    for name in ("list_registry_credentials", "get_registry_credential"):
        assert tools[name].annotations.readOnlyHint is True


def test_fleet2_constant_is_used():
    assert FLEET2 != FLEET
