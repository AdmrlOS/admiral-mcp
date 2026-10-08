"""Customer self-service troubleshooting tools: troubleshoot_device, check_device_connectivity,
explain_workload_failure, fleet_health_report. Each test fakes the HTTP API with httpx.MockTransport and
asserts requests (customer routes only) and the ranked findings."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from admrl_mcp import docs, server
from admrl_mcp import troubleshoot as ts
from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings

DEV = "0d9e8f7a-6b5c-4d3e-8f2a-1b0c9d8e7f6a"
FLEET = "11111111-2222-3333-4444-555555555555"
NOW = datetime(2026, 10, 8, 0, 0, 0, tzinfo=timezone.utc)
FIXTURE = Path(__file__).parent / "fixtures" / "docs-search-index.json"

MATCH_ONLINE = {"id": DEV, "name": "kiosk-7", "status": "online", "last_seen": "2026-10-07T23:59:50Z", "fleet": {"id": FLEET, "name": "shops"}}
MATCH_OFFLINE = {**MATCH_ONLINE, "status": "offline", "last_seen": "2026-10-06T10:00:00Z"}


def envelope(data: Any, **extra: Any) -> dict[str, Any]:
    return {"code": 200, "msg": "ok", "data": data, **extra}


class FakeAPI:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Any] = {}
        self.seen: list[httpx.Request] = []

    def on(self, method: str, path: str, response: Any) -> None:
        self.routes[(method, path)] = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        path = request.url.path.removeprefix("/v1")
        route = self.routes.get((request.method, path))
        if route is None:
            return httpx.Response(599, json={"msg": f"unexpected {request.method} {path}"})
        return route(request) if callable(route) else route

    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.path.removeprefix('/v1')}" for r in self.seen]

    def count(self, suffix: str) -> int:
        return sum(1 for p in self.paths() if p.endswith(suffix))


def good_state(**over: Any) -> dict[str, Any]:
    st: dict[str, Any] = {
        "observedGeneration": 5,
        "renderedRevision": "rev-5",
        "conditions": [
            {"type": "Converged", "status": "True"},
            {"type": "WorkloadReady", "status": "True"},
            {"type": "StorageOK", "status": "True"},
            {"type": "TimeSynced", "status": "True"},
        ],
        "workload": {"state": "RUNNING", "configurationName": "kiosk", "image": "ghcr.io/acme/kiosk:1", "version": 4},
        "runtime": {"failureCount": 0, "assignedConfigApplied": True},
        "storage": [{"mount": "/admrl", "totalBytes": 32_000_000_000, "usedBytes": 4_000_000_000, "level": "ok"}],
        "time": {"synced": True, "offsetMs": 12, "rtcDriftMs": 3, "authSkewBudgetMs": 5000, "stratum": 2},
        "transport": {"kind": "tcp", "endpoint": "secret-host.example:4222", "proxy": "http://user:pw@proxy:3128", "latencyMs": 40,
                      "reconnects": 0, "instability": 0, "jetStreamConnected": True, "lastBackendContact": "2026-10-07T23:59:55Z"},
        "network": {"interfaces": [{"name": "eth0", "state": "UP", "ips": [{"address": "10.0.0.5", "version": "v4"}],
                                    "clientState": {"ssid": "SECRET-SSID"}}]},
        "bootReasons": [],
    }
    st.update(over)
    return st


def state_payload(st: dict[str, Any], source: str = "stored") -> dict[str, Any]:
    return envelope({"deviceId": DEV, "state": st, "receivedAt": "2026-10-07T23:59:55Z", "source": source, "protocol": 1})


def healthy_diagnosis(online: bool = True, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "deviceId": DEV,
        "verdict": {"online": online, "workloadState": "RUNNING", "healthScore": 98, "topIssue": (issues or [{}])[0].get("code") if issues else None},
        "issues": issues or [],
        "signals": {},
        "schedule": None,
    }


def probe_report(*items: dict[str, Any], section: str = "network") -> dict[str, Any]:
    return {"generatedAt": "2026-10-07T23:59:58Z", "durationMs": 900, "sections": [{"key": section, "title": section.title(), "items": list(items)}]}


def build_api(*, online: bool = True, state: dict[str, Any] | None = None, diagnosis: dict[str, Any] | None = None,
              probe: Any = None, logs: list[dict[str, Any]] | None = None, events: list[dict[str, Any]] | None = None,
              document_generation: int = 5) -> FakeAPI:
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/diagnose", httpx.Response(200, json=envelope(diagnosis or healthy_diagnosis(online))))
    api.on("GET", f"/devices/{DEV}/state", lambda r: httpx.Response(200, json=state_payload(state or good_state(), "live" if r.url.params.get("live") else "stored")))
    api.on("GET", f"/devices/{DEV}/workload", httpx.Response(200, json=envelope({"state": "RUNNING", "running": {"configurationName": "kiosk", "image": "ghcr.io/acme/kiosk:1", "version": 4}})))
    api.on("GET", f"/devices/{DEV}/events", httpx.Response(200, json=envelope({"events": events or []})))
    api.on("POST", "/metrics/logs/query", httpx.Response(200, json=envelope({"logs": logs or [], "has_more": False})))
    api.on("GET", f"/devices/{DEV}/document", httpx.Response(200, json={"apiVersion": "admrl.co/v1", "metadata": {"generation": document_generation}, "status": {"observedGeneration": 5}}))
    api.on("POST", f"/devices/{DEV}/document:render", httpx.Response(200, json=envelope({"revision": "rev-6", "converged": False, "pushed": False, "diff": [{"path": "spec.workload.override.image"}, {"path": "spec.network.wifi"}]})))
    api.on("POST", f"/devices/{DEV}/diagnostics/probe", httpx.Response(200, json=envelope(probe if probe is not None else probe_report({"key": "dns.resolve", "label": "DNS", "value": "ok", "level": "ok"}))))
    api.on("GET", f"/devices/{DEV}/system-services", httpx.Response(200, json=envelope({"services": [{"name": "dhcpcd", "health": "stable"}]})))
    api.on("GET", f"/devices/{DEV}/device-stats", httpx.Response(200, json=envelope({"cpu": {"usage_percent": 10}, "memory": {"usage_percent": 40}, "disk": {"usage_percent": 12}})))
    api.on("GET", f"/devices/{DEV}", httpx.Response(200, json=envelope({"id": DEV, "name": "kiosk-7", "systemSpec": {"network": [{"name": "eth0", "ipV4Address": "10.0.0.5"}]}})))
    return api


def install(monkeypatch, api: FakeAPI, match: dict[str, Any] | None = None, fake_docs: bool = True) -> AdmiralClient:
    settings = Settings(api_base="https://api.test/v1", token_id="tok", secret_key="sec", org_id="org-1")
    client = AdmiralClient(settings=settings, transport=httpx.MockTransport(api))
    monkeypatch.setattr(server, "get_client", lambda: client)
    resolved = {"match": dict(match or MATCH_ONLINE), "candidates": [], "organization_id": "org-1"}
    monkeypatch.setattr(server, "_resolve_device", lambda q, organization_id=None, fleet_id=None: dict(resolved))
    monkeypatch.setattr(ts, "_now", lambda: NOW)
    ts.clear_probe_cache()
    if fake_docs:
        index = docs.build_index(json.loads(FIXTURE.read_text()), "https://docs.admrl.co")
        monkeypatch.setattr(docs, "get_index", lambda: index)
    return client


def codes(out: dict[str, Any]) -> list[str]:
    return [f["code"] for f in out["findings"]]


def by_code(out: dict[str, Any], code: str) -> dict[str, Any]:
    return next(f for f in out["findings"] if f["code"] == code)


def run(fn: Callable[..., str], *args: Any, **kwargs: Any) -> dict[str, Any]:
    return json.loads(fn(*args, **kwargs))


# --------------------------------------------------------------- healthy ---


def test_healthy_device_reports_no_findings_and_uses_only_customer_routes(monkeypatch):
    api = build_api()
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7")

    assert out["summary"].startswith("kiosk-7 looks healthy")
    assert out["overall"] == "healthy" and out["findings"] == [] and out["online"] is True
    assert out["support"]["needed"] is False
    assert out["state"]["source"] == "live"
    assert api.count("/diagnostics/probe") == 1 and api.count("/system-services") == 1
    assert not [p for p in api.paths() if "/admin" in p]
    # every request is a plain GET, or the probe/log-query/dry-run render POSTs
    posts = {p for p in api.paths() if p.startswith("POST")}
    assert posts <= {"POST /metrics/logs/query", f"POST /devices/{DEV}/diagnostics/probe"}


def test_summary_is_first_key_and_secrets_never_leak(monkeypatch):
    logs = [{"timestamp": "2026-10-07T23:00:00Z", "level": "error", "source": "workload", "message": "login failed Authorization: Bearer abcdef0123456789abcdef password=hunter2"}]
    probe = probe_report({"key": "net.wifi.ssid", "label": "SSID", "value": "HOME-WIFI", "level": "warn", "sensitive": True},
                         {"key": "boot.cmdline", "label": "cmdline", "value": "[redacted]", "level": "error"})
    api = build_api(logs=logs, probe=probe)
    install(monkeypatch, api)

    text = server.troubleshoot_device("kiosk-7")

    assert list(json.loads(text))[0] == "summary"
    for secret in ("hunter2", "abcdef0123456789abcdef", "secret-host.example", "user:pw", "SECRET-SSID", "HOME-WIFI", "cmdline"):
        assert secret not in text


# --------------------------------------------------------------- offline ---


def test_offline_device_explains_last_seen_and_skips_live_calls(monkeypatch):
    diag = healthy_diagnosis(False, [{"code": "device_offline", "severity": "critical", "title": "Device offline", "detail": "no check-in for 37h", "evidence": []}])
    diag["schedule"] = {"overdue": True, "restart": {"at": "2026-10-08T04:00:00Z", "reason": "deadman", "thenEveryMs": 3_600_000}}
    st = good_state(transport={"kind": "wss", "latencyMs": 400}, bootReasons=[{"kind": "watchdog_no_backend", "at": "2026-10-05T00:00:00Z"}])
    api = build_api(online=False, diagnosis=diag, state=st)
    install(monkeypatch, api, MATCH_OFFLINE)

    out = run(server.troubleshoot_device, "kiosk-7", symptom="it went offline")

    assert out["overall"] == "offline" and out["online"] is False
    top = out["findings"][0]
    assert top["code"] == "device_offline" and top["severity"] == "critical" and top["area"] == "connectivity"
    assert any("last seen 2026-10-06T10:00:00Z" in e for e in top["evidence"])
    assert "WebSocket fallback" in " ".join(top["evidence"]) and "watchdog" in top["likely_cause"]
    assert top["relevant_to_symptom"] is True
    assert top["docs"]["url"].startswith("https://docs.admrl.co/getting-started/pairing")
    assert out["offline"]["can_still_read"] and "common_causes" in out["offline"]
    assert api.count("/diagnostics/probe") == 0 and api.count("/system-services") == 0
    assert not any(r.url.params.get("live") for r in api.seen)
    assert out["checked"]["probe"].startswith("device is offline")


# ---------------------------------------------------------- crash looping ---


def crash_state() -> dict[str, Any]:
    return good_state(
        workload={"state": "CRASHING", "configurationName": "kiosk", "image": "ghcr.io/acme/kiosk:1", "version": 4, "error": "container exited"},
        runtime={"failureCount": 7, "exitCode": 137, "oomKilled": True, "oomEvents": 3, "lastExitAt": "2026-10-07T23:58:00Z"},
    )


def test_crash_loop_ranks_critical_workload_findings_first(monkeypatch):
    logs = [{"timestamp": "2026-10-07T23:58:00Z", "level": "error", "source": "workload", "message": "fatal: out of memory allocating 0x7f12ab"}] * 6
    events = [{"event": "WorkloadCrashed", "timestamp": "2026-10-07T23:58:00Z"}]
    api = build_api(state=crash_state(), logs=logs, events=events,
                    diagnosis=healthy_diagnosis(True, [{"code": "workload_crashing", "severity": "critical", "title": "Workload is crash-looping", "detail": "7 failures", "evidence": ["exit 137"]}]))
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7", symptom="keeps restarting")

    assert out["overall"] == "critical"
    assert codes(out)[:2] == ["workload_crashing", "workload_oom"]
    crash = by_code(out, "workload_crashing")
    assert any("exit code 137" in e and "SIGKILL" in e for e in crash["evidence"]) and crash["what_you_can_do"]
    assert crash["severity"] == "critical" and crash["needs_admiral_support"] is False
    assert "log_errors" in codes(out) and "notable_events" in codes(out)


def test_explain_workload_failure_returns_exit_meaning_and_scrubbed_excerpts(monkeypatch):
    logs = [{"timestamp": "2026-10-07T23:58:00Z", "level": "error", "source": "workload", "message": "db connect failed token=abc123secret host=db"}]
    events = [{"event": "WorkloadCrashed", "timestamp": "2026-10-07T23:58:00Z"}, {"event": "BootComplete", "timestamp": "2026-10-07T20:00:00Z"}]
    api = build_api(state=crash_state(), logs=logs, events=events)
    install(monkeypatch, api)

    text = server.explain_workload_failure("kiosk-7")
    out = json.loads(text)

    assert "abc123secret" not in text
    assert out["workload"]["last_exit_code"] == 137 and "SIGKILL" in out["workload"]["last_exit_meaning"]
    assert out["workload"]["failure_count"] == 7 and out["workload"]["image"] == "ghcr.io/acme/kiosk:1"
    assert {"workload_crashing", "workload_oom"} <= set(codes(out))
    assert out["log_excerpts"][0]["message"].endswith("host=db") and "token=[redacted]" in out["log_excerpts"][0]["message"]
    assert [e["event"] for e in out["restart_events"]] == ["WorkloadCrashed"]
    assert api.count("/diagnostics/probe") == 0  # not a probe tool


def test_explain_workload_failure_image_pull_signature_and_usb(monkeypatch):
    st = good_state(
        workload={"state": "STAGING", "signature": {"required": True, "verified": False, "error": "no matching signature", "digest": "sha256:abc"}},
        progress={"operations": [{"kind": "image_pull", "target": "ghcr.io/acme/kiosk:2", "phase": "failed", "error": "unauthorized: authentication required", "attempt": 3}]},
    )
    logs = [{"timestamp": "2026-10-07T23:00:00Z", "level": "warn", "source": "usb", "message": "usb device 1d6b:0002 blocked by policy"}]
    api = build_api(state=st, logs=logs)
    install(monkeypatch, api)

    out = run(server.explain_workload_failure, "kiosk-7")

    assert {"image_signature", "image_pull_failed", "usb_denied"} <= set(codes(out))
    pull = by_code(out, "image_pull_failed")
    assert any("unauthorized" in e for e in pull["evidence"]) and pull["severity"] == "critical"
    assert any("no matching signature" in e for e in by_code(out, "image_signature")["evidence"])


# ----------------------------------------------------------------- storage ---


def test_disk_full_is_critical_with_percentage(monkeypatch):
    st = good_state(storage=[
        {"mount": "/admrl", "totalBytes": 32_000_000_000, "usedBytes": 31_500_000_000, "level": "error"},
        {"mount": "/app", "totalBytes": 8_000_000_000, "usedBytes": 4_000_000_000, "level": "ok", "readOnly": True, "expected": "rw"},
    ])
    api = build_api(state=st)
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7", symptom="disk is full")

    disk = by_code(out, "disk_level")
    assert disk["severity"] == "critical" and "98.4% used" in disk["evidence"][0]
    mount = by_code(out, "mount_mode")
    assert mount["needs_admiral_support"] is True and out["support"]["needed"] is True
    assert disk["relevant_to_symptom"] is True


# -------------------------------------------------------------- clock skew ---


def test_clock_skew_beyond_auth_budget_is_critical(monkeypatch):
    st = good_state(time={"synced": False, "offsetMs": 9000, "rtcDriftMs": 20000, "authSkewBudgetMs": 5000, "stratum": 16})
    api = build_api(state=st)
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7")

    skew = by_code(out, "clock_skew")
    assert skew["severity"] == "critical" and "9000 ms" in skew["evidence"][0] and "5000 ms" in skew["evidence"][0]
    assert "rtc_drift" in codes(out) and "time_not_synced" not in codes(out)

    conn = run(server.check_device_connectivity, "kiosk-7")
    clock = next(c for c in conn["checks"] if c["check"] == "clock")
    assert clock["status"] == "error"
    assert any("Clock skew" in c for c in conn["suspected_causes"])


# --------------------------------------------------------------- dns failure ---


def test_dns_failure_from_probe_in_connectivity_check(monkeypatch):
    probe = probe_report(
        {"key": "dns.resolve", "label": "DNS resolve", "value": "SERVFAIL for api host", "level": "error", "hint": "Check the DHCP DNS server"},
        {"key": "tcp.latency.avg", "label": "TCP latency", "value": "35 ms", "level": "ok"},
    )
    api = build_api(probe=probe)
    install(monkeypatch, api)

    out = run(server.check_device_connectivity, "kiosk-7")

    assert next(c for c in out["checks"] if c["check"] == "dns")["status"] == "error"
    assert next(c for c in out["checks"] if c["check"] == "tcp")["status"] == "ok"
    dns = by_code(out, "dns_failure")
    assert dns["severity"] == "critical" and dns["what_you_can_do"][0] == "Check the DHCP DNS server"
    assert any("DNS" in c for c in out["suspected_causes"])
    assert out["interfaces"] and out["interfaces"][0]["ip"] == "10.0.0.5"
    body = json.loads(next(r for r in api.seen if r.url.path.endswith("/diagnostics/probe")).content)
    assert body == {"sections": ["network", "time"]}
    assert "captive portal" in " ".join(out["classic_causes"]).lower()


def test_connectivity_offline_shows_last_known_transport(monkeypatch):
    api = build_api(online=False, state=good_state(transport={"kind": "wss", "latencyMs": 800, "reconnects": 12, "instability": 3}))
    install(monkeypatch, api, MATCH_OFFLINE)

    out = run(server.check_device_connectivity, "kiosk-7")

    assert out["online"] is False and "offline" in out["summary"]
    assert api.count("/diagnostics/probe") == 0
    assert next(c for c in out["checks"] if c["check"] == "dns")["status"] == "unknown"
    assert next(c for c in out["checks"] if c["check"] == "transport")["status"] == "warn"


# -------------------------------------------------------------------- drift ---


def test_desired_vs_observed_drift_uses_dry_run_render(monkeypatch):
    st = good_state(
        observedGeneration=3,
        conditions=[{"type": "Converged", "status": "False", "reason": "Pending"}],
        runtime={"assignedConfigApplied": False},
        workload={"state": "RUNNING", "transition": {"state": "downloading", "message": "Downloading layer 2/5"}},
    )
    api = build_api(state=st, document_generation=5)
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7")

    drift = by_code(out, "config_drift")
    joined = " ".join(drift["evidence"])
    assert "generation 5" in joined and "applied 3" in joined and "layer 2/5" in joined
    renders = [r for r in api.seen if r.url.path.endswith("document:render")]
    assert len(renders) == 1 and renders[0].url.params["dryRun"] == "1"
    assert "spec.workload.override.image" in " ".join(drift["evidence"]) or any("pending changes" in e for e in drift["evidence"])


def test_no_render_call_when_converged(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    run(server.troubleshoot_device, "kiosk-7")
    assert api.count("document:render") == 0


# ---------------------------------------------------- probe rate limiting ---


def test_probe_429_reuses_previous_result_and_says_so(monkeypatch):
    probe = probe_report({"key": "dns.resolve", "label": "DNS", "value": "SERVFAIL", "level": "error"})
    api = build_api(probe=probe)
    install(monkeypatch, api)
    first = run(server.check_device_connectivity, "kiosk-7")
    assert next(c for c in first["checks"] if c["check"] == "dns")["status"] == "error"
    assert first["checked"]["probe"] == "fresh probe"

    api.on("POST", f"/devices/{DEV}/diagnostics/probe", httpx.Response(429, headers={"Retry-After": "4"}, json={"error": "rate_limited"}))
    second = run(server.check_device_connectivity, "kiosk-7")

    assert "previous result" in second["checked"]["probe"] and "rate-limited" in second["checked"]["probe"]
    assert next(c for c in second["checks"] if c["check"] == "dns")["status"] == "error"


def test_probe_429_without_earlier_result_degrades_gracefully(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    api.on("POST", f"/devices/{DEV}/diagnostics/probe", httpx.Response(429, headers={"Retry-After": "4"}, json={"error": "rate_limited"}))

    out = run(server.troubleshoot_device, "kiosk-7")

    assert "no earlier result" in out["checked"]["probe"]
    assert out["overall"] == "healthy"


def test_live_state_503_falls_back_to_stored_copy(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    api.on("GET", f"/devices/{DEV}/state", lambda r: httpx.Response(503, json={"error": "device_offline"}) if r.url.params.get("live") else httpx.Response(200, json=state_payload(good_state())))

    out = run(server.troubleshoot_device, "kiosk-7")

    assert out["state"]["source"] == "stored" and "live_state" in out["checked"]
    assert "using the stored copy" in out["checked"]["live_state"]


def test_telemetry_402_is_reported_not_fatal(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    api.on("POST", "/metrics/logs/query", httpx.Response(402, json={"msg": "requires the Telemetry add-on"}))

    out = run(server.troubleshoot_device, "kiosk-7")

    assert "telemetry_gated" in codes(out) and by_code(out, "telemetry_gated")["severity"] == "info"
    assert "Telemetry add-on" in out["checked"]["logs"]
    assert out["overall"] == "healthy"


# -------------------------------------------------------------- docs links ---


def test_docs_link_comes_from_search_docs_index_and_failure_is_not_fatal(monkeypatch):
    api = build_api(online=False, diagnosis=healthy_diagnosis(False, [{"code": "device_offline", "severity": "critical", "title": "Offline", "detail": "", "evidence": []}]))
    install(monkeypatch, api, MATCH_OFFLINE)
    ok = run(server.troubleshoot_device, "kiosk-7")
    assert ok["findings"][0]["docs"]["url"] == "https://docs.admrl.co/getting-started/pairing#device-stalled-or-offline"

    def boom():
        raise docs.DocsError("index down")

    monkeypatch.setattr(docs, "get_index", boom)
    bad = run(server.troubleshoot_device, "kiosk-7")
    assert bad["findings"][0]["docs"] is None and "docs lookup unavailable" in bad["docs_note"]


def test_attach_docs_uses_custom_searcher_and_caches_queries():
    calls: list[str] = []

    def fake(q: str) -> list[dict[str, Any]]:
        calls.append(q)
        return [{"title": "T", "url": "https://docs.admrl.co/x"}]

    items = [ts.finding("clock_skew", "critical", "a"), ts.finding("time_not_synced", "warning", "b")]
    assert ts.attach_docs(items, fake) is None
    assert items[0]["docs"] == {"title": "T", "url": "https://docs.admrl.co/x"}
    assert len(calls) == 1  # same docs query for both time findings


def test_scrub_masks_secrets_and_bounds_length():
    assert "[redacted]" in ts.scrub("psk=supersecret")
    assert "supersecret" not in ts.scrub("wifi password: supersecret")
    assert ts.scrub("Authorization: Bearer abcdefghijklmnop") == "Authorization=[redacted]"
    assert len(ts.scrub("x" * 500)) == 160


# ------------------------------------------------------------ fleet report ---


def fleet_api(n_online: int = 3, n_offline: int = 4, pages: bool = False) -> FakeAPI:
    api = FakeAPI()
    devices = [{"id": f"on-{i}", "name": f"on-{i}", "status": "online", "isOnline": {"isOnline": True}} for i in range(n_online)]
    devices += [{"id": f"off-{i}", "name": f"off-{i}", "status": "offline", "isOnline": {"isOnline": False, "lastSeen": "2026-10-07T20:00:00Z"}} for i in range(n_offline)]
    api.on("GET", "/fleets", httpx.Response(200, json=envelope([{"id": FLEET, "name": "shops"}])))
    api.on("GET", "/devices", httpx.Response(200, json=envelope(devices, pagination={"page": 1, "limit": 100, "total": len(devices), "totalPages": 1})))

    def diag(request: httpx.Request) -> httpx.Response:
        did = request.url.path.split("/")[-2]
        if did.startswith("off-"):
            return httpx.Response(200, json=envelope(healthy_diagnosis(False, [{"code": "device_offline", "severity": "critical", "title": "x", "detail": "", "evidence": []}])))
        if did == "on-0":
            d = healthy_diagnosis(True, [{"code": "disk_level", "severity": "warning", "title": "x", "detail": "", "evidence": []}])
            d["verdict"]["healthScore"] = 55
            return httpx.Response(200, json=envelope(d))
        return httpx.Response(200, json=envelope(healthy_diagnosis(True)))

    for d in devices:
        api.on("GET", f"/devices/{d['id']}/diagnose", diag)
    return api


def test_fleet_health_report_groups_by_problem(monkeypatch):
    api = fleet_api()
    install(monkeypatch, api)

    out = run(server.fleet_health_report, "shops")

    assert out["status_counts"] == {"online": 3, "offline": 4}
    assert out["diagnosed"] == 7 and out["not_diagnosed"] == 0 and out["healthy_among_diagnosed"] == 2
    top = out["problems"][0]
    assert top["code"] == "device_offline" and top["devices"] == 4 and top["severity"] == "critical"
    assert top["docs"]["url"].startswith("https://docs.admrl.co/") and top["first_step"]
    assert out["problems"][1]["code"] == "disk_level"
    assert out["not_online_by_last_seen"] == {"< 24 h": 4}
    assert [o["top_issue"] for o in out["worst_offenders"]][:1] == ["device_offline"]
    assert out["fleet"]["id"] == FLEET
    assert next(r for r in api.seen if r.url.path.endswith("/devices")).url.params["fleet_id"] == FLEET


def test_fleet_health_report_is_bounded_and_aggregated(monkeypatch):
    api = fleet_api(n_online=30, n_offline=20)
    install(monkeypatch, api)

    out = run(server.fleet_health_report, None, 5)

    assert out["diagnosed"] == 5 and out["not_diagnosed"] == 45 and "sample_limit" in out["note"]
    assert api.count("/diagnose") == 5
    # not-online devices are diagnosed first
    assert out["problems"][0]["code"] == "device_offline" and out["problems"][0]["devices"] == 5
    assert len(out["worst_offenders"]) <= 10
    assert all(len(p["examples"]) <= 5 for p in out["problems"])


# -------------------------------------------------------- annotations / browser ---

NEW_TOOLS = {"troubleshoot_device", "check_device_connectivity", "explain_workload_failure", "fleet_health_report"}


def test_new_tools_are_registered_read_only():
    import asyncio

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in NEW_TOOLS:
        assert tools[name].annotations is not None and tools[name].annotations.readOnlyHint is True


def test_new_tools_are_offered_in_the_browser_runtime():
    from admrl_mcp import browser
    import asyncio

    assert not NEW_TOOLS & set(browser.EXCLUDED_TOOLS)
    names = {t.name for t in asyncio.run(browser.build_browser_server().list_tools())}
    assert NEW_TOOLS <= names


def test_new_tools_work_sequentially_without_threads(monkeypatch):
    """Pyodide has no threads: the tools must go through the sequential _parallel_map fallback."""
    api = build_api()
    install(monkeypatch, api)
    monkeypatch.setattr(sys, "platform", "emscripten")

    def no_threads(*a, **k):
        raise AssertionError("threads are not available in the browser")

    monkeypatch.setattr(server, "ThreadPoolExecutor", no_threads)
    assert run(server.troubleshoot_device, "kiosk-7")["overall"] == "healthy"
    assert run(server.check_device_connectivity, "kiosk-7")["online"] is True
    assert "workload" in run(server.explain_workload_failure, "kiosk-7")
    install(monkeypatch, fleet_api())
    assert run(server.fleet_health_report, None)["diagnosed"] == 7


# ------------------------------------------------------------ usb / services ---

_BENIGN_USB = [
    "[USB] USB policy applied",
    "[USB] Stored USB policy enforced",
    "[USB] USB policy pull failed",
    "USB policy: failed to protect sysfs path",
    "[USB] Rejecting USB policy",
    "[USB] Policy enforcement incomplete",
]


@pytest.mark.parametrize("suffix", ["", " mode=off allow_rules=0"])
@pytest.mark.parametrize("line", _BENIGN_USB)
def test_usb_policy_info_lines_are_not_usb_denied(line, suffix):
    out = ts._logs({"logs": {"logs": [{"level": "info", "source": "admiral-init", "message": line + suffix}]}})
    assert "usb_denied" not in [f["code"] for f in out]


def test_usb_blocked_interface_is_usb_denied():
    msg = "[USB] Blocked interface interface=1-1:1.0 class=mass-storage"
    out = ts._logs({"logs": {"logs": [{"level": "warn", "source": "admiral-init", "message": msg}]}})
    assert "usb_denied" in [f["code"] for f in out]


def test_unavailable_service_health_is_not_flagged():
    ev = {"services": {"services": [
        {"name": "wpa_supplicant", "health": "unavailable", "normallyUp": True, "reason": "no Wi-Fi hardware"},
        {"name": "bluetoothd", "health": "unavailable", "normallyUp": False},
    ]}}
    assert ts._services(ev) == []
