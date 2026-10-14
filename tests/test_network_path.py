"""Packet loss / path quality: probe network items and the backend networkVerdict surfaced by probe_device,
troubleshoot_device, check_device_connectivity and diagnose_device (additive network_path blocks)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from admrl_mcp import server
from admrl_mcp import troubleshoot as ts
from test_troubleshoot import DEV, MATCH_OFFLINE, MATCH_ONLINE, build_api, by_code, codes, envelope, install, probe_report, run

CLEAN = "192.0.2.1: 20/20 replies, 0% loss, rtt min/avg/max 0.4/0.6/1.1 ms, jitter 0.1 ms"


def dns_lossy_probe(verdict: dict[str, Any] | None = None) -> dict[str, Any]:
    """The reported scenario: everything healthy except the DNS lookup and loss to the DNS server."""
    rep = probe_report(
        {"key": "network.gateway", "label": "Gateway", "value": "192.0.2.1 via eth0", "level": "ok"},
        {"key": "network.dns_servers", "label": "DNS servers", "value": "192.0.2.53", "level": "ok"},
        {"key": "network.dns_lookup", "label": "DNS lookup", "value": "lookup api.example.com: i/o timeout", "level": "error", "hint": "Check the DNS server"},
        {"key": "network.cloud_tcp_443", "label": "Cloud TCP 443", "value": "38 ms", "level": "ok"},
        {"key": "network.loss_gateway", "label": "Loss to gateway", "value": CLEAN, "level": "ok"},
        {"key": "network.loss_dns", "label": "Loss to DNS server", "value": "192.0.2.53: 13/20 replies, 35% loss, rtt min/avg/max 1/9/80 ms, jitter 12 ms", "level": "error"},
        {"key": "network.loss_cloud", "label": "Loss to cloud", "value": "203.0.113.10: 20/20 replies, 0% loss", "level": "ok"},
        {"key": "network.hop_1", "label": "Hop 1", "value": "[redacted]", "level": "ok", "sensitive": True},
        {"key": "network.iface_errors", "label": "Interface errors", "value": "no interface errors", "level": "ok"},
    )
    if verdict is not None:
        rep["networkVerdict"] = verdict
    return rep


DNS_VERDICT = {
    "code": "dns_path_lossy",
    "cause": "DNS server path lossy: lookups fail because packets to the DNS server are lost",
    "evidence": ["network.dns_lookup: lookup api.example.com: i/o timeout (error)", "network.loss_dns: 35% loss (error)"],
    "confidence": "high",
}

UPSTREAM_VERDICT = {
    "code": "upstream",
    "cause": "upstream of the gateway (ISP/WAN): the local link is clean but packets are lost further out",
    "evidence": ["network.loss_gateway: 0% loss (ok)", "network.loss_cloud: 20% loss (error)", "network.hop_3: 20% loss (warn)"],
    "confidence": "high",
}


def upstream_probe() -> dict[str, Any]:
    rep = probe_report(
        {"key": "network.dns_lookup", "label": "DNS lookup", "value": "ok 12 ms", "level": "ok"},
        {"key": "network.loss_gateway", "label": "Loss to gateway", "value": CLEAN, "level": "ok"},
        {"key": "network.loss_cloud", "label": "Loss to cloud", "value": "203.0.113.10: 16/20 replies, 20% loss", "level": "error", "hint": "Loss starts upstream of the gateway"},
        {"key": "network.hop_3", "label": "Hop 3", "value": "[redacted]", "level": "warn", "sensitive": True},
        {"key": "network.path_verdict", "label": "Path", "value": "loss begins at hop 3 and carries through", "level": "warn"},
    )
    rep["networkVerdict"] = UPSTREAM_VERDICT
    return rep


# ------------------------------------------------------------------ helpers ---


def test_network_path_none_for_older_agents_and_backends():
    assert ts.network_path(None) is None
    assert ts.network_path("nope") is None
    assert ts.network_path(probe_report({"key": "network.dns_lookup", "label": "DNS", "value": "timeout", "level": "error"})) is None
    # verdict without a cause is ignored
    rep = probe_report({"key": "network.gateway", "value": "x", "level": "ok"})
    rep["networkVerdict"] = {"code": "upstream", "evidence": []}
    assert ts.network_path(rep) is None


def test_network_path_lists_loss_items_and_verdict_without_sensitive_rows():
    path = ts.network_path(dns_lossy_probe(DNS_VERDICT))
    assert path is not None
    keys = [i["key"] for i in path["items"]]
    assert keys == ["network.loss_gateway", "network.loss_dns", "network.loss_cloud", "network.iface_errors"]
    assert path["worst"] == "error"
    assert path["verdict"]["code"] == "dns_path_lossy" and path["verdict"]["confidence"] == "high"
    assert "network.hop_1" not in json.dumps(path)
    # items only, no verdict (older backend)
    only = ts.network_path(dns_lossy_probe())
    assert only is not None
    assert "verdict" not in only and only["worst"] == "error"


def test_network_path_includes_non_sensitive_hops():
    """Current agents send hop rows without the sensitive flag (customer's own path): they are listed.
    Older agents still flag them sensitive and those stay dropped."""
    rep = probe_report(
        {"key": "network.loss_gateway", "label": "Loss to gateway", "value": CLEAN, "level": "ok"},
        {"key": "network.hop_1", "label": "Hop 1", "value": "192.0.2.1 sent 5 recv 5 loss 0% avg 1 ms", "level": "ok"},
        {"key": "network.hop_2", "label": "Hop 2", "value": "198.51.100.1 sent 5 recv 4 loss 20% avg 9 ms", "level": "warn"},
        {"key": "network.hop_3", "label": "Hop 3", "value": "[redacted]", "level": "ok", "sensitive": True},
    )
    path = ts.network_path(rep)
    assert path is not None
    keys = [i["key"] for i in path["items"]]
    assert keys == ["network.loss_gateway", "network.hop_1", "network.hop_2"]
    assert path["items"][2]["value"] == "198.51.100.1 sent 5 recv 4 loss 20% avg 9 ms"
    assert path["worst"] == "warn"


def test_path_items_are_not_misread_as_dns_failures():
    for key in ("network.loss_dns", "loss_gateway", "network.hop_2", "network.path_verdict", "network.iface_errors_eth0", "network.iface_errors"):
        assert ts.classify_item({"key": key, "label": "DNS gateway link"}) == "path", key
    assert ts.classify_item({"key": "network.dns_lookup", "label": "DNS lookup"}) == "dns"


# --------------------------------------------------------- probe_device ---


def test_probe_device_adds_network_path(monkeypatch):
    api = build_api(probe=dns_lossy_probe(DNS_VERDICT))
    install(monkeypatch, api)
    out = run(server.probe_device, "kiosk-7", sections=["network"])
    assert out["report"]["networkVerdict"]["code"] == "dns_path_lossy"
    assert out["network_path"]["verdict"]["cause"].startswith("DNS server path lossy")


def test_probe_device_older_agent_has_no_network_path(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    out = run(server.probe_device, "kiosk-7")
    assert "network_path" not in out and set(out) == {"device", "report"}


# ------------------------------------------------- troubleshoot_device ---


def test_troubleshoot_reported_scenario_dns_lookup_and_packet_loss(monkeypatch):
    api = build_api(probe=dns_lossy_probe(DNS_VERDICT))
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7")

    path = by_code(out, "network_path_quality")
    assert path["severity"] == "critical" and path["area"] == "connectivity"
    assert path["likely_cause"].startswith("DNS server path lossy")
    assert "likely cause: DNS server path lossy" in path["finding"]
    assert any("35% loss" in e for e in path["evidence"]) and any("high confidence" in e for e in path["evidence"])
    assert "DNS server" in path["what_you_can_do"][0]
    dns = by_code(out, "dns_failure")
    assert dns["likely_cause"].startswith("DNS server path lossy")
    assert dns["what_you_can_do"][0].startswith("Packets to the DNS server are being lost")
    # loss_dns must not be counted as a second DNS failure, nor as a generic probe_network finding
    assert codes(out).count("dns_failure") == 1 and "probe_network" not in codes(out)
    assert out["network_path"]["verdict"]["code"] == "dns_path_lossy"
    assert out["overall"] == "critical"


def test_troubleshoot_resolver_verdict_refines_dns_failure_only(monkeypatch):
    rep = probe_report(
        {"key": "network.dns_lookup", "label": "DNS lookup", "value": "SERVFAIL", "level": "error"},
        {"key": "network.loss_gateway", "label": "Loss to gateway", "value": CLEAN, "level": "ok"},
        {"key": "network.loss_dns", "label": "Loss to DNS server", "value": CLEAN, "level": "ok"},
    )
    rep["networkVerdict"] = {"code": "dns_resolver", "cause": "DNS resolver misconfigured or not serving: x", "evidence": ["network.loss_dns: 0% loss (ok)"], "confidence": "medium"}
    api = build_api(probe=rep)
    install(monkeypatch, api)

    out = run(server.troubleshoot_device, "kiosk-7")

    assert "network_path_quality" not in codes(out)
    assert by_code(out, "dns_failure")["likely_cause"].startswith("DNS resolver misconfigured")
    assert out["network_path"]["worst"] == "ok"


def test_troubleshoot_loss_items_without_backend_verdict(monkeypatch):
    api = build_api(probe=dns_lossy_probe())
    install(monkeypatch, api)
    out = run(server.troubleshoot_device, "kiosk-7")
    path = by_code(out, "network_path_quality")
    assert path["finding"] == "Packet loss or errors on the network path"
    assert path["likely_cause"] == ts.KB["network_path_quality"]["cause"]
    assert "verdict" not in out["network_path"]


def test_troubleshoot_older_agent_output_unchanged(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    out = run(server.troubleshoot_device, "kiosk-7")
    assert "network_path" not in out and out["findings"] == []


def test_troubleshoot_upstream_warn_uses_device_hint(monkeypatch):
    rep = upstream_probe()
    rep["sections"][0]["items"][2]["level"] = "warn"
    api = build_api(probe=rep)
    install(monkeypatch, api)
    out = run(server.troubleshoot_device, "kiosk-7")
    path = by_code(out, "network_path_quality")
    assert path["severity"] == "warning"
    assert path["what_you_can_do"][0].startswith("The local link is clean")
    assert "Loss starts upstream of the gateway" in path["what_you_can_do"]


# ------------------------------------------- check_device_connectivity ---


def test_connectivity_path_quality_check_and_suspected_cause(monkeypatch):
    api = build_api(probe=upstream_probe())
    install(monkeypatch, api)

    out = run(server.check_device_connectivity, "kiosk-7")

    row = next(c for c in out["checks"] if c["check"] == "path_quality")
    assert row["status"] == "error" and row["detail"].startswith("likely cause: upstream of the gateway")
    assert out["suspected_causes"][0].startswith("Measured on the device: upstream of the gateway")
    assert out["network_path"]["verdict"]["code"] == "upstream"
    assert "network_path_quality" in codes(out)
    assert "Hop 3" not in json.dumps(out)


def test_connectivity_older_agent_path_quality_unknown(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    out = run(server.check_device_connectivity, "kiosk-7")
    row = next(c for c in out["checks"] if c["check"] == "path_quality")
    assert row["status"] == "unknown" and "network_path" not in out


# ------------------------------------------------------- diagnose_device ---


class _Resolver:
    match: dict[str, Any] = MATCH_ONLINE

    def __init__(self, client: Any) -> None:
        pass

    def resolve_org(self, organization_id: Any) -> str:
        return "org-1"

    def find(self, query: str, org_id: str, fleet_id: Any = None) -> dict[str, Any]:
        return {"match": dict(self.match), "candidates": []}


def _diagnose(monkeypatch, api, match=MATCH_ONLINE, **kw) -> dict[str, Any]:
    install(monkeypatch, api)
    monkeypatch.setattr(_Resolver, "match", match)
    monkeypatch.setattr(server, "DeviceResolver", _Resolver)
    return run(server.diagnose_device, "kiosk-7", **kw)


def test_diagnose_default_runs_no_probe_and_is_unchanged(monkeypatch):
    api = build_api(probe=dns_lossy_probe(DNS_VERDICT))
    out = _diagnose(monkeypatch, api)
    rep = out["reports"][0]
    assert api.count("/diagnostics/probe") == 0
    assert "network_path" not in rep and "probe_note" not in rep and rep["source"] == "server"


def test_diagnose_include_probe_adds_network_path(monkeypatch):
    api = build_api(probe=dns_lossy_probe(DNS_VERDICT))
    out = _diagnose(monkeypatch, api, include_probe=True)
    rep = out["reports"][0]
    assert api.count("/diagnostics/probe") == 1
    body = json.loads(next(r for r in api.seen if r.url.path.endswith("/diagnostics/probe")).content)
    assert body == {"sections": ["network"]}
    assert rep["network_path"]["verdict"]["code"] == "dns_path_lossy" and rep["network_path"]["source"] == "fresh probe"


def test_diagnose_reuses_recent_probe_from_this_session(monkeypatch):
    api = build_api(probe=upstream_probe())
    install(monkeypatch, api)
    run(server.troubleshoot_device, "kiosk-7")  # fills the probe cache
    monkeypatch.setattr(_Resolver, "match", MATCH_ONLINE)
    monkeypatch.setattr(server, "DeviceResolver", _Resolver)
    out = run(server.diagnose_device, "kiosk-7")
    path = out["reports"][0]["network_path"]
    assert path["verdict"]["code"] == "upstream" and path["source"].startswith("probe from ")
    assert api.count("/diagnostics/probe") == 1


def test_diagnose_ignores_stale_cached_probe(monkeypatch):
    api = build_api()
    install(monkeypatch, api)
    ts._probe_cache[DEV] = (0.0, upstream_probe())
    monkeypatch.setattr(server, "DeviceResolver", _Resolver)
    out = run(server.diagnose_device, "kiosk-7")
    assert "network_path" not in out["reports"][0]


def test_diagnose_include_probe_offline_device(monkeypatch):
    api = build_api(online=False)
    out = _diagnose(monkeypatch, api, match=MATCH_OFFLINE, include_probe=True)
    rep = out["reports"][0]
    assert api.count("/diagnostics/probe") == 0
    assert "not online" in rep["probe_note"] and "network_path" not in rep


def test_diagnose_include_probe_failure_is_noted(monkeypatch):
    api = build_api()
    api.on("POST", f"/devices/{DEV}/diagnostics/probe", httpx.Response(504, json=envelope(None, error="device_timeout")))
    out = _diagnose(monkeypatch, api, include_probe=True)
    rep = out["reports"][0]
    assert rep["probe_note"].startswith("probe unavailable") and "network_path" not in rep


def test_diagnose_include_probe_older_agent(monkeypatch):
    api = build_api()
    out = _diagnose(monkeypatch, api, include_probe=True)
    rep = out["reports"][0]
    assert api.count("/diagnostics/probe") == 1 and "network_path" not in rep and "probe_note" not in rep


def test_attach_network_path_without_device_id():
    report: dict[str, Any] = {}
    server._attach_network_path(None, report, {}, "org-1", live=True)  # type: ignore[arg-type]
    assert report == {}
