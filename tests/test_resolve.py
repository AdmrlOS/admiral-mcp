from __future__ import annotations

import httpx

from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings
from admrl_mcp.distill import distill_events, distill_logs
from admrl_mcp.resolve import DeviceResolver, collect_ips, match_score, summarise_device


PRINTER = {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "shop-printer-01",
    "status": "offline",
    "ipAddress": "10.1.1.42",
    "hardwareType": "Raspberry Pi 4",
    "notes": "Front counter receipt printer",
    "isOnline": {"isOnline": False, "lastSeen": "2026-09-14T04:00:00Z", "latencyMs": 0},
    "fleet": {"id": "aaaa1111-e29b-41d4-a716-446655440000", "name": "retail-syd"},
    "effective_tags": [
        {"key": "role", "value": "printer", "source": "device"},
        {"key": "site", "value": "padstow", "source": "fleet"},
    ],
    "systemSpec": {
        "network": [
            {"name": "eth0", "ipV4Address": "10.1.1.42", "macAddress": "aa:bb:cc:dd:ee:ff"}
        ]
    },
}


def test_summarise_device_includes_ip_and_tags():
    summary = summarise_device(PRINTER)
    assert summary["ip"] == "10.1.1.42"
    assert summary["status"] == "offline"
    assert {"key": "role", "value": "printer"} in summary["tags"]


def test_match_score_prefers_tag_and_name():
    assert match_score("printer", PRINTER) > match_score("warehouse", PRINTER)
    assert match_score("shop-printer-01", PRINTER) > match_score("printer", PRINTER)
    assert match_score("10.1.1.42", PRINTER) > 0
    assert match_score("role=printer", PRINTER) > 0


def test_collect_ips_merges_spec_and_live_status():
    network = {
        "interfaces": [
            {
                "name": "wlan0",
                "state": "up",
                "ips": [{"address": "10.1.1.88"}],
            }
        ]
    }
    ips = collect_ips(PRINTER, network)
    values = {row["ip"] for row in ips}
    assert "10.1.1.42" in values
    assert "10.1.1.88" in values


def test_collect_ips_prefers_spec_interface_attribution():
    device = {
        "ipAddress": "192.0.2.10",
        "systemSpec": {
            "network": [
                {"name": "eth0", "macAddress": "02:00:c6:ce:88:b1"},
                {
                    "name": "wlan0",
                    "ipV4Address": "192.0.2.10",
                    "ipV6Address": "fe80::1ab3:1cdd:6a39:9246",
                },
            ]
        },
    }
    ips = collect_ips(device)
    assert ips[0] == {
        "ip": "192.0.2.10",
        "interface": "wlan0",
        "version": "v4",
        "source": "system_spec",
    }
    assert not any(row["interface"] is None for row in ips)
    assert [row["ip"] for row in ips].count("192.0.2.10") == 1


def test_collect_ips_falls_back_to_list_ip_without_spec():
    ips = collect_ips({"ipAddress": "10.1.1.42"})
    assert ips == [{"ip": "10.1.1.42", "interface": None, "version": "v4", "source": "device_list"}]


def test_distill_logs_groups_crash_signatures():
    payload = {
        "logs": [
            {
                "timestamp": "2026-09-14T04:01:00Z",
                "level": "info",
                "source": "supervisor",
                "message": "workload started",
            },
            {
                "timestamp": "2026-09-14T04:02:00Z",
                "level": "error",
                "source": "workload",
                "message": "panic: nil pointer 0xdeadbeef",
            },
            {
                "timestamp": "2026-09-14T04:03:00Z",
                "level": "fatal",
                "source": "workload",
                "message": "panic: nil pointer 0xcafebabe",
            },
            {
                "timestamp": "2026-09-14T04:04:00Z",
                "level": "error",
                "source": "kernel",
                "message": "oom-killer: killed process 412 workload",
            },
        ],
        "has_more": False,
    }
    distilled = distill_logs(payload)
    assert distilled["count"] == 4
    assert distilled["crash_like"] == 3
    assert distilled["levels"]["error"] == 2
    panic_sig = next(s for s in distilled["top_signatures"] if "panic" in s["message"])
    assert panic_sig["count"] >= 2


def test_distill_events_flags_offline():
    payload = {
        "events": [
            {"event": "device_online", "source": "system", "timestamp": "t1"},
            {"event": "device_offline", "source": "system", "timestamp": "t2"},
            {"event": "workload_crash", "source": "agent", "timestamp": "t3"},
        ]
    }
    distilled = distill_events(payload)
    assert distilled["count"] == 3
    assert distilled["kinds"]["device_offline"] == 1
    assert len(distilled["notable"]) == 2


def test_distill_events_flags_pascal_case_workload_crashed():
    payload = {
        "events": [
            {"event": "WorkloadStarted", "source": "kernel", "timestamp": "t1"},
            {"event": "WorkloadCrashed", "source": "kernel", "timestamp": "t2", "data": {"message": "default-workload restart_attempt=16"}},
            {"event": "DeviceOnline", "source": "kernel", "timestamp": "t3"},
            {"event": "DeviceOffline", "source": "kernel", "timestamp": "t4"},
        ]
    }
    distilled = distill_events(payload)
    names = [e["event"] for e in distilled["notable"]]
    assert names == ["WorkloadCrashed", "DeviceOffline"]


def test_match_score_uses_name_tokens_and_tags_not_unrelated_devices():
    tagged = {
        "id": "aaaa1111-e29b-41d4-a716-446655440000",
        "name": "shop-front-01",
        "status": "online",
        "ipAddress": "10.1.1.42",
        "fleet": {"id": "f1", "name": "retail"},
        "effective_tags": [{"key": "role", "value": "kiosk"}],
    }
    named = {
        "id": "bbbb1111-e29b-41d4-a716-446655440000",
        "name": "lobby-kiosk",
        "status": "online",
        "ipAddress": "10.1.1.43",
        "fleet": {"id": "f1", "name": "retail"},
        "effective_tags": [],
    }
    other = {
        "id": "cccc1111-e29b-41d4-a716-446655440000",
        "name": "orin-lab",
        "status": "online",
        "ipAddress": "10.1.1.9",
        "fleet": {"id": "f2", "name": "debug"},
        "effective_tags": [],
    }
    assert match_score("kiosk", tagged) > match_score("kiosk", other)
    assert match_score("kiosk", named) > match_score("kiosk", other)
    assert match_score("kiosk", other) == 0


def test_find_picks_unique_tagged_device_from_org_search_fleet():
    devices = [
        {
            "id": "aaaa1111-e29b-41d4-a716-446655440000",
            "name": "orin-lab",
            "status": "online",
            "ipAddress": "10.0.0.1",
            "fleet": {"id": "f2", "name": "debug"},
        },
        {
            "id": "bbbb1111-e29b-41d4-a716-446655440000",
            "name": "lobby-kiosk",
            "status": "online",
            "ipAddress": "10.0.0.2",
            "fleet": {"id": "f1", "name": "retail-kiosk"},
            "effective_tags": [{"key": "role", "value": "kiosk"}],
        },
        {
            "id": "cccc1111-e29b-41d4-a716-446655440000",
            "name": "back-office",
            "status": "online",
            "ipAddress": "10.0.0.3",
            "fleet": {"id": "f1", "name": "retail-kiosk"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/devices"):
            fleet_id = request.url.params.get("fleet_id")
            rows = devices if not fleet_id else [d for d in devices if d["fleet"]["id"] == fleet_id]
            return httpx.Response(200, json={"code": 200, "data": rows})
        if path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "data": {
                        "devices": None,
                        "fleets": [{"id": "f1", "name": "retail-kiosk", "devices": 2}],
                        "total": 1,
                    },
                },
            )
        raise AssertionError(request.url)

    client = AdmiralClient(
        settings=Settings(
            api_base="https://api.admrl.co/v1",
            token_id="tok",
            secret_key="sec",
            org_id="org-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    found = DeviceResolver(client).find("kiosk", org_id="org-1")
    assert found["match"]["name"] == "lobby-kiosk"
    assert found["match"]["ip"] == "10.0.0.2"


def test_find_does_not_dump_unrelated_fleet_when_nothing_scores():
    devices = [
        {"id": "1", "name": "orin-super-turbo-extra-max-ultra", "status": "online", "ipAddress": "10.0.0.1"},
        {"id": "2", "name": "demo-dell", "status": "online", "ipAddress": "10.0.0.2"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/devices"):
            return httpx.Response(200, json={"code": 200, "data": devices})
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json={"code": 200, "data": {"devices": None, "total": 0}})
        raise AssertionError(request.url)

    client = AdmiralClient(
        settings=Settings(
            api_base="https://api.admrl.co/v1",
            token_id="tok",
            secret_key="sec",
            org_id="org-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    found = DeviceResolver(client).find("warehouse-kiosk", org_id="org-1")
    assert found["match"] is None
    assert found["candidates"] == []
    assert "No device matched" in found["error"]


CLAIMED = {
    "id": "dddd1111-e29b-41d4-a716-446655440000",
    "name": "r36s-kaira",
    "status": "claimed",
    "ipAddress": "10.0.0.96",
    "isOnline": {"isOnline": False, "lastSeen": "2026-09-18T07:50:05Z", "latencyMs": 0},
}
ONLINE = {
    "id": "eeee1111-e29b-41d4-a716-446655440000",
    "name": "raynboy",
    "status": "online",
    "ipAddress": "10.0.0.112",
    "isOnline": {"isOnline": True, "lastSeen": "2026-09-22T07:01:00Z", "latencyMs": 14},
}


def test_list_not_online_matches_counts_offline_bucket():
    """The status=offline filter matches ~nothing while counts.offline counts
    claimed/stale devices. list_not_online must return that same set."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/devices")
        assert request.url.params.get("status") is None  # filter must not be sent
        return httpx.Response(
            200,
            json={
                "code": 200,
                "data": [ONLINE, CLAIMED],
                "counts": {"total": 2, "online": 1, "offline": 1},
            },
        )

    client = AdmiralClient(
        settings=Settings(
            api_base="https://api.admrl.co/v1",
            token_id="tok",
            secret_key="sec",
            org_id="org-1",
        ),
        transport=httpx.MockTransport(handler),
    )
    rows = DeviceResolver(client).list_not_online(org_id="org-1")
    assert [r["id"] for r in rows] == [CLAIMED["id"]]
