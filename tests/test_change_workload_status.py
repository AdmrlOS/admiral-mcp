from __future__ import annotations

import json

from admrl_mcp import server


def _patch_resolution(monkeypatch):
    resolved = {
        "match": {"id": "5f0c2a1e-7b3d-4c8e-9a6f-1d2e3f4a5b6c", "name": "dotmatrixboi"},
        "candidates": [],
        "organization_id": "org-1",
    }
    monkeypatch.setattr(server, "_resolve_device", lambda *a, **k: dict(resolved))


def test_change_workload_status_sends_stop_and_reports_acceptance(monkeypatch):
    captured: dict = {}

    class StubClient:
        def workload_command(self, device_id, *, action, org_id=None, **kw):
            captured.update({"device_id": device_id, "action": action, "org_id": org_id})
            return {
                "deviceId": device_id,
                "action": "STOP",
                "success": True,
                "timestamp": "2026-09-23T01:47:12Z",
            }

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    _patch_resolution(monkeypatch)

    out = json.loads(server.change_workload_status("dotmatrixboi", action="stop", confirm=True))
    assert captured == {
        "device_id": "5f0c2a1e-7b3d-4c8e-9a6f-1d2e3f4a5b6c",
        "action": "stop",
        "org_id": "org-1",
    }
    assert out["result"]["action"] == "STOP"
    assert out["result"]["success"] is True
    # Acceptance is not steady state — the tool must tell the caller to read back.
    assert "get_device_workload" in out["verify_with"]


def test_change_workload_status_supports_all_four_actions(monkeypatch):
    sent: list[str] = []

    class StubClient:
        def workload_command(self, device_id, *, action, org_id=None, **kw):
            sent.append(action)
            return {"success": True, "action": action.upper()}

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    _patch_resolution(monkeypatch)

    for action in ("start", "restart", "recreate", "stop"):
        out = json.loads(server.change_workload_status("dotmatrixboi", action=action, confirm=True))
        assert out["result"]["success"] is True
    assert sent == ["start", "restart", "recreate", "stop"]


def test_change_workload_status_rejects_unsupported_actions(monkeypatch):
    """DELETE/RESET are edge-capable but not exposed; reject before resolving."""

    def fail_resolve(*a, **k):
        raise AssertionError("invalid action must be rejected before device resolution")

    monkeypatch.setattr(server, "_resolve_device", fail_resolve)

    out = json.loads(server.change_workload_status("dotmatrixboi", action="reset", confirm=True))
    assert "error" in out
    assert sorted(out["choices"]) == ["recreate", "restart", "start", "stop"]


def test_change_workload_status_ambiguous_device_never_sends(monkeypatch):
    """A mutation sits behind the resolver: ambiguous candidates must not fire a command."""

    class StubClient:
        def workload_command(self, *a, **k):
            raise AssertionError("must not send a workload command without a unique match")

    monkeypatch.setattr(server, "get_client", lambda: StubClient())
    monkeypatch.setattr(
        server,
        "_resolve_device",
        lambda *a, **k: {"match": None, "candidates": [{"id": "x"}, {"id": "y"}], "organization_id": "org-1"},
    )

    out = json.loads(server.change_workload_status("twin", action="stop", confirm=True))
    assert out.get("match") is None
    assert len(out["candidates"]) == 2
