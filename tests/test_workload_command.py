from __future__ import annotations

import json

import httpx
import pytest

from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings


def _settings() -> Settings:
    return Settings(
        api_base="https://api.admrl.co/v1",
        token_id="tok",
        secret_key="sec",
        org_id="org-1",
    )


def test_workload_command_puts_action_to_cmd_route():
    """client.workload_command must hit PUT /devices/{id}/workload/cmd with the action body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "code": 200,
                "msg": "Success",
                "data": {
                    "deviceId": "d1",
                    "action": "STOP",
                    "success": True,
                    "timestamp": "2026-09-23T01:47:12Z",
                },
            },
        )

    client = AdmiralClient(settings=_settings(), transport=httpx.MockTransport(handler))
    payload = client.workload_command("d1", action="stop", org_id="org-1")
    assert captured["method"] == "PUT"
    assert captured["path"].endswith("/devices/d1/workload/cmd")
    # The edge's switch matches uppercase actions; the client canonicalises.
    assert captured["body"] == {"action": "STOP"}
    # The edge executes synchronously under a 60s server budget; the client
    # read timeout must not be the 30s default or slow edges time out first.
    assert captured["timeout"]["read"] > 60
    assert payload["success"] is True


def test_workload_command_unknown_action_never_reaches_the_network():
    """DELETE/RESET/garbage must be rejected client-side, not shipped to the edge."""
    client = AdmiralClient(
        settings=_settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    with pytest.raises(ValueError) as exc:
        client.workload_command("d1", action="reset", org_id="org-1")
    assert "start" in str(exc.value) and "recreate" in str(exc.value)
