import httpx
import pytest

from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import ConfigError, Settings


@pytest.mark.parametrize("escape", ["redirect", "absolute", "path"])
def test_allowed_base_blocks_escaping_requests(escape):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example.com/v1/devices"})

    base = "https://isolated.example.com/v1"
    settings = Settings(base, "test-token", "test-secret", "test-org", allowed_api_base=base)
    with AdmiralClient(settings, transport=httpx.MockTransport(handler)) as client:
        path = {"redirect": "devices", "absolute": "https://other.example.com/v1/devices", "path": "../outside"}[escape]
        with pytest.raises(ConfigError, match="allowed API base"):
            client._client.get(path)
    assert len(seen) == (1 if escape == "redirect" else 0)
    assert all(request.url.host == "isolated.example.com" for request in seen)


def test_allowed_base_accepts_normal_request():
    base = "https://isolated.example.com/v1"
    settings = Settings(base, "test-token", "test-secret", "test-org", allowed_api_base=base)
    with AdmiralClient(settings, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))) as client:
        assert client._client.get("devices").json() == {"ok": True}
