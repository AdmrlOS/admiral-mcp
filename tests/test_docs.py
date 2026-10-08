from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from admrl_mcp import browser, docs, server
from admrl_mcp.client import AdmiralClient
from admrl_mcp.config import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "docs-search-index.json"
BASE = "https://docs.admrl.co"


def raw_index():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    saved_client = server._client
    server._client = None
    browser.reset()
    docs.clear_cache()
    monkeypatch.delenv("ADMRL_DOCS_BASE", raising=False)
    yield
    if server._client is not None:
        server._client.close()
    server._client = saved_client
    browser.reset()
    docs.clear_cache()


@pytest.fixture
def index():
    return docs.build_index(raw_index(), BASE)


def _serve(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    monkeypatch.setattr(docs, "_transport", lambda: transport)
    return seen


# ------------------------------------------------------------------ scorer ---


def test_bluetooth_query_ranks_provisioning_page_first(index):
    hits = docs.search("how do I provision a device over bluetooth", 5, index)
    assert hits[0]["url"].startswith(f"{BASE}/getting-started/bluetooth-provisioning")
    assert {"title", "section", "url", "breadcrumbs", "snippet"} <= set(hits[0])
    assert all(h["url"].startswith(BASE + "/") for h in hits)


def test_title_hit_outweighs_body_hit(index):
    hits = docs.search("quickstart", 3, index)
    assert hits[0]["url"].startswith(f"{BASE}/getting-started/quickstart")


def test_at_most_two_sections_per_page(index):
    hits = docs.search("bluetooth", 10, index)
    per_page: dict[str, int] = {}
    for h in hits:
        page = h["url"].split("#")[0]
        per_page[page] = per_page.get(page, 0) + 1
    assert max(per_page.values()) <= 2
    assert len(per_page) >= 2  # several pages share the slots


def test_limit_and_empty_and_stopword_only_queries(index):
    assert len(docs.search("bluetooth", 1, index)) == 1
    assert docs.search("the of and", 5, index) == []
    assert docs.search("zzzxqv", 5, index) == []


def test_anchor_urls_come_from_section_anchor(index):
    hits = docs.search("connect find device chooser", 3, index)
    assert any("#" in h["url"] for h in hits)


def test_snippet_is_bounded_and_contains_match(index):
    long = "word " * 200 + "needle in the middle " + "word " * 200
    snip = docs.make_snippet(long, docs.tokenize("needle"))
    assert "needle" in snip and len(snip) <= docs.SNIPPET_CHARS + 4
    assert docs.make_snippet("short text", ["short"]) == "short text"


def test_part3_keyword_boilerplate_is_ignored(index):
    # every page carries "OTA updates ... airdetect" keywords; they must not match
    assert docs.search("airdetect", 5, index) == []


# ------------------------------------------------------------------- reading ---


def test_read_page_orders_sections_and_resolves_inputs(index):
    page = docs.read_page(f"{BASE}/getting-started/bluetooth-provisioning#1-connect", index=index)
    assert page["title"] == "Provisioning Over Bluetooth"
    assert page["breadcrumbs"] and page["description"]
    names = [s["section"] for s in page["sections"]]
    assert names.index("Requirements") < names.index("Steps") < names.index("1. Connect")
    assert page["sections"][0]["url"].startswith(f"{BASE}/getting-started/bluetooth-provisioning")
    for form in ("/getting-started/bluetooth-provisioning/", "getting-started/bluetooth-provisioning", "/getting-started/bluetooth-provisioning.html"):
        assert docs.read_page(form, index=index)["title"] == page["title"]


def test_read_page_truncates(index):
    page = docs.read_page("/advanced/ble", max_chars=500, index=index)
    assert page["truncated"] is True
    assert sum(len(s["text"]) for s in page["sections"]) <= 501


def test_read_page_rejects_unknown_and_foreign_hosts(index):
    with pytest.raises(docs.DocsError, match="No docs page"):
        docs.read_page("/nope", index=index)
    with pytest.raises(docs.DocsError, match="Refusing"):
        docs.read_page("https://evil.example/getting-started/quickstart", index=index)


# ---------------------------------------------------------- fetch and cache ---


def test_index_is_cached_with_ttl_and_refetched(monkeypatch):
    seen = _serve(monkeypatch, lambda r: httpx.Response(200, json=raw_index()))
    clock = [1000.0]
    monkeypatch.setattr(docs, "_now", lambda: clock[0])
    docs.search("bluetooth")
    docs.search("pairing")
    assert len(seen) == 1 and str(seen[0].url) == BASE + "/search-index.json"
    clock[0] += docs.CACHE_TTL_SECONDS - 1
    docs.search("bluetooth")
    assert len(seen) == 1
    clock[0] += 2
    docs.search("bluetooth")
    assert len(seen) == 2


def test_failure_gives_clear_error_and_stale_copy_is_served(monkeypatch):
    state = {"ok": True}

    def handler(request):
        return httpx.Response(200, json=raw_index()) if state["ok"] else httpx.Response(503)

    _serve(monkeypatch, handler)
    clock = [0.0]
    monkeypatch.setattr(docs, "_now", lambda: clock[0])
    docs.clear_cache()
    state["ok"] = False
    out = json.loads(server.search_docs("bluetooth"))
    assert "Could not load the Admiral docs" in out["error"] and "503" in out["error"]
    state["ok"] = True
    assert json.loads(server.search_docs("bluetooth"))["results"]
    clock[0] += docs.CACHE_TTL_SECONDS + 1
    state["ok"] = False
    assert json.loads(server.search_docs("bluetooth"))["results"]  # stale beats failing


def test_bad_json_shape_is_a_clear_error(monkeypatch):
    _serve(monkeypatch, lambda r: httpx.Response(200, json={"not": "a list"}))
    assert "unexpected shape" in json.loads(server.search_docs("x"))["error"]


def test_network_error_is_reported(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("offline", request=request)

    _serve(monkeypatch, boom)
    assert "Could not load the Admiral docs" in json.loads(server.read_docs_page("/intro"))["error"]


def test_docs_base_env_override_and_configure(monkeypatch):
    seen = _serve(monkeypatch, lambda r: httpx.Response(200, json=raw_index()))
    monkeypatch.setenv("ADMRL_DOCS_BASE", "https://docs.example.test/")
    out = json.loads(server.search_docs("bluetooth"))
    assert str(seen[0].url) == "https://docs.example.test/search-index.json"
    assert out["results"][0]["url"].startswith("https://docs.example.test/")
    browser.configure("https://api.example/v1", docs_base="https://other.example")
    assert docs.docs_base() == "https://other.example"
    browser.configure("https://api.example/v1")
    assert docs.docs_base() == docs.DEFAULT_DOCS_BASE


def test_redirect_to_other_host_is_not_followed(monkeypatch):
    seen = _serve(monkeypatch, lambda r: httpx.Response(302, headers={"location": "https://evil.example/x"}))
    assert "error" in json.loads(server.search_docs("bluetooth"))
    assert len(seen) == 1


# -------------------------------------------------- credentials never leak ---

_FORBIDDEN = ("authorization", "x-organization-id", "cookie")


def _assert_clean(request: httpx.Request):
    names = {k.lower() for k in request.headers.keys()}
    assert not names & set(_FORBIDDEN), names
    assert not [n for n in names if n.startswith("x-api-")], names


def test_docs_request_has_no_credentials_in_pat_mode(monkeypatch):
    api_seen: list[httpx.Request] = []

    def api(request):
        api_seen.append(request)
        return httpx.Response(200, json={"code": 200, "msg": "ok", "data": []})

    server._client = AdmiralClient(Settings("https://api.example/v1", "tid", "secret", "org-1"), transport=httpx.MockTransport(api))
    seen = _serve(monkeypatch, lambda r: httpx.Response(200, json=raw_index()))
    server._client.list_devices()  # the Admiral client does carry credentials ...
    assert "x-api-token-id" in api_seen[0].headers
    server.search_docs("bluetooth")
    server.read_docs_page("/advanced/ble")
    assert seen and all(r.url.host == "docs.admrl.co" for r in seen)
    for r in seen:  # ... the docs request does not
        _assert_clean(r)


def test_docs_request_has_no_credentials_in_bearer_mode():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        if request.url.host == "docs.admrl.co":
            return httpx.Response(200, json=raw_index())
        return httpx.Response(200, json={"code": 200, "msg": "ok", "data": []})

    browser._state.transport = httpx.MockTransport(handler)
    browser.configure("https://api.example/v1")
    browser.set_auth("jwt-secret", "org-7")
    server.get_client().list_devices()
    assert seen[0].headers["authorization"] == "Bearer jwt-secret"  # sanity: the API call is authenticated
    out = json.loads(server.search_docs("bluetooth"))
    assert out["results"]
    docs_requests = [r for r in seen if r.url.host == "docs.admrl.co"]
    assert len(docs_requests) == 1
    _assert_clean(docs_requests[0])
    assert "jwt-secret" not in str(docs_requests[0].headers) and "org-7" not in str(docs_requests[0].headers)


# ------------------------------------------------------------ tool surface ---


def test_tools_registered_read_only_open_world_and_in_browser():
    by = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    for name in ("search_docs", "read_docs_page"):
        assert by[name].annotations.readOnlyHint is True and by[name].annotations.openWorldHint is True
        assert "docs" in by[name].description and "cite" in by[name].description
    shown = {t.name for t in asyncio.run(browser._server().list_tools())}
    assert {"search_docs", "read_docs_page"} <= shown
    sig = by["search_docs"].inputSchema["properties"]
    assert sig["limit"]["default"] == 5 and "query" in sig


def test_repeated_browser_configure_keeps_the_cache():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=raw_index())

    browser._state.transport = httpx.MockTransport(handler)
    browser.configure("https://api.example/v1")
    server.search_docs("bluetooth")
    browser.configure("https://api.example/v1")  # the engine does this before every call
    server.read_docs_page("/advanced/ble")
    assert len(seen) == 1


def test_read_page_does_not_repeat_description_as_a_section(index):
    page = docs.read_page("/getting-started/bluetooth-provisioning", index=index)
    assert all(s["text"] != page["description"] for s in page["sections"])
