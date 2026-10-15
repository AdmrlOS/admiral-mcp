"""NVIDIA GPU surfacing: driver/CUDA in specs, the gpu diagnostics section, and gpu_* device metrics."""

from __future__ import annotations

import json

import httpx
import pytest

from admrl_mcp import server
from admrl_mcp import troubleshoot as ts
from test_troubleshoot import DEV, build_api, install as install_ts, probe_report, run
from test_w1_tools import FakeAPI, RESOLVED, assert_pat, envelope, install

FLEET = "11111111-2222-3333-4444-555555555555"
GPU = {
    "model": "NVIDIA RTX A2000",
    "vendor": "nvidia",
    "pci": "0000:01:00.0",
    "integrated": False,
    "driver": "nvidia",
    "node": "/dev/nvidia0",
    "driverVersion": "615.78.08",
    "cudaVersion": "13.4",
}


def test_specs_pass_through_gpu_versions(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/specifications", httpx.Response(200, json=envelope({"gpus": [GPU]})))
    install(monkeypatch, api)
    out = json.loads(server.get_device_specs("kiosk-7"))
    gpu = out["specifications"]["gpus"][0]
    assert gpu["driverVersion"] == "615.78.08" and gpu["cudaVersion"] == "13.4"
    assert gpu["model"] == "NVIDIA RTX A2000" and gpu["integrated"] is False
    assert_pat(api.only())


def test_specs_description_mentions_gpu_versions():
    desc = server.mcp._tool_manager.get_tool("get_device_specs").description
    assert "driverVersion" in desc and "cudaVersion" in desc


GPU_ITEMS = (
    {"key": "gpu.power_0", "label": "GPU 0 power", "value": "42 W of 215 W limit", "level": "ok"},
    {"key": "gpu.memory_0", "label": "GPU 0 memory", "value": "1 MiB of 8192 MiB (0%)", "level": "ok"},
    {"key": "gpu.probe", "label": "GPU probe", "value": "nvidia-smi ok", "level": "info"},
)


def test_probe_device_returns_gpu_section(monkeypatch):
    api = build_api(probe=probe_report(*GPU_ITEMS, section="gpu"))
    install_ts(monkeypatch, api)
    out = run(server.probe_device, "kiosk-7", sections=["gpu"])
    [section] = out["report"]["sections"]
    assert section["key"] == "gpu"
    assert [i["key"] for i in section["items"]] == ["gpu.power_0", "gpu.memory_0", "gpu.probe"]
    assert section["items"][0]["value"] == "42 W of 215 W limit"
    assert "network_path" not in out
    assert json.loads(next(r for r in api.seen if r.url.path.endswith("/diagnostics/probe")).content) == {"sections": ["gpu"]}


def test_probe_description_documents_gpu_section():
    desc = server.mcp._tool_manager.get_tool("probe_device").description
    assert "gpu.power_<n>" in desc and "gpu.memory_<n>" in desc


def test_gpu_items_are_not_sensitive_and_flow_through_probe_items():
    items = ts.probe_items(probe_report(*GPU_ITEMS, section="gpu"))
    assert [i["key"] for i in items] == ["gpu.power_0", "gpu.memory_0", "gpu.probe"]
    assert all(i["section"] == "gpu" for i in items)


def test_gpu_warning_becomes_probe_finding_and_ok_items_stay_quiet():
    warn = {"key": "gpu.power_0", "label": "GPU 0 power", "value": "214 W of 215 W limit", "level": "warn"}
    found = ts._probe({"probe": probe_report(warn, *GPU_ITEMS[1:], section="gpu")})
    [f] = found
    assert f["code"] == "probe_gpu" and "214 W of 215 W limit" in json.dumps(f["evidence"])
    assert ts._probe({"probe": probe_report(*GPU_ITEMS, section="gpu")}) == []


def test_gpu_symptom_maps_to_probe_area():
    areas = {a for words, ar in ts._SYMPTOM_AREAS if "vram" in words for a in ar}
    assert "probe" in areas


@pytest.mark.parametrize("metric", ["gpu_temperature", "gpu_power", "gpu_memory", "gpu_utilization"])
def test_get_device_metrics_accepts_gpu_metrics(monkeypatch, metric):
    api = FakeAPI()
    series = [{"labels": {"gpu": "0"}, "points": [[1, 2]]}, {"labels": {"gpu": "1"}, "points": [[1, 3]]}]
    api.on("GET", f"/devices/{DEV}/metrics", httpx.Response(200, json=envelope({"series": series})))
    monkeypatch.setitem(RESOLVED["match"], "fleet_id", FLEET)
    install(monkeypatch, api)
    out = json.loads(server.get_device_metrics("kiosk-7", metric=metric, lookback_hours=2))
    req = api.only()
    assert req.url.params["metric_name"] == metric and req.url.params["fleet_id"] == FLEET
    assert out["query"]["metric"] == metric
    assert len(out["result"]["series"]) == 2


def test_get_device_metrics_gpu_without_data_is_not_an_error(monkeypatch):
    api = FakeAPI()
    api.on("GET", f"/devices/{DEV}/metrics", httpx.Response(200, json=envelope({"series": None})))
    monkeypatch.setitem(RESOLVED["match"], "fleet_id", FLEET)
    install(monkeypatch, api)
    out = json.loads(server.get_device_metrics("kiosk-7", metric="gpu_power"))
    assert "error" not in out and out["result"]["series"] is None


def test_get_device_metrics_still_rejects_unknown_and_lists_gpu_choices():
    out = json.loads(server.get_device_metrics("kiosk-7", metric="gpu_fan"))
    assert "gpu_power" in out["choices"] and "cpu" in out["choices"]


def test_get_device_metrics_description_documents_gpu_units():
    desc = server.mcp._tool_manager.get_tool("get_device_metrics").description
    for word in ("gpu_temperature", "Celsius", "gpu_power", "watts", "gpu_memory", "gpu_utilization"):
        assert word in desc
