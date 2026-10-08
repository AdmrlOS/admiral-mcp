"""Device memory tests: error mapping and human-friendly summaries.

Only the online modes are exposed through this server:

* ``live``: the workload keeps running; a bounded slice of free memory is tested.
* ``full_online``: the workload is stopped for the duration so nearly all free memory can be tested.

A test boot (reboot into a dedicated test) is deliberately not available to API tokens; it is
started by a person from the dashboard or the device console.

Memory-test errors come back as a bare body ``{"error": <code>, "message"?, "status"?}`` rather
than the usual envelope; :func:`describe_error` understands both.
"""

from __future__ import annotations

from typing import Any

from .client import AdmiralAPIError

MODES = ("live", "full_online")

# mode -> request body fields (impact, placement). Placement is always "online".
MODE_BODY: dict[str, dict[str, str]] = {
    "live": {"impact": "running", "placement": "online"},
    "full_online": {"impact": "stopped", "placement": "online"},
}

OFFLINE_MESSAGE = (
    "A test boot (reboot into a dedicated memory test) cannot be started through the API or this server. "
    "Start it from the dashboard (device page, Memory test, Test boot) or from the device console. "
    "Use mode 'live' or 'full_online' here."
)

ERROR_MESSAGES: dict[str, str] = {
    "invalid": "The memory test request was rejected as invalid.",
    "operator_required": (
        "This memory test mode can only be started by a person from the dashboard or the device console, "
        "not through the API. Use mode 'live' or 'full_online'."
    ),
    "busy": "A memory test is already running on this device. Check it with get_memory_test or stop it with cancel_memory_test.",
    "not_running": "No memory test is running on this device, so there is nothing to cancel.",
    "unsupported": (
        "This device does not support memory tests (older firmware or an unsupported board). "
        "Update the device to current firmware and try again."
    ),
    "rate_limited": "Memory tests can be started at most once every 5 seconds per device. Wait a few seconds and retry.",
    "workload_stop_failed": (
        "The device could not stop the workload to run the test, so nothing was started. "
        "Check get_device_workload and try again, or use mode 'live', which leaves the workload running."
    ),
    "internal": "The device failed to handle the memory test request. Try again; if it repeats, check troubleshoot_device.",
    "device_offline": "The device is offline. Memory tests need a connected device (stored results can still be listed).",
    "device_timeout": "The device did not answer in time. It may be busy or the connection may be unstable; retry shortly.",
}

_STATUS_FALLBACK = {
    400: "invalid",
    422: "unsupported",
    429: "rate_limited",
    503: "device_offline",
    504: "device_timeout",
}


def error_code(exc: AdmiralAPIError) -> str | None:
    body = exc.body
    if isinstance(body, dict):
        code = body.get("error")
        if isinstance(code, str) and code in ERROR_MESSAGES:
            return code
    # 409 (busy vs not_running) and 403 (no access vs operator_required) need the body to tell apart.
    return _STATUS_FALLBACK.get(exc.status)


def describe_error(exc: AdmiralAPIError) -> dict[str, Any]:
    """Map an API error to ``{error, code, status, detail?, active_run?}`` with a clear message."""
    code = error_code(exc)
    out: dict[str, Any] = {"http_status": exc.status}
    if code is None:
        out["error"] = str(exc)
        return out
    out["code"] = code
    out["error"] = ERROR_MESSAGES[code]
    body = exc.body if isinstance(exc.body, dict) else {}
    detail = body.get("message")
    if isinstance(detail, str) and detail and detail != out["error"]:
        out["detail"] = detail
    status = body.get("status")
    if code == "busy" and isinstance(status, dict):
        out["active_run"] = summarise_status(status)
    return out


# ------------------------------------------------------------ summaries ---


def _pct(num: Any, den: Any) -> float | None:
    try:
        if den and num is not None:
            return round(100.0 * float(num) / float(den), 1)
    except (TypeError, ValueError):
        pass
    return None


def _mib(n: Any) -> float | None:
    return round(n / (1024 * 1024), 1) if isinstance(n, (int, float)) and n else None


def _temp(milli: Any) -> float | None:
    return round(milli / 1000.0, 1) if isinstance(milli, (int, float)) and milli > 0 else None


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != "" and v != []}


_ACTIVE_PHASES = {"preparing", "stopping_workload", "rebooting", "kernel_sweep", "testing", "confirming", "retiring", "restoring"}

_VERDICTS = {
    "ok": "No memory errors found.",
    "weak_cells": "Scattered single-bit errors: weak or failing memory cells. Pages that failed repeatably were retired.",
    "data_line": "The same bit fails across many addresses: a faulty data line. Replace the board/RAM.",
    "address_line": "Address-correlated failures: a faulty address line. Replace the board/RAM.",
    "thermal": "Errors appeared only at high temperature. Check cooling and ambient temperature, then retest.",
    "unknown": "Errors were found but could not be classified.",
}


def summarise_status(st: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(st, dict):
        return None
    plan = st.get("plan") or {}
    phase = st.get("phase")
    return _drop_none(
        {
            "run_id": st.get("runId"),
            "phase": phase,
            "active": phase in _ACTIVE_PHASES,
            "mode": _mode_label(plan),
            "pass": f"{st.get('pass')}/{st.get('passes')}" if st.get("passes") else None,
            "pattern": st.get("pattern"),
            "pattern_step": f"{st.get('patternIndex')}/{st.get('patternCount')}" if st.get("patternCount") else None,
            "pass_progress_percent": _pct(st.get("testedBytes"), st.get("targetBytes")),
            "coverage_percent": _pct(st.get("coverageBytes"), st.get("totalBytes")),
            "coverage_mib": _mib(st.get("coverageBytes")),
            "rate_mib_per_s": _mib(st.get("rateBps")),
            "eta_seconds": st.get("etaSeconds") or None,
            "errors": st.get("errors"),
            "confirmed_errors": st.get("confirmedErrors"),
            "temperature_c": _temp(st.get("tempMilliC")),
            "started_at": st.get("startedAt"),
            "updated_at": st.get("updatedAt"),
            "message": st.get("message"),
        }
    )


def _mode_label(plan: dict[str, Any]) -> str | None:
    if not plan:
        return None
    if plan.get("placement") == "offline":
        return "test boot"
    if plan.get("impact") == "stopped":
        return "full online (workload stopped)"
    return "live (quick)" if plan.get("quick") else "live"


def summarise_result(res: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(res, dict):
        return None
    verdict = res.get("verdict")
    outcome = res.get("outcome")
    retired = res.get("retired") or []
    out = _drop_none(
        {
            "run_id": res.get("runId"),
            "mode": _mode_label(res.get("plan") or {}),
            "outcome": outcome,
            "verdict": verdict,
            "verdict_meaning": _VERDICTS.get(str(verdict)),
            "summary": res.get("summary"),
            "started_at": res.get("startedAt"),
            "finished_at": res.get("finishedAt"),
            "passes_completed": res.get("passesCompleted"),
            "coverage_percent": _pct(res.get("coverageBytes"), res.get("totalBytes")),
            "coverage_mib": _mib(res.get("coverageBytes")),
            "max_temperature_c": _temp(res.get("maxTempMilliC")),
            "error_count": res.get("errorCount"),
            "retired_pages": len(retired),
            "fault": bool(res.get("fault")),
        }
    )
    if outcome == "interrupted":
        intr = res.get("interrupted") or {}
        out["interrupted_at_phase"] = intr.get("phase")
        out["note"] = "The device reset or crashed during this test, so it did not finish. Run it again."
    errs = [e for e in (res.get("errors") or []) if isinstance(e, dict)]
    if errs:
        out["sample_errors"] = [
            _drop_none(
                {
                    "physical_address": f"0x{e['physAddr']:x}" if isinstance(e.get("physAddr"), int) else None,
                    "pattern": e.get("pattern"),
                    "mask": f"0x{e['mask']:x}" if isinstance(e.get("mask"), int) else None,
                    "confirmed": e.get("confirmed"),
                }
            )
            for e in errs[:8]
        ]
    return out


def last_interrupted(health: dict[str, Any] | None, history: list[Any] | None = None) -> dict[str, Any] | None:
    """The interrupted-run marker, if the newest result is an interrupted run."""
    h = health or {}
    last = h.get("lastResult")
    if not isinstance(last, dict) and history:
        last = next((r for r in history if isinstance(r, dict)), None)
    if isinstance(last, dict) and last.get("outcome") == "interrupted":
        return last.get("interrupted") or {"runId": last.get("runId")}
    if isinstance(h.get("interrupted"), dict):
        return h["interrupted"]
    return None


def summarise_health(health: dict[str, Any] | None, history: list[Any] | None = None) -> dict[str, Any] | None:
    if not isinstance(health, dict):
        return None
    caps = health.get("capabilities") or {}
    retired = health.get("retiredPages") or []
    out: dict[str, Any] = {
        "fault": bool(health.get("fault")),
        "fault_reason": health.get("faultReason") or None,
        "retired_pages": len(retired),
        "retired_pages_reapplied_this_boot": health.get("retiredApplied"),
        "hardware_corrupted_mib": _mib(health.get("hardwareCorruptedBytes")),
        "ecc_corrected": _none_if_negative(health.get("edacCorrected")),
        "ecc_uncorrected": _none_if_negative(health.get("edacUncorrected")),
        "recent_kernel_memory_errors": (health.get("kmsgHits") or [])[:5],
        "capabilities": {
            "live_and_full_online": bool(caps.get("online")),
            "test_boot_available": bool(caps.get("offline")),
            "page_retirement": bool(caps.get("softOffline")),
            "ecc_counters": bool(caps.get("edac")),
        },
        "last_result": summarise_result(health.get("lastResult")),
    }
    kernel = health.get("kernel")
    if isinstance(kernel, dict) and kernel.get("ran"):
        out["kernel_sweep"] = {"passes": kernel.get("passes"), "bad_ranges": len(kernel.get("badRanges") or [])}
    if last_interrupted(health, history):
        out["interrupted"] = True
    out = _drop_none(out)
    if health.get("fault"):
        out["fault"] = True
    return out


def _none_if_negative(v: Any) -> Any:
    return None if isinstance(v, (int, float)) and v < 0 else v


def attention(supported: bool, status: dict[str, Any] | None, health: dict[str, Any] | None, history: list[Any] | None = None) -> list[dict[str, str]]:
    """What a person should look at, worst first."""
    out: list[dict[str, str]] = []
    h = health or {}
    if h.get("fault"):
        reason = h.get("faultReason")
        out.append(
            {
                "severity": "critical",
                "code": "memory_fault",
                "text": "Memory fault: replace the board/RAM." + (f" ({reason})" if reason else ""),
            }
        )
    if last_interrupted(h, history):
        out.append(
            {
                "severity": "warning",
                "code": "memory_test_interrupted",
                "text": "The last memory test was interrupted (the device reset or crashed during it). Run it again.",
            }
        )
    if isinstance(status, dict) and status.get("phase") in _ACTIVE_PHASES and (status.get("plan") or {}).get("impact") == "stopped":
        out.append(
            {
                "severity": "info",
                "code": "memory_test_holding_workload",
                "text": "A full memory test is holding the workload stopped. This is maintenance, not a failure; the workload restarts when the test ends.",
            }
        )
    return out


def start_body(
    mode: str,
    quick: bool | None,
    passes: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = dict(MODE_BODY[mode])
    if quick and mode == "live":
        body["quick"] = True
    if passes is not None:
        body["passes"] = int(passes)
    return body
