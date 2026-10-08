"""Summaries for observed-state and rollout SSE streams.

The streams repeat whole documents on every update; these trackers keep only
what changed (condition transitions, generation/workload changes, progress
operation phases, rollout status and per-device phase transitions) so a
watch tool returns a short timeline instead of N copies of the state.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

ROLLOUT_TERMINAL = ("completed", "failed", "cancelled", "rolled_back")

# Conditions that are "good" when True; used only to label a final verdict.
_POSITIVE_CONDITIONS = ("Converged", "WorkloadReady", "StorageOK", "TimeSynced")


def compact_operation(op: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id",
        "kind",
        "target",
        "phase",
        "bytesDone",
        "bytesTotal",
        "itemsDone",
        "itemsTotal",
        "rateBps",
        "etaSeconds",
        "message",
        "error",
        "attempt",
        "parent",
        "updatedAt",
        "finishedAt",
    )
    out = {k: op.get(k) for k in keep if op.get(k) not in (None, "", 0) or k in ("phase", "kind")}
    total = op.get("bytesTotal") or 0
    if total:
        out["pct"] = round(100.0 * (op.get("bytesDone") or 0) / total, 1)
    elif op.get("itemsTotal"):
        out["pct"] = round(100.0 * (op.get("itemsDone") or 0) / op["itemsTotal"], 1)
    return out


def summarise_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Compact view of a DeviceState document (json tags from admrl-core)."""
    st = state or {}
    workload = st.get("workload") or {}
    progress = st.get("progress") or {}
    conditions = st.get("conditions") or []
    local = st.get("localOverride")
    system = st.get("system") or {}
    return {
        "observedGeneration": st.get("observedGeneration"),
        "renderedRevision": st.get("renderedRevision"),
        "collectedAt": st.get("collectedAt"),
        "bootId": st.get("bootId"),
        "conditions": {
            c.get("type"): {k: c.get(k) for k in ("status", "reason", "message") if c.get(k)}
            for c in conditions
            if isinstance(c, dict) and c.get("type")
        },
        "workload": {
            k: workload.get(k)
            for k in ("state", "configurationName", "version", "image", "error")
            if workload.get(k) not in (None, "")
        }
        or None,
        "transition": workload.get("transition"),
        "versions": system.get("versions"),
        "progress": {
            "targetGeneration": progress.get("targetGeneration"),
            "operations": [compact_operation(op) for op in progress.get("operations") or [] if isinstance(op, dict)],
        }
        if progress
        else None,
        "localOverride": {k: local.get(k) for k in ("source", "paths", "editedAt", "reason")} if isinstance(local, dict) else None,
    }


class StateWatch:
    """Fold `event: state` payloads ({deviceId, kind, at, state}) into transitions."""

    def __init__(self) -> None:
        self.initial: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.kinds: Counter[str] = Counter()
        self.transitions: list[dict[str, Any]] = []
        self.operations: dict[str, dict[str, Any]] = {}
        self._conditions: dict[str, tuple[Any, Any]] = {}
        self._generation: Any = None
        self._workload_state: Any = None
        self._transition_state: Any = None

    def add(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        kind = str(payload.get("kind") or "unknown")
        at = payload.get("at")
        self.kinds[kind] += 1
        state = payload.get("state")
        if not isinstance(state, dict):
            return
        first = self.initial is None
        summary = summarise_state(state)
        if first:
            self.initial = summary
        self.last = summary

        for cond in state.get("conditions") or []:
            if not isinstance(cond, dict) or not cond.get("type"):
                continue
            ctype = cond["type"]
            now = (cond.get("status"), cond.get("reason"))
            before = self._conditions.get(ctype)
            self._conditions[ctype] = now
            if first or before == now:
                continue
            self.transitions.append(
                {
                    "at": at,
                    "kind": kind,
                    "type": "condition",
                    "condition": ctype,
                    "from": before[0] if before else None,
                    "to": now[0],
                    "reason": cond.get("reason"),
                    "message": cond.get("message"),
                }
            )

        gen = state.get("observedGeneration")
        if not first and gen != self._generation:
            self.transitions.append(
                {"at": at, "kind": kind, "type": "observedGeneration", "from": self._generation, "to": gen}
            )
        self._generation = gen

        workload = state.get("workload") or {}
        wstate = workload.get("state")
        if not first and wstate != self._workload_state:
            self.transitions.append(
                {"at": at, "kind": kind, "type": "workload", "from": self._workload_state, "to": wstate}
            )
        self._workload_state = wstate
        tstate = (workload.get("transition") or {}).get("state")
        if not first and tstate != self._transition_state:
            self.transitions.append(
                {
                    "at": at,
                    "kind": kind,
                    "type": "workload_transition",
                    "from": self._transition_state,
                    "to": tstate,
                    "message": (workload.get("transition") or {}).get("message"),
                }
            )
        self._transition_state = tstate

        for op in (state.get("progress") or {}).get("operations") or []:
            if not isinstance(op, dict):
                continue
            key = str(op.get("id") or f"{op.get('kind')}:{op.get('target')}")
            seen = self.operations.get(key)
            compact = compact_operation(op)
            if seen is None:
                self.operations[key] = {
                    "id": op.get("id"),
                    "kind": op.get("kind"),
                    "target": op.get("target"),
                    "phases": [op.get("phase")],
                    "first_seen": at,
                    "updates": 1,
                    "last": compact,
                }
                continue
            seen["updates"] += 1
            seen["last"] = compact
            if seen["phases"][-1] != op.get("phase"):
                seen["phases"].append(op.get("phase"))

    def converged(self) -> bool:
        cond = self._conditions.get("Converged")
        return bool(cond and cond[0] == "True")

    def result(self) -> dict[str, Any]:
        return {
            "initial": self.initial,
            "final": self.last,
            "transitions": self.transitions,
            "progress_operations": list(self.operations.values()),
            "updates_by_kind": dict(self.kinds),
            "converged": self.converged() if self._conditions else None,
        }


class RolloutWatch:
    """Fold rollout SSE events: `rollout` (counts/status) and `device` (phase)."""

    def __init__(self, initial_status: str | None = None, initial_paused_reason: str | None = None) -> None:
        self.status = initial_status
        # Seeded from GET: the stream does not replay the connect-time state.
        self.paused_reason: str | None = initial_paused_reason
        self.counts: dict[str, Any] | None = None
        self.phase_transitions: list[dict[str, Any]] = []
        self.device_transitions: list[dict[str, Any]] = []
        self.device_phase: dict[str, str | None] = {}
        self.device_progress: dict[str, Any] = {}
        self.events: Counter[str] = Counter()

    @property
    def terminal(self) -> bool:
        return (self.status or "") in ROLLOUT_TERMINAL

    def add(self, event: str, payload: Any) -> bool:
        """Returns True when the rollout reached a terminal status."""
        self.events[event] += 1
        if not isinstance(payload, dict):
            return False
        if event == "rollout":
            status = payload.get("status")
            reason = payload.get("pausedReason")
            if status != self.status or reason != self.paused_reason:
                self.phase_transitions.append(
                    {
                        "at": payload.get("at"),
                        "from": self.status,
                        "to": status,
                        "pausedReason": reason,
                        "counts": payload.get("counts"),
                    }
                )
            self.status = status
            self.paused_reason = reason
            self.counts = payload.get("counts") or self.counts
            return self.terminal
        if event == "device":
            device_id = str(payload.get("deviceId") or "")
            phase = payload.get("phase")
            before = self.device_phase.get(device_id)
            if phase != before:
                self.device_transitions.append(
                    {"at": payload.get("at"), "deviceId": device_id, "from": before, "to": phase}
                )
            self.device_phase[device_id] = phase
            if payload.get("progress"):
                self.device_progress[device_id] = compact_operation(payload["progress"])
        return False

    def result(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "terminal": self.terminal,
            "pausedReason": self.paused_reason,
            "counts": self.counts,
            "phase_transitions": self.phase_transitions,
            "device_transitions": self.device_transitions,
            "device_phases": self.device_phase,
            "device_progress": self.device_progress,
            "events": dict(self.events),
        }
