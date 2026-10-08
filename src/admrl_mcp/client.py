from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urljoin
from datetime import datetime, timedelta

import httpx

from .config import AUTH_BEARER, ConfigError, Settings, require_settings


class AdmiralAPIError(RuntimeError):
    def __init__(self, status: int, message: str, body: Any = None):
        super().__init__(f"Admiral API {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


def _unwrap(payload: Any) -> Any:
    if not (isinstance(payload, dict) and "data" in payload and (
        "code" in payload or "msg" in payload or "success" in payload
    )):
        return payload
    data = payload["data"]
    extra = {k: payload[k] for k in ("pagination", "counts") if k in payload}
    if extra:
        if isinstance(data, list):
            return {"items": data, **extra}
        if isinstance(data, dict):
            return {**data, **extra}
        return {"data": data, **extra}
    return data


def _log_entries(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    if isinstance(payload, dict):
        logs = payload.get("logs")
        if isinstance(logs, list):
            return [e for e in logs if isinstance(e, dict)]
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("logs"), list):
            return [e for e in data["logs"] if isinstance(e, dict)]
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict)]
    return []


def _api_error(response: httpx.Response) -> AdmiralAPIError:
    try:
        payload = response.json() if response.content else None
    except ValueError:
        payload = {"raw": response.text[:2000]}
    message = None
    if isinstance(payload, dict):
        message = payload.get("msg") or payload.get("message") or payload.get("error")
    return AdmiralAPIError(response.status_code, str(message or response.reason_phrase), payload)


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Parse a text/event-stream into (event, data) pairs.

    Comment lines (": keepalive") are skipped; multi-line data is joined with
    newlines; an event without an `event:` field is named "message".
    """
    event = ""
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data:
                yield (event or "message", "\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield (event or "message", "\n".join(data))


def _newest_first(entries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not entries:
        return []

    def stamp(entry: dict[str, Any]) -> str:
        return str(entry.get("timestamp") or entry.get("time") or "")

    ordered = entries
    if len(entries) >= 2 and stamp(entries[0]) < stamp(entries[-1]):
        ordered = list(reversed(entries))
    return ordered[:limit]


class _SessionBearerAuth(httpx.Auth):
    """Adds ``Authorization: Bearer <token>`` using the client's *current* settings."""

    def __init__(self, client: "AdmiralClient"):
        self._client = client

    def auth_flow(self, request: httpx.Request) -> Iterator[httpx.Request]:
        token = self._client.settings.bearer_token
        if not token:
            raise ConfigError("Not signed in: no session token is set.")
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


class AdmiralClient:
    def __init__(self, settings: Settings | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings or require_settings()
        base = self.settings.api_base.rstrip("/") + "/"
        auth: httpx.Auth | None = None
        if self.settings.auth_mode == AUTH_BEARER:
            # Session mode (browser): no User-Agent (a forbidden header there)
            # and the token is read per request from self.settings.
            headers = {"Accept": "application/json"}
            auth = _SessionBearerAuth(self)
        else:
            headers = {
                "Accept": "application/json",
                "User-Agent": "admrl-mcp/0.1",
                "X-API-Token-ID": self.settings.token_id,
                "X-API-Secret-Key": self.settings.secret_key,
            }
        self._client = httpx.Client(
            base_url=base,
            headers=headers,
            auth=auth,
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
            follow_redirects=True,
        )

    def _org_required_message(self) -> str:
        if self.settings.auth_mode == AUTH_BEARER:
            return "Organisation context required. Pass organization_id (list_organisations shows the options)."
        return "Organisation context required. Pass organization_id or set ADMRL_ORG_ID."

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AdmiralClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        org_id: str | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
        require_org: bool = True,
        timeout: float | None = None,
    ) -> Any:
        headers: dict[str, str] = {}
        resolved_org = org_id or self.settings.org_id
        if require_org:
            if not resolved_org:
                raise AdmiralAPIError(400, self._org_required_message())
            headers["X-Organization-ID"] = resolved_org
        elif resolved_org:
            headers["X-Organization-ID"] = resolved_org

        # Drop empty query params so the API keeps its defaults.
        clean_params = None
        if params:
            clean_params = {k: v for k, v in params.items() if v is not None and v != ""}

        url = path.lstrip("/")
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = httpx.Timeout(timeout, connect=10.0)
        try:
            response = self._client.request(
                method,
                url,
                headers=headers,
                params=clean_params,
                json=json,
                **extra,
            )
        except httpx.HTTPError as exc:
            raise AdmiralAPIError(0, f"request failed: {exc}") from exc

        try:
            payload = response.json() if response.content else None
        except ValueError:
            payload = {"raw": response.text}

        if response.status_code >= 400:
            message = None
            if isinstance(payload, dict):
                message = payload.get("msg") or payload.get("message") or payload.get("error")
            raise AdmiralAPIError(response.status_code, str(message or response.reason_phrase), payload)

        return _unwrap(payload)

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    # ---------------------------------------------------------------- raw ---

    def _org_headers(self, org_id: str | None, require_org: bool) -> dict[str, str]:
        resolved_org = org_id or self.settings.org_id
        if require_org and not resolved_org:
            raise AdmiralAPIError(400, self._org_required_message())
        return {"X-Organization-ID": resolved_org} if resolved_org else {}

    @staticmethod
    def _clean(params: dict[str, Any] | None) -> dict[str, Any] | None:
        if not params:
            return None
        return {k: v for k, v in params.items() if v is not None and v != ""} or None

    def raw_request(
        self,
        method: str,
        path: str,
        *,
        org_id: str | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        require_org: bool = True,
        timeout: float | None = None,
    ) -> httpx.Response:
        """Request returning the httpx.Response (headers, YAML bodies, ETags).

        Raises AdmiralAPIError on HTTP >= 400 like request().
        """
        hdrs = self._org_headers(org_id, require_org)
        hdrs.update(headers or {})
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = httpx.Timeout(timeout, connect=10.0)
        try:
            response = self._client.request(
                method,
                path.lstrip("/"),
                headers=hdrs,
                params=self._clean(params),
                json=json,
                content=content,
                **extra,
            )
        except httpx.HTTPError as exc:
            raise AdmiralAPIError(0, f"request failed: {exc}") from exc
        if response.status_code >= 400:
            raise _api_error(response)
        return response

    def stream_events(
        self,
        path: str,
        on_event: Callable[[str, Any], bool],
        *,
        org_id: str | None = None,
        params: dict[str, Any] | None = None,
        timeout_s: float = 60.0,
        require_org: bool = True,
    ) -> dict[str, Any]:
        """Consume a server-sent-event stream until on_event returns True,
        the server closes the stream, or timeout_s elapses.

        on_event receives (event_name, data) with data JSON-decoded when
        possible. Returns {"ended": "stop"|"closed"|"timeout", "elapsed_s",
        "events"}. The HTTP read runs in a daemon worker thread and the
        caller waits on a queue with the remaining time, so a stream that is
        silent between keepalives (25 s) cannot hold the call past timeout_s
        (closing a blocked httpx response from another thread does not
        reliably unblock the read). on_event only ever runs on the caller's
        thread.
        """
        import json as _json
        import queue

        timeout_s = max(float(timeout_s), 0.1)
        hdrs = self._org_headers(org_id, require_org)
        hdrs["Accept"] = "text/event-stream"
        hdrs["Cache-Control"] = "no-cache"
        started = time.monotonic()
        deadline = started + timeout_s
        items: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        stop = threading.Event()
        holder: dict[str, httpx.Response] = {}

        def worker() -> None:
            try:
                with self._client.stream(
                    "GET",
                    path.lstrip("/"),
                    headers=hdrs,
                    params=self._clean(params),
                    timeout=httpx.Timeout(connect=10.0, read=timeout_s + 30.0, write=10.0, pool=10.0),
                ) as response:
                    holder["response"] = response
                    if response.status_code >= 400:
                        response.read()
                        items.put(("error", _api_error(response), None))
                        return
                    items.put(("open", response.status_code, None))

                    def lines() -> Iterator[str]:
                        for line in response.iter_lines():
                            if stop.is_set():
                                return
                            yield line

                    for name, raw in parse_sse(lines()):
                        if stop.is_set():
                            return
                        items.put(("event", name, raw))
                items.put(("end", None, None))
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller unless it stopped
                if not stop.is_set():
                    items.put(("error", AdmiralAPIError(0, f"stream failed: {exc}"), None))

        thread = threading.Thread(target=worker, name="admrl-sse", daemon=True)
        thread.start()
        ended = "timeout"
        count = 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    kind, a, b = items.get(timeout=remaining)
                except queue.Empty:
                    break
                if kind == "error":
                    if count == 0:
                        raise a
                    # Keep what was collected; report the mid-stream failure.
                    ended = f"error: {a}"
                    break
                if kind == "end":
                    ended = "closed"
                    break
                if kind == "event":
                    try:
                        data: Any = _json.loads(b)
                    except ValueError:
                        data = b
                    count += 1
                    if on_event(a, data):
                        ended = "stop"
                        break
        finally:
            stop.set()
            response = holder.get("response")
            if response is not None and thread.is_alive():
                try:
                    response.close()
                except Exception:  # noqa: BLE001 - best effort; the daemon worker exits on its next read
                    pass
        return {"ended": ended, "elapsed_s": round(time.monotonic() - started, 2), "events": count}

    # ------------------------------------------------- state / documents ---

    def get_device_state(self, device_id: str, *, org_id: str | None = None, live: bool = False) -> Any:
        # live=1 asks the device for its cached snapshot (backend bounds it at 8 s).
        return self.get(
            f"devices/{device_id}/state",
            org_id=org_id,
            params={"live": "1"} if live else None,
            timeout=20.0 if live else None,
        )

    def get_device_document(
        self, device_id: str, *, org_id: str | None = None, fmt: str = "json"
    ) -> httpx.Response:
        accept = "application/yaml" if fmt == "yaml" else "application/json"
        return self.raw_request("GET", f"devices/{device_id}/document", org_id=org_id, headers={"Accept": accept})

    def patch_device_document(
        self,
        device_id: str,
        merge_patch: Any,
        *,
        org_id: str | None = None,
        if_match: str | None = None,
        change_reason: str | None = None,
    ) -> httpx.Response:
        import json as _json

        headers = {"Content-Type": "application/merge-patch+json", "Accept": "application/json"}
        if if_match:
            headers["If-Match"] = if_match if if_match.startswith(('"', "W/")) else f'"{if_match}"'
        if change_reason:
            headers["X-Change-Reason"] = change_reason[:500]
        return self.raw_request(
            "PATCH",
            f"devices/{device_id}/document",
            org_id=org_id,
            content=_json.dumps(merge_patch).encode(),
            headers=headers,
        )

    def render_device_document(self, device_id: str, *, org_id: str | None = None, dry_run: bool = True) -> Any:
        # Without dryRun the backend also pushes the rendered bundle.
        return self.post(
            f"devices/{device_id}/document:render",
            org_id=org_id,
            params={"dryRun": "1"} if dry_run else None,
        )

    def adopt_local_override(
        self, device_id: str, *, org_id: str | None = None, if_match: str | None = None
    ) -> httpx.Response:
        headers = {"Accept": "application/json"}
        if if_match:
            headers["If-Match"] = if_match if if_match.startswith(('"', "W/")) else f'"{if_match}"'
        return self.raw_request(
            "POST", f"devices/{device_id}/document:adoptLocalOverride", org_id=org_id, headers=headers
        )

    def discard_local_override(
        self, device_id: str, *, org_id: str | None = None, paths: list[str] | None = None
    ) -> Any:
        return self.post(
            f"devices/{device_id}/document:discardLocalOverride",
            org_id=org_id,
            json={"paths": list(paths or ["*"])},
        )

    def diagnose(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}/diagnose", org_id=org_id)

    def duplicate_fleet(
        self,
        fleet_id: str,
        name: str,
        *,
        org_id: str | None = None,
        description: str | None = None,
        copy_configuration: bool = True,
    ) -> Any:
        body: dict[str, Any] = {"name": name, "copyConfiguration": bool(copy_configuration)}
        if description:
            body["description"] = description
        return self.post(f"fleets/{fleet_id}/duplicate", org_id=org_id, json=body)

    def duplicate_configuration(
        self,
        config_id: str,
        name: str,
        *,
        org_id: str | None = None,
        description: str | None = None,
        version: int | None = None,
    ) -> Any:
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        if version:
            body["version"] = int(version)  # absent/0 = latest
        return self.post(f"configurations/{config_id}/duplicate", org_id=org_id, json=body)

    def probe_device(
        self,
        device_id: str,
        *,
        org_id: str | None = None,
        sections: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        body: dict[str, Any] = {}
        if sections:
            body["sections"] = list(sections)
        if timeout_ms:
            body["timeoutMs"] = int(timeout_ms)
        # Backend bounds the probe at 25 s.
        return self.post(
            f"devices/{device_id}/diagnostics/probe",
            org_id=org_id,
            json=body,
            timeout=35.0,
        )

    # ----------------------------------------------------------- rollouts ---

    def create_rollout(self, body: dict[str, Any], *, org_id: str | None = None) -> Any:
        return self.post("rollouts", org_id=org_id, json=body)

    def get_rollout(self, rollout_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"rollouts/{rollout_id}", org_id=org_id)

    def list_rollout_devices(
        self,
        rollout_id: str,
        *,
        org_id: str | None = None,
        status: str | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> Any:
        return self.get(
            f"rollouts/{rollout_id}/devices",
            org_id=org_id,
            params={"status": status, "page": page, "limit": limit},
        )

    ROLLOUT_ACTIONS = ("pause", "resume", "cancel", "rollback")

    def rollout_control(
        self, rollout_id: str, action: str, *, org_id: str | None = None, reason: str | None = None
    ) -> Any:
        normalised = (action or "").strip().lower()
        if normalised not in self.ROLLOUT_ACTIONS:
            raise ValueError(f"Unsupported rollout action {action!r}; one of {self.ROLLOUT_ACTIONS}")
        body = {"reason": reason[:500]} if reason else {}
        return self.post(f"rollouts/{rollout_id}/{normalised}", org_id=org_id, json=body)

    def list_configurations(self, *, org_id: str | None = None, search: str | None = None, limit: int = 100) -> Any:
        return self.get("configurations", org_id=org_id, params={"search": search, "limit": limit})

    def list_organisations(self) -> list[dict[str, Any]]:
        payload = self.get("user/organisations", require_org=False)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("organisations", "organizations", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        return []

    def list_fleets(
        self,
        *,
        org_id: str | None = None,
        search: str | None = None,
        tag: str | None = None,
        page: int = 1,
        limit: int = 100,
    ) -> Any:
        return self.get(
            "fleets",
            org_id=org_id,
            params={"search": search, "tag": tag, "page": page, "limit": limit},
        )

    def list_devices(
        self,
        *,
        org_id: str | None = None,
        search: str | None = None,
        status: str | None = None,
        fleet_id: str | None = None,
        page: int = 1,
        limit: int = 100,
        sort: str | None = None,
        direction: str | None = None,
    ) -> Any:
        return self.get(
            "devices",
            org_id=org_id,
            params={
                "search": search,
                "status": status,
                "fleet_id": fleet_id,
                "page": page,
                "limit": min(limit, 100),
                "sort": sort,
                "direction": direction,
            },
        )

    def get_device(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}", org_id=org_id)

    def get_device_network(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}/network/status", org_id=org_id)

    def get_device_specs(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}/specifications", org_id=org_id)

    def get_device_stats(self, device_id: str, *, org_id: str | None = None, include_metrics: bool = True) -> Any:
        return self.get(
            f"devices/{device_id}/device-stats",
            org_id=org_id,
            params={"include_metrics": str(include_metrics).lower()},
        )

    def get_device_system_services(
        self,
        device_id: str,
        *,
        org_id: str | None = None,
        name: str | None = None,
        deaths: int | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if name:
            params["name"] = name
        if deaths is not None:
            params["deaths"] = deaths
        return self.get(f"devices/{device_id}/system-services", org_id=org_id, params=params or None)

    def get_device_workload(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}/workload", org_id=org_id)

    def get_device_metrics(
        self,
        device_id: str,
        *,
        metric_name: str,
        fleet_id: str,
        start: str,
        end: str,
        org_id: str | None = None,
    ) -> Any:
        # Historical timeseries (VictoriaMetrics query_range). start/end are
        # RFC3339 — unix seconds are rejected with 400 "Invalid start time".
        # fleet_id is required context, not optional filtering.
        return self.get(
            f"devices/{device_id}/metrics",
            org_id=org_id,
            params={
                "metric_name": metric_name,
                "fleet_id": fleet_id,
                "start": start,
                "end": end,
            },
        )

    def get_device_screenshot(
        self,
        device_id: str,
        *,
        org_id: str | None = None,
        display: int = 0,
    ) -> Any:
        # Live capture: the platform asks the edge to grab its display now and
        # waits for the comms round-trip. Slower than any cached endpoint and
        # 504s when the device does not answer, so allow a longer read timeout.
        # The edge chooses the encoding — it cannot be requested; the response
        # `format` reports what actually came back.
        return self.get(
            f"devices/{device_id}/screenshot",
            org_id=org_id,
            params={"display": display},
            timeout=90.0,
        )

    def get_device_events(
        self,
        device_id: str,
        *,
        org_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 50,
        offset: int = 0,
        event: str | None = None,
        source: str | None = None,
    ) -> Any:
        return self.get(
            f"devices/{device_id}/events",
            org_id=org_id,
            params={
                "start": start,
                "end": end,
                "limit": limit,
                "offset": offset,
                "event": event,
                "source": source,
            },
        )

    def get_device_logs(
        self,
        device_id: str,
        *,
        org_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        level: str | None = None,
    ) -> Any:
        # Historical VictoriaLogs query. Works for offline devices.
        # Live tail is a JWT websocket at /v1/ws/devices/{id}/logs/stream and
        # cannot be authenticated with a PAT.
        return self.get(
            f"devices/{device_id}/logs",
            org_id=org_id,
            params={"start": start, "end": end, "level": level},
        )

    def query_logs(
        self,
        *,
        org_id: str | None = None,
        device_id: str | None = None,
        fleet_id: str | None = None,
        start_time: str,
        end_time: str,
        level: str | None = None,
        source: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> Any:
        body = {
            "device_id": device_id,
            "fleet_id": fleet_id,
            "level": level,
            "source": source,
            "search": search,
            "start_time": start_time,
            "end_time": end_time,
            "limit": limit,
        }
        body = {k: v for k, v in body.items() if v is not None and v != ""}
        return self.post("metrics/logs/query", org_id=org_id, json=body)

    def fetch_logs(
        self,
        *,
        device_id: str,
        org_id: str | None = None,
        start: str,
        end: str,
        level: str | None = None,
        source: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Historical logs. Prefer VictoriaLogs query; PAT often 403s that path."""
        query_error: str | None = None
        used = "query"
        payload: Any = None
        cap = min(max(limit, 1), 1000)
        try:
            payload = self.query_logs(
                org_id=org_id,
                device_id=device_id,
                start_time=start,
                end_time=end,
                level=level,
                source=source,
                search=search,
                limit=cap,
            )
        except AdmiralAPIError as exc:
            if exc.status not in (401, 403, 404):
                raise
            query_error = str(exc)
            used = "device_logs"
            payload = self.get_device_logs(
                device_id,
                org_id=org_id,
                start=start,
                end=end,
                level=level,
            )
        entries = _log_entries(payload)
        if search and used == "device_logs":
            needle = search.lower()
            entries = [
                e
                for e in entries
                if needle in str(e.get("message") or e.get("msg") or "").lower()
                or needle in str(e.get("source") or "").lower()
            ]
        newest = _newest_first(entries, cap)
        has_more = False
        if isinstance(payload, dict):
            has_more = bool(payload.get("has_more") or payload.get("hasMore"))
        if len(entries) > len(newest):
            has_more = True
        return {
            "logs": newest,
            "has_more": has_more,
            "source": used,
            "query_error": query_error,
        }

    def get_fleet_health(self, fleet_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"fleets/{fleet_id}/health", org_id=org_id)

    def get_fleet_logs(
        self,
        fleet_id: str,
        *,
        org_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> Any:
        return self.get(
            f"fleets/{fleet_id}/logs",
            org_id=org_id,
            params={"start": start, "end": end},
        )

    def search(self, query: str, *, org_id: str | None = None, limit: int = 20) -> Any:
        return self.get("search", org_id=org_id, params={"q": query, "limit": limit})

    def reboot_device(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.post(f"devices/{device_id}/reboot", org_id=org_id, json={})

    # Memory tests (online modes only; a test boot is operator-only and is not exposed here).
    def start_memory_test(self, device_id: str, body: dict[str, Any], *, org_id: str | None = None) -> Any:
        return self.post(f"devices/{device_id}/memory-test", org_id=org_id, json=body, timeout=45.0)

    def cancel_memory_test(self, device_id: str, *, run_id: str | None = None, org_id: str | None = None) -> Any:
        return self.post(
            f"devices/{device_id}/memory-test/cancel",
            org_id=org_id,
            json={"runId": run_id} if run_id else {},
            timeout=45.0,
        )

    def get_memory_test(self, device_id: str, *, org_id: str | None = None) -> Any:
        return self.get(f"devices/{device_id}/memory-test", org_id=org_id, timeout=35.0)

    def list_memory_test_results(self, device_id: str, *, limit: int | None = None, org_id: str | None = None) -> Any:
        return self.get(
            f"devices/{device_id}/memory-test/results",
            org_id=org_id,
            params={"limit": limit} if limit is not None else None,
        )

    # Edge-executed workload lifecycle actions. The backend forwards `action`
    # verbatim (SendDeviceWorkloadCommandHandler) and admrl-init's handleCommand
    # switch matches uppercase, so canonicalise on the way out. Swagger's enum
    # (START,STOP,RESTART,DELETE) is stale: RECREATE is live, and DELETE
    # (wipes the edge's applied config) and RESET (full device wipe) are
    # deliberately not exposed through this client.
    WORKLOAD_ACTIONS = ("start", "stop", "restart", "recreate")

    def workload_command(self, device_id: str, *, action: str, org_id: str | None = None) -> Any:
        normalised = (action or "").strip().lower()
        if normalised not in self.WORKLOAD_ACTIONS:
            raise ValueError(
                f"Unsupported workload action {action!r}; one of {self.WORKLOAD_ACTIONS}"
            )
        # The edge executes the command synchronously under the handler's 60s
        # budget — the 30s default read timeout would cut slow edges off first.
        return self.put(
            f"devices/{device_id}/workload/cmd",
            org_id=org_id,
            json={"action": normalised.upper()},
            timeout=70.0,
        )


def _log_stamp(entry: dict[str, Any]) -> str:
    return str(entry.get("timestamp") or entry.get("time") or "")


def parse_rfc3339(value: str) -> datetime:
    """Parse RFC3339 UTC timestamps, tolerating sub-microsecond fractional seconds."""
    stamp = value.strip()
    if "." in stamp:
        head, _, tail = stamp.partition(".")
        frac = ""
        rest = tail
        while rest and rest[0].isdigit():
            frac += rest[0]
            rest = rest[1:]
        stamp = f"{head}.{(frac + '000000')[:6]}{rest}"
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def log_page_cursor(logs: list[dict[str, Any]], has_more: bool) -> str | None:
    """Cursor just older than the oldest returned line, for paging older history.

    Returns an RFC3339 timestamp to pass as `before` on the next
    get_device_logs call, or None when the page is the oldest one.
    """
    if not has_more or not logs:
        return None
    stamps = [parse_rfc3339(_log_stamp(e)) for e in logs if _log_stamp(e)]
    if not stamps:
        return None
    oldest = min(stamps)
    cursor = oldest - timedelta(milliseconds=1)
    return cursor.isoformat().replace("+00:00", "Z")


def url_for(settings: Settings, path: str) -> str:
    return urljoin(settings.api_base.rstrip("/") + "/", path.lstrip("/"))
