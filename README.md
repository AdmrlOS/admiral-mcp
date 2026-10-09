# Admiral MCP server

Talk to an Admiral fleet from Hermes (or any MCP client) using a Personal API Token.

Natural questions this is built for:

- "What's the IP of my printer?"
- "Which devices are offline?"
- "Grab logs from the crashing kiosk and tell me why"

It is **not** a 1:1 dump of the swagger file. Tools resolve devices by name, tag, notes, IP, or UUID, then call the real Admiral API.

## Auth

PAT only (for now):

| Header | Env var |
|---|---|
| `X-API-Token-ID` | `ADMRL_API_TOKEN_ID` |
| `X-API-Secret-Key` | `ADMRL_API_SECRET_KEY` |
| `X-Organization-ID` | `ADMRL_ORG_ID` (optional if the token sees one org) |

Create a token in **app.admrl.co → Settings → API Tokens**. The secret is shown once.

Copy `.env.example` and fill it in, or put the same keys in `~/.hermes/.env` (preferred for Hermes — stdio MCP subprocesses only inherit env you pass explicitly). Empty `${ADMRL_*}` placeholders from Hermes config are ignored so a project `.env` still works.

## Logs

Two Admiral log surfaces exist:

| Surface | Path | Auth | Offline devices |
|---|---|---|---|
| Historical query (VictoriaLogs) | `GET /v1/devices/{id}/logs`, `POST /v1/metrics/logs/query` | PAT | Yes |
| Live websocket tail | `wss://api.admrl.co/v1/ws/devices/{id}/logs/stream` | Firebase/JWT first-message `auth` | No |

This server uses the historical query. Live tail cannot be opened with a PAT — that is how `DeviceLogsTab.svelte` authenticates today.

## Tools

| Tool | Use |
|---|---|
| `list_organisations` | Pick an org when `ADMRL_ORG_ID` is unset |
| `list_fleets` | Fleet list, optional search/tag |
| `list_devices` | Filter by status / fleet / query |
| `find_device` | Resolve name, tag (`role=printer`), IP, UUID |
| `get_device` | Detail + effective tags |
| `get_device_network` | IPs from list cache, spec, and live status |
| `get_device_specs` | Hardware / versions |
| `get_device_workload` | Container state + live config identity |
| `get_device_stats` | Health gauges |
| `get_device_screenshot` | Live display capture (metadata + image block) |
| `get_device_logs` | Historical logs + crash-signature distill |
| `get_device_events` | Lifecycle events |
| `diagnose_device` | Offline/error sweep, or one device: stats + events + logs |
| `troubleshoot_device` | "What is wrong with this device?" in one call: state, diagnosis, probe, logs, events, workload, storage, time, connectivity, drift → ranked findings with steps and docs links |
| `check_device_connectivity` | Link, DNS, TCP, NTP/clock and NATS/transport checks plus the classic causes |
| `explain_workload_failure` | Crash loops, exit codes, pull/signature/USB denials, OOM, with scrubbed log excerpts |
| `fleet_health_report` | Devices by status and grouped by top problem for a fleet or organisation (bounded) |
| `get_fleet_metrics` | Fleet CPU/memory/disk/network over a window: avg, max, latest, device count; `per_device=true` ranks devices; `device=` compares one device with its fleet |
| `get_fleet_health` | Fleet snapshot: devices online/offline and average CPU, memory, disk |
| `get_fleet_uptime` | Fleet uptime: current percentage plus hourly/daily/weekly/monthly buckets |
| `get_org_metrics` | Organisation-wide metric by fleet or device (avg/max, top N, optional fleet filter) with names resolved |
| `query_telemetry_metrics` | Advanced: bounded read-only PromQL (instant or range) through the organisation-scoped Telemetry API |
| `get_telemetry_scope` | What telemetry the caller can query (org-wide or specific fleets/devices) |
| `search` | Global search |
| `reboot_device` | Destructive; only on explicit request |
| `start_memory_test` | RAM test. `mode=live` keeps the workload running (`quick`, `passes` optional); `mode=full_online` stops the workload for the run and needs `confirm=true`. A test boot is not offered here: start it from the dashboard or the device console |
| `cancel_memory_test` | Stop the running memory test (restarts the workload if it was held) |
| `get_memory_test` | Status, coverage %, errors, verdict, retired pages, memory fault, capabilities; flags a fault or an interrupted last test |
| `list_memory_test_results` | Stored results, newest first (`limit` 1-100); works offline |

### Operations: configurations, fleets, devices, rollouts

Mutating tools resolve names (ambiguous names return candidates and change nothing), refuse no-ops, and read the
state back from the API after writing. They run only on an explicit request.

| Tool | Use |
|---|---|
| `list_configurations` | Configurations with status, latest version and the fleets using each |
| `get_configuration` | Metadata, spec of the latest (or a given) version, version history, fleets (credential-looking env values masked) |
| `diff_configuration_versions` | Structured spec diff between two versions (default: previous → latest) |
| `create_configuration` | New configuration (image/env/ports/command or a full spec) at version 1 |
| `edit_configuration` | Edit the latest spec into a **new version**: `image`/`image_tag`, `env_set`/`env_unset`, `merge_patch`, or full `spec`. Needs `change_reason`; `base_version` guards concurrent edits; `dry_run` previews; shows which fleets follow `latest` vs pinned |
| `update_configuration_metadata` | Name, description, tags and lifecycle status |
| `rollback_configuration` | New latest version copied from an earlier one |
| `delete_configuration` | Refuses while a fleet or unfinished rollout still uses it |
| `get_fleet` | Devices, tags, assigned configuration (`latest` or pinned, resolved version), update policy/window, history, unfinished rollouts |
| `assign_fleet_configuration` | Assign a configuration/version to a fleet **directly** (no canary; use a rollout for running fleets) |
| `get_fleet_configuration_history` | Who assigned which configuration/version, when and why |
| `create_fleet`, `update_fleet` | Create a fleet; rename/describe/relocate and add/remove/replace tags |
| `set_fleet_update_policy` | OS update policy (`latest`/`pinned` targets) and update window |
| `update_device` | Name, notes, location, tags |
| `move_device_to_fleet` | Move a device; it takes the new fleet's configuration immediately |
| `get_device_configuration` | Inherited configuration, per-device override and the merged result |
| `set_device_configuration_override`, `clear_device_configuration_override` | Per-device override layered over the fleet configuration |
| `list_rollouts` | Compact rollout rows, filter by fleet/status/type |
| `preview_rollout` | Read-only plan: current vs target per fleet, spec diff, device impact, strategy, warnings, and the exact `create_rollout` call |
| `create_rollout` | `config` (default), `reboot`, `restart_workload` or `system_update` over one or more fleets |
| `get_rollout`, `list_rollout_devices`, `rollout_control`, `watch_rollout` | Inspect, pause/resume/cancel/rollback, and follow a rollout |

**Applying a configuration change.** Saving a new version or assigning a configuration to a fleet pushes nothing: a
device reads its desired state when it (re)connects and when a rollout, fleet move or document push reaches it. A
fleet that follows `latest` therefore picks a new version up unevenly and without a canary. For fleets with running
devices use `edit_configuration` → `preview_rollout` → `create_rollout`.

## Run locally

```bash
cd /path/to/admrl-mcp
uv sync --extra dev
uv run pytest
```

Stdio server:

```bash
export ADMRL_API_TOKEN_ID=...
export ADMRL_API_SECRET_KEY=...
export ADMRL_ORG_ID=...   # optional
uv run admrl-mcp
```

## Hosted mode (remote connector, streamable HTTP + OAuth 2.1)

`admrl-mcp-http` serves the same tools (minus the SSE `watch_*` tools) as a stateless streamable-HTTP MCP
server at `/mcp`, acting as an OAuth 2.1 protected resource. It never holds a PAT or shared credential: every
request must carry `Authorization: Bearer admrl_mcp_at_...`, and that token is forwarded to the Admiral API
for that request only (the backend validates it, enforces scope and organisation).

```bash
ADMRL_API_BASE=https://api.admrl.co/v1 uv run admrl-mcp-http        # listens on :8080
curl localhost:8080/healthz
docker build -t admrl-mcp . && docker run --rm --read-only -p 8080:8080 admrl-mcp
```

Endpoints: `/mcp` (401 + `WWW-Authenticate: Bearer resource_metadata=..., scope="admrl:read"` without a valid-looking
bearer; a backend 401 during a tool call is returned the same way so the client refreshes), `/healthz`,
`/.well-known/oauth-protected-resource[/mcp]`, `/.well-known/oauth-authorization-server`.

| Env var | Default | Purpose |
|---|---|---|
| `ADMRL_MCP_RESOURCE_URL` | `https://mcp.admrl.co/mcp` | resource identifier; its host is the allowed `Host` |
| `ADMRL_MCP_ISSUER` | `https://mcp.admrl.co` | OAuth issuer |
| `ADMRL_MCP_AUTHORIZE_URL` | `https://app.admrl.co/oauth/authorize` | consent UI |
| `ADMRL_MCP_PUBLIC_API_BASE` | `https://api.admrl.co/v1` | token/registration/revocation endpoint base |
| `ADMRL_API_BASE` | `https://api.admrl.co/v1` | API the server calls (may be the in-cluster service) |
| `ADMRL_MCP_ALLOWED_HOSTS` | (none) | extra comma-separated `Host` values (DNS-rebinding allow-list); `host:*` allowed |
| `ADMRL_MCP_ALLOWED_ORIGINS` | claude.ai, claude.com, chatgpt.com, issuer | extra allowed `Origin`s |
| `ADMRL_MCP_DNS_REBINDING_PROTECTION` | `true` | set `false` only for local debugging |
| `ADMRL_MCP_MAX_BODY_BYTES` | `1048576` | request body limit (413 above) |
| `ADMRL_MCP_MAX_THREADS` | `64` | worker threads for sync tools |
| `ADMRL_MCP_HOST` / `ADMRL_MCP_PORT` | `0.0.0.0` / `8080` | bind address |
| `ADMRL_MCP_LOG_LEVEL` | `INFO` | JSON logs on stderr; tokens are never logged |

Organisation: tools take `organization_id` as before; when omitted no `X-Organization-ID` is sent and the
backend uses the grant's organisation. The image is `ghcr.io/admrlos/admiral-mcp` (built by
`.github/workflows/image.yaml`: `:main`, `:vX.Y.Z`, `:X.Y`, `:sha-<short>`).

## Hermes

```bash
hermes mcp add admrl \
  --command uv \
  --args --directory /path/to/admrl-mcp run admrl-mcp \
  --env ADMRL_API_TOKEN_ID \
  --env ADMRL_API_SECRET_KEY \
  --env ADMRL_ORG_ID
```

Hermes prefixes tools as `mcp_admrl_*`. Restart the desktop app after adding.

## In the browser (Admiral dashboard assistant)

`admrl_mcp.browser` runs the same tools inside Pyodide in a Web Worker, authenticated with the
dashboard user's session token (`Authorization: Bearer` + `X-Organization-ID`) instead of a PAT.
HTTP goes through a synchronous-XHR httpx transport; `set_auth()` is called before every tool call.
Tools that cannot work there (`watch_*` SSE streams) are not offered
(`browser.EXCLUDED_TOOLS`). The PAT/stdio server is unchanged. The dashboard builds the wheel set
with `npm run mcp:bundle`; see `admiral-dashboard/src/lib/assistant/runtime/README.md`.

## Extensions

Extra tool packages can be plugged in without changing this package. An extension is an importable
module with `register(mcp: FastMCP) -> list[str]` that adds tools and returns their names. Every tool it
adds must set `ToolAnnotations(readOnlyHint=...)` explicitly, or the extension is rejected. Optional module
attributes: `INSTRUCTIONS` (text appended to the server instructions) and `BROWSER_EXCLUDED_TOOLS`
(`{tool: reason}`, tools the browser must not offer).

- Python: `admrl_mcp.extensions.register_extension(module_name, mcp) -> list[str]`.
- stdio: `ADMRL_MCP_EXTENSIONS=mod1,mod2 admrl-mcp`. A module that is not installed is logged to stderr
  and skipped; the server still starts.
- Browser (Pyodide): `admrl_mcp.browser.load_extension(module_name) -> str` (synchronous) registers the
  module on the browser server and returns `{"tools": [...]}`. The open in-memory MCP session sees the
  new tools on its next `list_tools`; call it before the host freezes its tool list. The module's wheel
  must already be installed (e.g. `micropip.install(..., deps=False)` from the Pyodide filesystem).
