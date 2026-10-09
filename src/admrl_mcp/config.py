from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_API_BASE = "https://api.admrl.co/v1"
AUTH_PAT = "pat"
AUTH_BEARER = "bearer"
_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Z0-9_]+\}$")
_CRED_KEYS = (
    "ADMRL_API_BASE",
    "ADMRL_API_TOKEN_ID",
    "ADMRL_API_SECRET_KEY",
    "ADMRL_ORG_ID",
)


class ConfigError(RuntimeError):
    """Raised when required Admiral credentials are missing."""


@dataclass(frozen=True)
class Settings:
    api_base: str
    token_id: str
    secret_key: str
    org_id: str | None
    # "pat": X-API-Token-ID / X-API-Secret-Key headers (stdio, default).
    # "bearer": Authorization: Bearer <bearer_token> (dashboard session in the
    # browser). The token is refreshed hourly, so the client reads it from its
    # current Settings on every request instead of capturing it.
    auth_mode: str = AUTH_PAT
    bearer_token: str = ""
    # Hosted mode: the grant is bound to one organisation server-side, so a missing
    # organisation is not an error; no X-Organization-ID is sent and the backend defaults it.
    org_optional: bool = False

    @property
    def configured(self) -> bool:
        if self.auth_mode == AUTH_BEARER:
            return bool(self.bearer_token)
        return bool(self.token_id and self.secret_key)


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _usable(value: str | None) -> str:
    """Drop blanks and unexpanded ${VAR} placeholders Hermes forwards."""
    cleaned = _clean(value)
    if not cleaned or _PLACEHOLDER_RE.match(cleaned):
        return ""
    return cleaned


def _project_dotenv_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _load_dotenv() -> None:
    """Fill unset / placeholder ADMRL_* from the project .env.

    Hermes forwards ``mcp_servers.admrl.env`` as empty strings or literal
    ``${ADMRL_*}`` when ``~/.hermes/.env`` has no PAT. Those must not win
    over a real project ``.env``. Already-set real values stay in place.
    """
    path = _project_dotenv_path()
    if not path.is_file():
        return
    file_values = _parse_dotenv(path)
    for key in _CRED_KEYS:
        if _usable(os.environ.get(key)):
            continue
        file_value = _usable(file_values.get(key))
        if file_value:
            os.environ[key] = file_value


def load_settings() -> Settings:
    _load_dotenv()
    return Settings(
        api_base=_usable(os.environ.get("ADMRL_API_BASE")) or DEFAULT_API_BASE,
        token_id=_usable(os.environ.get("ADMRL_API_TOKEN_ID")),
        secret_key=_usable(os.environ.get("ADMRL_API_SECRET_KEY")),
        org_id=_usable(os.environ.get("ADMRL_ORG_ID")) or None,
    )


def require_settings() -> Settings:
    settings = load_settings()
    missing = []
    if not settings.token_id:
        missing.append("ADMRL_API_TOKEN_ID")
    if not settings.secret_key:
        missing.append("ADMRL_API_SECRET_KEY")
    if missing:
        raise ConfigError(
            "Admiral PAT is not configured. Set "
            + ", ".join(missing)
            + " (from app.admrl.co → Settings → API Tokens). "
            "Optional: ADMRL_ORG_ID, ADMRL_API_BASE."
        )
    return settings
