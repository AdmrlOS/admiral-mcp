from __future__ import annotations

from pathlib import Path

from admrl_mcp.config import ConfigError, load_settings, require_settings


def test_empty_and_placeholder_env_fill_from_dotenv(tmp_path: Path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ADMRL_API_TOKEN_ID=from-file-token\n"
        "ADMRL_API_SECRET_KEY=from-file-secret\n"
        "ADMRL_ORG_ID=from-file-org\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("admrl_mcp.config._project_dotenv_path", lambda: dotenv)
    monkeypatch.setenv("ADMRL_API_TOKEN_ID", "")
    monkeypatch.setenv("ADMRL_API_SECRET_KEY", "${ADMRL_API_SECRET_KEY}")
    monkeypatch.setenv("ADMRL_ORG_ID", "${ADMRL_ORG_ID}")
    monkeypatch.delenv("ADMRL_API_BASE", raising=False)

    settings = load_settings()
    assert settings.token_id == "from-file-token"
    assert settings.secret_key == "from-file-secret"
    assert settings.org_id == "from-file-org"


def test_real_process_env_wins_over_dotenv(tmp_path: Path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ADMRL_API_TOKEN_ID=from-file-token\n"
        "ADMRL_API_SECRET_KEY=from-file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("admrl_mcp.config._project_dotenv_path", lambda: dotenv)
    monkeypatch.setenv("ADMRL_API_TOKEN_ID", "process-token")
    monkeypatch.setenv("ADMRL_API_SECRET_KEY", "process-secret")
    monkeypatch.delenv("ADMRL_ORG_ID", raising=False)

    settings = load_settings()
    assert settings.token_id == "process-token"
    assert settings.secret_key == "process-secret"


def test_require_settings_still_fails_when_blank(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("admrl_mcp.config._project_dotenv_path", lambda: tmp_path / "missing.env")
    monkeypatch.setenv("ADMRL_API_TOKEN_ID", "${ADMRL_API_TOKEN_ID}")
    monkeypatch.setenv("ADMRL_API_SECRET_KEY", "")
    try:
        require_settings()
        raise AssertionError("expected ConfigError")
    except ConfigError as exc:
        assert "ADMRL_API_TOKEN_ID" in str(exc)
        assert "ADMRL_API_SECRET_KEY" in str(exc)
