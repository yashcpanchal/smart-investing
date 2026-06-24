"""Runtime configuration, loaded from environment / .env (gitignored)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional — LLM-driven phases only (5, 6). Engine runs without either.
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # SEC EDGAR fair-access requires a descriptive User-Agent (name + email).
    sec_user_agent: str = "smart-investing example@example.com"

    # Robinhood Agentic MCP (Phase 11). We are the client/host.
    robinhood_mcp_url: str = "https://agent.robinhood.com/mcp/trading"

    # Used by the build agent to push to a dev branch, if provided.
    github_token: str | None = None

    # Annual risk-free rate used in Sharpe / max-Sharpe optimization.
    risk_free_rate: float = 0.04

    # Local cache / storage directory.
    data_dir: str = "data"


settings = Settings()
