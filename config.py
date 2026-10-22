"""Env-driven settings plus loaders for the JSON config under data/."""
from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from engine.schemas import PolicyRules

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["mock", "anthropic", "openai", "ollama"] = "mock"
    llm_model: str = ""
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"

    database_url: str = f"sqlite:///{ROOT / 'data' / 'app.db'}"
    data_dir: Path = ROOT / "data"
    checkpoint_path: Path = ROOT / "data" / "checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_policy_rules(data_dir: Path) -> PolicyRules:
    # parse_float=Decimal keeps money thresholds exact if written as JSON numbers.
    raw = json.loads((data_dir / "policy_rules.json").read_text(), parse_float=Decimal)
    if isinstance(raw.get("confidence_threshold"), Decimal):
        raw["confidence_threshold"] = float(raw["confidence_threshold"])
    return PolicyRules.model_validate(raw)


def load_customers(data_dir: Path) -> dict[str, dict[str, str]]:
    return json.loads((data_dir / "customers.json").read_text())


def load_known_resolution_codes(data_dir: Path) -> frozenset[str]:
    items = json.loads((data_dir / "known_resolutions.json").read_text())
    return frozenset(item["code"] for item in items)


def load_policy_kb(data_dir: Path) -> str:
    return (data_dir / "policy_kb.md").read_text()
