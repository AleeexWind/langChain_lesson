"""
Configuration loaded from environment variables.

Copy `.env.example` to `.env` and adjust values. Never commit `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the LLM and Articles API."""

    lm_studio_base_url: str
    lm_studio_api_key: str
    lm_studio_model: str
    articles_api_base_url: str


def get_settings() -> Settings:
    return Settings(
        lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        # Placeholder only — LM Studio ignores the key but the OpenAI client requires one.
        lm_studio_api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        lm_studio_model=os.getenv("LM_STUDIO_MODEL", "local-model"),
        articles_api_base_url=os.getenv(
            "ARTICLES_API_BASE_URL",
            "https://jsonplaceholder.typicode.com",
        ).rstrip("/"),
    )
