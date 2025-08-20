from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    openai_api_key: str = ""
    default_model: str = "gpt-4o-mini"
    max_output_tokens: int = 800
    temperature: float = 0.2
    max_concurrent_requests: int = 10
    processing_batch_size: int = 50
    request_timeout: float = 45.0
    max_attempts: int = 4
    initial_backoff: float = 2.0
    backoff_factor: float = 1.7

    def __post_init__(self) -> None:
        load_dotenv(find_dotenv())
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.default_model = os.getenv("DEFAULT_MODEL", self.default_model)
        self.max_output_tokens = int(
            os.getenv("MAX_OUTPUT_TOKENS", os.getenv("MAX_TOKENS", self.max_output_tokens))
        )
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", self.max_concurrent_requests))
        self.processing_batch_size = int(os.getenv("PROCESSING_BATCH_SIZE", self.processing_batch_size))
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", self.request_timeout))
        self.max_attempts = int(os.getenv("MAX_ATTEMPTS", self.max_attempts))
        self.initial_backoff = float(os.getenv("INITIAL_BACKOFF", self.initial_backoff))
        self.backoff_factor = float(os.getenv("BACKOFF_FACTOR", self.backoff_factor))
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.processing_batch_size <= 0:
            raise ValueError("processing_batch_size must be positive")

