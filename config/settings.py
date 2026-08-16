"""
Global settings loaded from config.yaml and .env
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent


class AppSettings(BaseSettings):
    # Zerodha
    ZERODHA_API_KEY: str = Field(default="", env="ZERODHA_API_KEY")
    ZERODHA_API_SECRET: str = Field(default="", env="ZERODHA_API_SECRET")
    ZERODHA_ACCESS_TOKEN: str = Field(default="", env="ZERODHA_ACCESS_TOKEN")
    ZERODHA_TOTP_SECRET: str = Field(default="", env="ZERODHA_TOTP_SECRET")

    # Angel One
    ANGEL_API_KEY: str = Field(default="", env="ANGEL_API_KEY")
    ANGEL_CLIENT_ID: str = Field(default="", env="ANGEL_CLIENT_ID")
    ANGEL_PASSWORD: str = Field(default="", env="ANGEL_PASSWORD")
    ANGEL_TOTP_SECRET: str = Field(default="", env="ANGEL_TOTP_SECRET")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: str = Field(default="", env="TELEGRAM_CHAT_ID")

    class Config:
        env_file = str(ROOT_DIR / ".env")
        extra = "allow"


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()


@lru_cache()
def get_config() -> dict:
    config_path = ROOT_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


settings = get_settings()
config = get_config()
