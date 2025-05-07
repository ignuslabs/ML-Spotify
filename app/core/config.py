# app/core/config.py - MongoDB Version
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field, MongoDsn # Use MongoDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

logger = logging.getLogger(__name__) # Get logger instance

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables (MongoDB version).
    """
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    # --- Spotify API Credentials ---
    SPOTIPY_CLIENT_ID: str = Field(..., validation_alias="SPOTIPY_CLIENT_ID")
    SPOTIPY_CLIENT_SECRET: str = Field(..., validation_alias="SPOTIPY_CLIENT_SECRET")
    SPOTIPY_REDIRECT_URI: AnyHttpUrl = Field(..., validation_alias="SPOTIPY_REDIRECT_URI")

    # --- MongoDB Configuration ---
    MONGODB_URI: MongoDsn = Field(..., validation_alias="MONGODB_URI")
    MONGODB_DB_NAME: str = Field("playlist_recommender", validation_alias="MONGODB_DB_NAME")
    # Collection names (can be customized via env vars)
    MONGO_PLAYLIST_COLLECTION: str = Field("playlists", validation_alias="MONGO_PLAYLIST_COLLECTION")
    MONGO_TRACK_COLLECTION: str = Field("tracks", validation_alias="MONGO_TRACK_COLLECTION")
    MONGO_AUDIO_FEATURES_COLLECTION: str = Field("audio_features", validation_alias="MONGO_AUDIO_FEATURES_COLLECTION")

    # --- AWS Configuration (Only if needed for other services, e.g., Fargate region) ---
    AWS_REGION: str = Field("us-east-1", validation_alias="AWS_REGION")
    # S3_BUCKET_NAME: str = Field(..., validation_alias="S3_BUCKET_NAME") # REMOVED
    # AWS_ACCESS_KEY_ID: Optional[str] = Field(None, validation_alias="AWS_ACCESS_KEY_ID") # REMOVED (unless needed elsewhere)
    # AWS_SECRET_ACCESS_KEY: Optional[str] = Field(None, validation_alias="AWS_SECRET_ACCESS_KEY") # REMOVED (unless needed elsewhere)

    # --- Redis Configuration (for Caching - Phase 6) ---
    # Keep Redis if still planned for caching API responses/embeddings
    REDIS_URL: Optional[str] = Field(None, validation_alias="REDIS_URL") # Use str for flexibility, validation later if needed

    # --- Application Settings ---
    LOG_LEVEL: str = Field("INFO", validation_alias="LOG_LEVEL")

    # --- Model/Feature Configuration ---
    RECOMMENDER_ALPHA_WEIGHT: float = Field(0.6, validation_alias="RECOMMENDER_ALPHA_WEIGHT")
    RECOMMENDER_N_RESULTS: int = Field(30, validation_alias="RECOMMENDER_N_RESULTS")
    ANN_N_CANDIDATES: int = Field(500, validation_alias="ANN_N_CANDIDATES")

    # --- Prefect Configuration (Optional) ---
    PREFECT_API_URL: Optional[str] = Field(None, validation_alias="PREFECT_API_URL")
    PREFECT_API_KEY: Optional[str] = Field(None, validation_alias="PREFECT_API_KEY")

    # --- Evidently AI Monitoring (Phase 7) ---
    EVIDENTLY_SERVICE_URL: Optional[str] = Field(None, validation_alias="EVIDENTLY_SERVICE_URL")

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding='utf-8',
        extra='ignore'
    )

# Use lru_cache to load settings only once
@lru_cache()
@lru_cache()
def get_settings() -> Settings:
    settings = Settings()

    # Strip port from SRV URIs if someone left it in .env
    uri = str(settings.MONGODB_URI)
    if uri.startswith("mongodb+srv://") and ":27017" in uri:
        cleaned = uri.replace(":27017", "")
        settings.MONGODB_URI = cleaned
        logger.info(f"Stripped port from MONGODB_URI, new URI: {cleaned}")

    return settings

settings = get_settings()

# Example usage:
# from app.core.config import settings
# print(settings.MONGODB_URI)
