#!/usr/bin/env python3
# data/scripts/features.py
"""
CLI for feature engineering: compute and save playlist centroids.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.core.config import settings
from data.features import (
    select_playlist_interactively,
    compute_playlist_centroid,
    save_centroid,
)
from data.db.mongo_client import get_mongo_client, close_mongo_connection

# Configure root logger
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature engineering CLI: compute playlist centroids"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compute-centroid subcommand
    compute_parser = subparsers.add_parser(
        "compute-centroid", help="Compute and save centroid for a playlist"
    )
    compute_parser.add_argument(
        "--playlist-id", type=str, help="Ingested playlist ID (falls back to interactive selection)"
    )

    args = parser.parse_args()

    # Verify MongoDB connection
    try:
        get_mongo_client().admin.command("ismaster")
        logger.info("MongoDB connection successful.")
    except Exception:
        logger.exception("Failed to connect to MongoDB.")
        print("Error: Cannot connect to MongoDB. Check your MONGODB_URI.")
        sys.exit(1)

    if args.command == "compute-centroid":
        playlist_id = args.playlist_id or select_playlist_interactively()
        if not playlist_id:
            print("No playlist selected; exiting.")
            sys.exit(1)
        centroid = compute_playlist_centroid(playlist_id)
        if centroid is None:
            print(f"Failed to compute centroid for playlist {playlist_id}.")
            sys.exit(1)
        save_centroid(playlist_id, centroid)
        print(f"Centroid computed and saved for playlist {playlist_id}.")


if __name__ == "__main__":
    try:
        main()
    finally:
        close_mongo_connection()
