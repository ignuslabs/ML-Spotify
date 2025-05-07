#!/usr/bin/env python3
# data/scripts/ingest.py
"""
CLI for ingesting Spotify playlist data into MongoDB.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.core.config import settings
from data.spotify_client import SpotifyAPIClient
from data.db.mongo_client import get_mongo_client, get_mongo_db, close_mongo_connection
from data.ingestion import ingest_playlist_data

# Configure root logger
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def setup_database() -> None:
    """
    Ensure MongoDB collections have necessary indexes.
    """
    db = get_mongo_db()
    # Text index for track names
    db[settings.MONGO_TRACK_COLLECTION].create_index(
        [("name", "text")], name="track_name_text_idx", background=True
    )
    # Index on artist IDs
    db[settings.MONGO_TRACK_COLLECTION].create_index(
        [("artist_ids", 1)], name="artist_ids_idx", background=True
    )
    # Index on playlist owner
    db[settings.MONGO_PLAYLIST_COLLECTION].create_index(
        [("owner_id", 1)], name="owner_id_idx", background=True
    )
    logger.info("Database indexes ensured.")


def select_playlist_interactively(client: SpotifyAPIClient) -> str | None:
    """
    Prompt user to select a playlist from their Spotify account.
    """
    playlists = client.get_current_user_playlists()
    if not playlists:
        print("No playlists found for the authenticated user.")
        return None

    print("\nYour Spotify Playlists:")
    for idx, p in enumerate(playlists, start=1):
        name = p.get("name", "<unknown>")
        owner = p.get("owner", {}).get("display_name", "<unknown>")
        count = p.get("tracks", {}).get("total", "?")
        print(f"  {idx}. {name} ({count} tracks) by {owner}")

    while True:
        choice = input(f"Enter playlist number (1-{len(playlists)}) or 0 to cancel: ")
        try:
            num = int(choice)
        except ValueError:
            print("Invalid input; please enter a number.")
            continue
        if num == 0:
            logger.info("Playlist selection cancelled.")
            return None
        if 1 <= num <= len(playlists):
            selected = playlists[num - 1]
            print(f"Selected playlist: {selected.get('name')}")
            return selected.get("id")
        print("Choice out of range; try again.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Spotify playlist data into MongoDB."
    )
    parser.add_argument(
        "playlist_id",
        nargs="?",
        help="Spotify playlist ID to ingest (optional).",
    )
    parser.add_argument(
        "--setup-db",
        action="store_true",
        help="Create/verify MongoDB indexes before ingestion.",
    )
    args = parser.parse_args()

    if args.setup_db:
        try:
            setup_database()
            print("Database setup complete.")
            sys.exit(0)
        except Exception:
            logger.exception("Database setup failed.")
            sys.exit(1)

    # Verify DB connection
    try:
        get_mongo_client().admin.command("ismaster")
        logger.info("MongoDB connection successful.")
    except Exception:
        logger.exception("Failed to connect to MongoDB.")
        print("Error: Cannot connect to MongoDB. Check your MONGODB_URI.")
        sys.exit(1)

    # Initialize Spotify client
    try:
        spotify_client = SpotifyAPIClient()
    except Exception as e:
        logger.exception("Spotify client initialization failed.")
        print(f"Error initializing Spotify client: {e}")
        sys.exit(1)

    # Determine playlist ID
    playlist_id = args.playlist_id or select_playlist_interactively(spotify_client)
    if not playlist_id:
        print("No playlist selected; exiting.")
        sys.exit(1)

    print(f"\nStarting ingestion for playlist: {playlist_id}")
    success = ingest_playlist_data(playlist_id, spotify_client)
    if success:
        print("Ingestion completed successfully.")
        sys.exit(0)
    else:
        print("Ingestion failed. Review logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        close_mongo_connection()
# Ensure MongoDB connection is closed
# close_mongo_connection()