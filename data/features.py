# data/features.py
"""
Feature engineering utilities: compute playlist centroids from stored audio features
and save daily snapshot as Parquet.
"""
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.db.mongo_client import get_mongo_db
from app.core.config import settings

logger = logging.getLogger(__name__)

# Directory for feature snapshots
SNAPSHOT_DIR = Path("data/features/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# List of audio feature fields to use
FEATURE_FIELDS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]


def list_ingested_playlists() -> List[Dict[str, Any]]:
    """
    Fetches all playlists previously ingested from MongoDB.

    Returns:
        A list of dicts with 'id', 'name', and 'track_count'.
    """
    db = get_mongo_db()
    coll = db[settings.MONGO_PLAYLIST_COLLECTION]
    cursor = coll.find({}, {"name": 1, "tracks": 1})
    playlists = []
    for doc in cursor:
        playlists.append({
            "id": doc["_id"],
            "name": doc.get("name"),
            "track_count": len(doc.get("tracks", [])),
        })
    return playlists


def select_playlist_interactively() -> Optional[str]:
    """
    Prompts user to select a playlist from ingested collection.

    Returns:
        Selected playlist ID or None if cancelled.
    """
    playlists = list_ingested_playlists()
    if not playlists:
        print("No ingested playlists found. Run ingestion first.")
        return None
    print("\nSelect a playlist to compute centroid for:")
    for i, p in enumerate(playlists, 1):
        print(f"  {i}. {p['name']} ({p['track_count']} tracks)")
    while True:
        choice = input(f"Enter number (1-{len(playlists)}) or 0 to cancel: ")
        try:
            idx = int(choice)
        except ValueError:
            print("Invalid input; please enter a number.")
            continue
        if idx == 0:
            return None
        if 1 <= idx <= len(playlists):
            return playlists[idx-1]["id"]
        print("Choice out of range; try again.")


def compute_playlist_centroid(playlist_id: str) -> Optional[np.ndarray]:
    """
    Computes the mean vector of audio features for a given playlist.

    Args:
        playlist_id: ID of an ingested playlist.

    Returns:
        1D numpy array of length len(FEATURE_FIELDS), or None if no data.
    """
    db = get_mongo_db()
    # Fetch feature docs for tracks in playlist
    playlist = db[settings.MONGO_PLAYLIST_COLLECTION].find_one({"_id": playlist_id}, {"tracks.track_id": 1})
    if not playlist or "tracks" not in playlist:
        logger.error(f"Playlist {playlist_id} not found in DB.")
        return None
    track_ids = [t["track_id"] for t in playlist["tracks"]]
    if not track_ids:
        logger.warning(f"Playlist {playlist_id} has no tracks.")
        return None
    # Load features into DataFrame
    feature_docs = list(db[settings.MONGO_AUDIO_FEATURES_COLLECTION].find({"_id": {"$in": track_ids}}))
    df = pd.DataFrame(feature_docs)
    if df.empty:
        logger.warning(f"No audio features found for playlist {playlist_id}.")
        return None
    # Ensure only numeric feature columns
    df = df.set_index("_id")[FEATURE_FIELDS]
    # Normalize features (z-score)
    df_norm = (df - df.mean()) / df.std(ddof=0)
    centroid = df_norm.mean(axis=0).values
    logger.info(f"Computed centroid for playlist {playlist_id}.")
    return centroid


def save_centroid(playlist_id: str, centroid: np.ndarray) -> None:
    """
    Saves a centroid vector to a dated Parquet snapshot.

    Args:
        playlist_id: playlist ID
        centroid: 1D numpy array of feature values
    """
    date = datetime.date.today().isoformat()
    path = SNAPSHOT_DIR / f"{date}.parquet"
    record = {"playlist_id": playlist_id, "timestamp": datetime.datetime.now(datetime.timezone.utc)}
    record.update({f: float(centroid[i]) for i, f in enumerate(FEATURE_FIELDS)})
    # Append to existing or create new
    df = pd.DataFrame([record])
    if path.exists():
        df_existing = pd.read_parquet(path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved centroid for {playlist_id} to {path}.")


if __name__ == "__main__":
    # Example interactive run
    pid = select_playlist_interactively()
    if not pid:
        print("No playlist selected; exiting.")
        exit(1)
    cent = compute_playlist_centroid(pid)
    if cent is None:
        print("Failed to compute centroid.")
        exit(1)
    save_centroid(pid, cent)
    print(f"Centroid computed and saved for playlist {pid}.")
