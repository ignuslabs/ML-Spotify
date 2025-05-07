# models/content_based.py
"""
Content-based recommendation: compute track similarities to a playlist centroid.
"""
import logging
from typing import List

import numpy as np
import pandas as pd

from data.features import compute_playlist_centroid
from data.db.mongo_client import get_mongo_db
from app.core.config import settings

logger = logging.getLogger(__name__)


def compute_cosine_similarity(centroid: np.ndarray, feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single centroid vector and each row in a feature matrix.

    Args:
        centroid: 1D numpy array of shape (d,).
        feature_matrix: 2D numpy array of shape (n, d).

    Returns:
        similarities: 1D numpy array of shape (n,) with cosine similarity scores.
    """
    # handle zero vectors safely by adding small epsilon
    centroid_norm = np.linalg.norm(centroid) + 1e-12
    feats_norm = np.linalg.norm(feature_matrix, axis=1) + 1e-12
    dot_prods = feature_matrix.dot(centroid)
    similarities = dot_prods / (feats_norm * centroid_norm)
    return similarities


def get_top_n_by_content(playlist_id: str, n: int = 500) -> List[str]:
    """
    Retrieve top-n track IDs by content-based cosine similarity to the playlist centroid.

    Args:
        playlist_id: ID of the playlist to base recommendations on.
        n: number of candidate track IDs to return.

    Returns:
        List of recommended track IDs (excluding those already in playlist).
    """
    logger.info(f"Computing content-based candidates for playlist {playlist_id}")
    # Compute playlist centroid
    centroid = compute_playlist_centroid(playlist_id)
    if centroid is None:
        logger.error("Failed to compute centroid; returning empty list.")
        return []

    # Load all audio feature documents
    db = get_mongo_db()
    coll = db[settings.MONGO_AUDIO_FEATURES_COLLECTION]
    feature_docs = list(coll.find({}))
    if not feature_docs:
        logger.warning("No audio features found in DB.")
        return []

    # Build DataFrame
    df = pd.DataFrame(feature_docs)
    # Ensure feature fields align with compute_playlist_centroid
    feature_fields = [
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
    ]
    feature_matrix = df[feature_fields].to_numpy()
    track_ids = df["_id"].tolist()

    # Compute similarities
    similarities = compute_cosine_similarity(centroid, feature_matrix)

    # Build DataFrame of results
    results = pd.DataFrame({"track_id": track_ids, "score": similarities})

    # Exclude tracks in the original playlist
    playlist_doc = db[settings.MONGO_PLAYLIST_COLLECTION].find_one({"_id": playlist_id}, {"tracks.track_id": 1})
    existing_ids = {t["track_id"] for t in playlist_doc.get("tracks", [])} if playlist_doc else set()
    candidates = results[~results["track_id"].isin(existing_ids)]

    # Sort descending by score, take top n
    top = candidates.nlargest(n, "score")
    recommended_ids = top["track_id"].tolist()
    logger.info(f"Selected {len(recommended_ids)} content-based candidates for playlist {playlist_id}")
    return recommended_ids
