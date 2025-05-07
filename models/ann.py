# models/ann.py
"""
ANN candidate generation using Faiss (IVF + HNSW).
"""
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pandas as pd

from data.db.mongo_client import get_mongo_db
from app.core.config import settings
from data.features import FEATURE_FIELDS
from models.cf import ImplicitCFRecommender

logger = logging.getLogger(__name__)


def build_item_embeddings(
    cf_recommender: ImplicitCFRecommender,
    normalize_audio: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Build combined embeddings for all tracks by concatenating
    normalized audio features and CF latent factors.

    Args:
        cf_recommender: A trained ImplicitCFRecommender instance.
        normalize_audio: Whether to z-score normalize audio features.

    Returns:
        embeddings: np.ndarray of shape (num_tracks, D)
        track_ids: List of track IDs in the same order as embeddings.
    """
    # Ensure the CF model and item_map are built
    if cf_recommender.model is None or not cf_recommender.item_map:
        raise ValueError("CF model not trained or item_map not built.")

    # Determine track order from CF recommender
    num_items = len(cf_recommender.reverse_item_map)
    track_ids = [cf_recommender.reverse_item_map[i] for i in range(num_items)]

    # Load audio features from MongoDB
    db = get_mongo_db()
    coll = db[settings.MONGO_AUDIO_FEATURES_COLLECTION]

    # Auto-detect the ID field in our feature documents
    # Sample a full document to see if the 'id' key holds the track ID
    sample = coll.find_one()
    if sample and "id" in sample and sample["id"] in track_ids:
        id_field = "id"
    else:
        id_field = "_id"

    # Fetch feature docs using the detected ID field
    docs = list(coll.find(
        { id_field: { "$in": track_ids } },
        { id_field: 1, **{ f: 1 for f in FEATURE_FIELDS } }
    ))

    # Build audio_matrix, with fallback to zeros
    if not docs:
        logger.warning("No audio feature docs found; using zeros matrix")
        audio_matrix = np.zeros((len(track_ids), len(FEATURE_FIELDS)), dtype=np.float32)
    else:
        df = pd.DataFrame(docs)
        if id_field != "_id":
            df = df.rename(columns={id_field: "_id"})
        df = df.set_index("_id").reindex(track_ids)
        audio_matrix = df[FEATURE_FIELDS].to_numpy(dtype=np.float32)
        if normalize_audio:
            means = np.nanmean(audio_matrix, axis=0, keepdims=True)
            stds = np.nanstd(audio_matrix, axis=0, keepdims=True) + 1e-12
            audio_matrix = (audio_matrix - means) / stds

    cf_indices = [cf_recommender.item_map[tid] for tid in track_ids]
    cf_matrix = cf_recommender.model.item_factors[cf_indices].astype(np.float32)

    # Concatenate audio + CF factors
    embeddings = np.hstack([audio_matrix, cf_matrix])
    logger.info(f"Built item embeddings with shape {embeddings.shape}")
    return embeddings, track_ids


def build_faiss_index(
    embeddings: np.ndarray,
    nlist: int = 10000,
    hnsw_m: int = 32,
) -> faiss.Index:
    """
    Constructs an IVF+HNSW Faiss index over the embeddings.

    Args:
        embeddings: np.ndarray of shape (N, D) (dtype float32).
        nlist: number of Voronoi cells for IVF.
        hnsw_m: connectivity parameter for HNSW quantizer.

    Returns:
        A trained Faiss Index instance.
    """
    N, D = embeddings.shape
    # HNSW quantizer
    quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
    # IVF index with flat vectors
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)

    logger.info(f"Training Faiss index: {N} vectors, dimension={D}, nlist={nlist}, hnsw_m={hnsw_m}")
    index.train(embeddings)
    index.add(embeddings)
    logger.info("Faiss index built and populated.")
    return index


def save_faiss_index(index: faiss.Index, path: str) -> None:
    """
    Saves a Faiss index to disk.

    Args:
        index: Faiss Index object.
        path: File path to write the index to.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    logger.info(f"Faiss index saved to {out}")
