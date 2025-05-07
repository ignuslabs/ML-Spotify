# models/cf.py
"""
Implicit collaborative filtering recommender using ALS.
Requires the `implicit` package: `pip install implicit`
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from data.db.mongo_client import get_mongo_db
from app.core.config import settings

logger = logging.getLogger(__name__)


class ImplicitCFRecommender:
    """
    Collaborative filtering recommender using implicit ALS on playlist-track interactions.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
    ) -> None:
        """
        Initialize the ALS model parameters.
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.model: Optional[AlternatingLeastSquares] = None
        # Mappings between playlist/track IDs and matrix indices
        self.user_map: dict[str, int] = {}
        self.item_map: dict[str, int] = {}
        self.reverse_item_map: dict[int, str] = {}
        self.interaction_matrix: Optional[csr_matrix] = None

    def build_interaction_matrix(self) -> csr_matrix:
        """
        Build the sparse playlist-track interaction matrix from MongoDB.

        Returns:
            csr_matrix of shape (n_playlists, n_tracks) with implicit counts.
        """
        db = get_mongo_db()
        # Load all playlists and track IDs
        playlists = list(db[settings.MONGO_PLAYLIST_COLLECTION].find({}, {"tracks.track_id": 1}))
        # Collect unique track IDs
        track_ids = set()
        for p in playlists:
            for t in p.get("tracks", []):
                track_ids.add(t["track_id"])
        track_ids = sorted(track_ids)
        self.item_map = {tid: idx for idx, tid in enumerate(track_ids)}
        self.reverse_item_map = {idx: tid for tid, idx in self.item_map.items()}

        # Build row indices, col indices, data
        row_inds = []
        col_inds = []
        data_vals = []

        for u_idx, p in enumerate(playlists):
            pid = p["_id"]
            self.user_map[pid] = u_idx
            for t in p.get("tracks", []):
                tid = t["track_id"]
                if tid in self.item_map:
                    row_inds.append(u_idx)
                    col_inds.append(self.item_map[tid])
                    data_vals.append(1.0)  # implicit count = 1 per playlist occurrence

        mat = csr_matrix(
            (np.array(data_vals), (np.array(row_inds), np.array(col_inds))),
            shape=(len(playlists), len(track_ids)),
        )
        # Apply confidence weight (alpha)
        self.interaction_matrix = mat * self.alpha
        logger.info(f"Built interaction matrix: {mat.shape}")
        return self.interaction_matrix

    def train(self, interaction_matrix: Optional[csr_matrix] = None) -> None:
        """
        Train the ALS model on the interaction matrix.

        Args:
            interaction_matrix: Optional pre-built matrix; if None, calls build_interaction_matrix().
        """
        if interaction_matrix is None:
            interaction_matrix = self.build_interaction_matrix()
        if interaction_matrix is None:
            logger.error("No interaction matrix available for training.")
            return

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
        )
        logger.info("Training ALS model...")
        # implicit expects item-user matrix
        self.model.fit(interaction_matrix.T)
        logger.info("ALS model training complete.")

    def recommend(self, playlist_id: str, n: int = 500) -> List[Tuple[str, float]]:
        """
        Recommend top-n track IDs for a given playlist.

        Args:
            playlist_id: Playlist ID to make recommendations for.
            n: Number of recommendations to return.

        Returns:
            List of (track_id, score) tuples.
        """
        if self.model is None or self.interaction_matrix is None:
            logger.error("Model has not been trained yet.")
            return []
        user_idx = self.user_map.get(playlist_id)
        if user_idx is None:
            logger.error(f"Playlist {playlist_id} not found in training data.")
            return []

        # Get recommendation from implicit; returns list of (item_idx, score)
        recs = self.model.recommend(
            user_idx,
            self.interaction_matrix[user_idx],
            N=n,
        )
        # Map back to track IDs
        return [(self.reverse_item_map[idx], float(score)) for idx, score in recs]
