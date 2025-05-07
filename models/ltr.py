# models/ltr.py
"""
Learning-to-Rank (LTR) re-ranker using LightGBM.
"""
import logging
from typing import Dict, List, Tuple

import lightgbm as lgb
import pandas as pd
from data.db.mongo_client import get_mongo_db
from models.cf import ImplicitCFRecommender
from app.core.config import settings

logger = logging.getLogger(__name__)


def build_ltr_dataset(
    playlist_ids: List[str],
    top_k: int = 500
) -> pd.DataFrame:
    """
    Build a dataset for learning-to-rank from given playlists and their true next-track labels.

    Args:
        playlist_ids: List of playlist IDs to include.
        top_k: Number of ANN candidates to fetch per playlist.

    Returns:
        DataFrame with columns:
            - playlist_id: str
            - track_id: str
            - label: int (1 if next-track, else 0)
            - cf_score: float
    """
    # Train CF model
    cf = ImplicitCFRecommender()
    cf.train()

    # Load playlist collection
    db = get_mongo_db()
    coll = db[settings.MONGO_PLAYLIST_COLLECTION]

    rows: List[Dict] = []
    for pid in playlist_ids:
        playlist = coll.find_one({'_id': pid})
        if not playlist or not playlist.get('tracks'):
            continue
        # The last track in the 'tracks' array is our positive label
        last_entry = playlist['tracks'][-1]
        next_tid = last_entry['track_id'] if isinstance(last_entry, dict) else last_entry

        # Get CF recommendations (track_id, score)
        candidates = cf.recommend(pid, n=top_k)
        for tid, score in candidates:
            rows.append({
                'playlist_id': pid,
                'track_id': tid,
                'label': 1 if tid == next_tid else 0,
                'cf_score': score,
            })

    return pd.DataFrame(rows)


def train_ranker(
    df: pd.DataFrame,
    params: Dict = None,
    model_path: str = "models/ltr.model"
) -> lgb.Booster:
    """
    Train a LightGBM ranker on the provided dataset.

    Args:
        df: DataFrame from build_ltr_dataset including feature columns and label.
        params: LightGBM training parameters.
        model_path: Path to save the trained model.

    Returns:
        Trained LightGBM Booster model.
    """
    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            # ... other default params
        }

    # Group data by playlist_id
    df_sorted = df.sort_values(['playlist_id'])
    groups = df_sorted.groupby('playlist_id').size().to_list()

    # Features excluding label and ids
    feature_cols = [c for c in df.columns if c not in ('playlist_id', 'track_id', 'label')]
    dtrain = lgb.Dataset(
        df_sorted[feature_cols],
        label=df_sorted['label'],
        group=groups
    )
    logger.info("Training LightGBM ranker...")
    bst = lgb.train(params, dtrain)
    bst.save_model(model_path)
    logger.info(f"LTR model saved to {model_path}")
    return bst


def predict_ranker(
    model: lgb.Booster,
    df_candidates: pd.DataFrame
) -> List[Tuple[str, float]]:
    """
    Predict scores for candidate tracks and return ranked list.

    Args:
        model: Trained LightGBM Booster.
        df_candidates: DataFrame with feature columns for a single playlist,
                       must include 'track_id'.

    Returns:
        List of (track_id, score) tuples sorted by descending score.
    """
    feature_cols = [c for c in df_candidates.columns if c not in ('playlist_id', 'track_id', 'label')]
    scores = model.predict(df_candidates[feature_cols])
    df_candidates = df_candidates.copy()
    df_candidates['score'] = scores
    results = df_candidates[['track_id', 'score']].sort_values('score', ascending=False)
    ranked = list(zip(results['track_id'], results['score']))
    logger.info(f"Predicted and ranked {len(ranked)} candidates.")
    return ranked
