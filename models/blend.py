# models/blend.py
"""
Blend content-based and collaborative filtering scores via a weighted sum.
"""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _min_max_scale(values: List[float]) -> List[float]:
    """
    Min-max normalize a list of floats to range [0, 1].

    Args:
        values: list of numeric scores.
    Returns:
        list of normalized scores.
    """
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val + 1e-12
    return [(v - min_val) / range_val for v in values]


def blend_scores(
    content_scores: Dict[str, float],
    cf_scores: Dict[str, float],
    alpha: float,
) -> List[Tuple[str, float]]:
    """
    Blend two score dictionaries by normalizing and weighted sum.

    Args:
        content_scores: Mapping of track_id to content-based score.
        cf_scores: Mapping of track_id to collaborative filtering score.
        alpha: Weight for content-based scores (0 <= alpha <= 1).

    Returns:
        Sorted list of (track_id, blended_score) in descending order.
    """
    # Union of all track IDs
    track_ids = set(content_scores) | set(cf_scores)
    ids = list(track_ids)

    # Gather raw score arrays
    content_arr = [content_scores.get(t, 0.0) for t in ids]
    cf_arr = [cf_scores.get(t, 0.0) for t in ids]

    # Normalize to [0,1]
    content_norm = _min_max_scale(content_arr)
    cf_norm = _min_max_scale(cf_arr)

    # Blend scores
    blended = []
    for tid, c, f in zip(ids, content_norm, cf_norm):
        score = alpha * c + (1 - alpha) * f
        blended.append((tid, score))

    # Sort by score descending
    blended.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Blended {len(blended)} scores with alpha={alpha}")
    return blended
