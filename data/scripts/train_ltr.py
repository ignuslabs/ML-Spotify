#!/usr/bin/env python3
# data/scripts/train_ltr.py
"""
CLI to build dataset and train the LightGBM Learning-to-Rank model.
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data.db.mongo_client import get_mongo_client, close_mongo_connection
from models.ltr import build_ltr_dataset, train_ranker
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Learning-to-Rank model for playlist recommendations"
    )
    parser.add_argument(
        "--playlists", type=str, default=None,
        help="Optional path to a file containing playlist IDs (one per line)."
    )
    parser.add_argument(
        "--model-path", type=str, default="models/ltr.model",
        help="Path to save the trained LTR model"
    )
    args = parser.parse_args()

    try:
        # Verify MongoDB connection
        get_mongo_client().admin.command("ismaster")
        logger.info("MongoDB connection successful.")

        # Determine playlist IDs
        if args.playlists:
            pfile = Path(args.playlists)
            playlist_ids = [line.strip() for line in pfile.read_text().splitlines() if line.strip()]
        else:
            db = get_mongo_client()
            playlist_ids = [p["_id"] for p in db[settings.MONGO_PLAYLIST_COLLECTION].find({}, {"_id": 1})]
        logger.info(f"Found {len(playlist_ids)} playlists to include in LTR dataset.")

        # Build LTR dataset
        df = build_ltr_dataset(playlist_ids, top_k=settings.ANN_N_CANDIDATES)
        logger.info(f"Built dataset with {len(df)} rows.")

        # Train ranker
        model = train_ranker(df, params=None, model_path=args.model_path)
        logger.info(f"LTR model training complete. Model saved to {args.model_path}")

    except Exception as e:
        logger.exception(f"Failed to train LTR model: {e}")
        sys.exit(1)
    finally:
        close_mongo_connection()


if __name__ == "__main__":
    main()