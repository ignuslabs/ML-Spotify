import click
import logging
from data.db.mongo_client import get_mongo_db, close_mongo_connection
from models.cf import ImplicitCFRecommender
from models.ann import build_item_embeddings, build_faiss_index, save_faiss_index

logger = logging.getLogger(__name__)

@click.command()
@click.option("--nlist", default=1000, help="Number of IVF lists for Faiss index.")
@click.option("--hnsw-m", default=32, help="HNSW parameter for quantizer.")
@click.option("--normalize-audio", is_flag=True, default=False, help="Whether to z-score audio features before combining.")
def main(nlist: int, hnsw_m: int, normalize_audio: bool):
    # 0) Setup MongoDB connection
    db = get_mongo_db()

    try:
        # 1) Build and train CF recommender to populate item_map
        cf = ImplicitCFRecommender()
        cf.train()

        # Sanity-check
        assert cf.model is not None, "CF model failed to train"
        assert hasattr(cf, "item_map") and cf.item_map, "No item_map after training"

        # 2) Generate combined embeddings and track IDs
        embeddings, track_ids = build_item_embeddings(cf, normalize_audio)
        logger.info(f"Building Faiss index on {len(track_ids)} items...")

        # 3) Build and save Faiss index
        index = build_faiss_index(embeddings, nlist, hnsw_m)
        save_faiss_index(index, "faiss.index")

        logger.info("Faiss index built and saved successfully.")

    except Exception:
        logger.exception("Error building ANN index:")
        raise

    finally:
        close_mongo_connection()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
