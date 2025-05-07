# data/ingestion.py
"""
Core ingestion logic: parsing Spotify data and upserting into MongoDB.
"""
import logging
import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pymongo import UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError

from data.spotify_client import SpotifyAPIClient
from data.db.mongo_client import get_mongo_db
from app.core.config import settings

logger = logging.getLogger(__name__)


def parse_track_data(track_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts relevant fields from a Spotify playlist item for storage.

    Args:
        track_item: raw item from Spotify API's playlist_items.

    Returns:
        Dict with track data (including '_id') or None if invalid.
    """
    if not track_item or 'track' not in track_item or not track_item['track']:
        return None

    track = track_item['track']
    track_id = track.get('id')
    if not track_id:
        logger.warning("Skipping track with missing ID")
        return None

    artists = track.get('artists', [])
    album = track.get('album', {})
    external = track.get('external_urls', {})

    doc: Dict[str, Any] = {
        '_id': track_id,
        'name': track.get('name'),
        'artist_ids': [a['id'] for a in artists if a.get('id')],
        'artist_names': [a.get('name') for a in artists if a.get('name')],
        'album_id': album.get('id'),
        'album_name': album.get('name'),
        'popularity': track.get('popularity'),
        'duration_ms': track.get('duration_ms'),
        'explicit': track.get('explicit'),
        'external_url_spotify': external.get('spotify'),
        'last_updated_at': datetime.datetime.now(datetime.timezone.utc),
    }
    return doc


def parse_audio_features_data(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:

    #Transform Spotify audio-features JSON into a dict for Mongo upsert,
    #using the Spotify track ID as the Mongo _id.
    if not raw or not raw.get("id"):
        return None

    # Use the Spotify track ID as Mongo’s _id
    track_id = raw["id"]
    return {
        "_id": track_id,
        "danceability": raw.get("danceability"),
        "energy": raw.get("energy"),
        "key": raw.get("key"),
        "loudness": raw.get("loudness"),
        "mode": raw.get("mode"),
        "speechiness": raw.get("speechiness"),
        "acousticness": raw.get("acousticness"),
        "instrumentalness": raw.get("instrumentalness"),
        "liveness": raw.get("liveness"),
        "valence": raw.get("valence"),
        "tempo": raw.get("tempo"),
        "time_signature": raw.get("time_signature"),
        "last_updated_at": datetime.datetime.now(datetime.timezone.utc),
    }


def upsert_documents(
    db: Database, collection_name: str, documents: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """
    Bulk upserts documents into specified MongoDB collection.

    Args:
        db: MongoDB Database instance.
        collection_name: target collection name.
        documents: list of dicts with '_id' keys.

    Returns:
        Tuple of (matched_count, modified_or_upserted_count).
    """
    if not documents:
        return 0, 0

    coll = db[collection_name]
    ops = []
    for doc in documents:
        _id = doc.get('_id')
        if not _id:
            logger.warning("Document missing '_id', skipping upsert")
            continue
        ops.append(UpdateOne({'_id': _id}, {'$set': doc}, upsert=True))

    if not ops:
        return 0, 0

    try:
        result = coll.bulk_write(ops, ordered=False)
        matched = result.matched_count
        modified = result.modified_count + result.upserted_count
        logger.info(f"Upserted {len(ops)} docs into {collection_name} (matched={matched}, affected={modified})")
        return matched, modified
    except BulkWriteError as bwe:
        details = bwe.details.get('writeErrors', [])
        logger.error(f"Bulk write error on {collection_name}: {details}")
        res = bwe.details.get('result', {})
        matched = res.get('nMatched', 0)
        modified = res.get('nModified', 0) + res.get('nUpserted', 0)
        return matched, modified
    except Exception as e:
        logger.exception(f"Unexpected error during bulk upsert: {e}")
        return 0, 0


def ingest_playlist_data(playlist_id: str, spotify_client: SpotifyAPIClient) -> bool:
    """
    Main orchestration for ingesting a playlist into MongoDB.

    Args:
        playlist_id: Spotify playlist ID.
        spotify_client: authenticated SpotifyAPIClient.

    Returns:
        True if ingestion succeeds, False otherwise.
    """
    logger.info(f"Beginning ingestion for playlist: {playlist_id}")
    try:
        # Fetch playlist metadata
        meta = spotify_client.get_playlist_info(playlist_id)
        if not meta:
            logger.error("Failed to fetch playlist metadata")
            return False

        # Fetch tracks
        items = spotify_client.get_playlist_tracks(playlist_id)
        track_docs: List[Dict[str, Any]] = []
        feature_ids: List[str] = []
        playlist_order: List[Dict[str, Any]] = []

        for idx, item in enumerate(items):
            tdoc = parse_track_data(item)
            if tdoc:
                track_docs.append(tdoc)
                tid = tdoc['_id']
                feature_ids.append(tid)
                added = item.get('added_at')
                ts = pd.to_datetime(added, utc=True).to_pydatetime() if added else None
                playlist_order.append({'track_id': tid, 'position': idx, 'added_at': ts})

        # Fetch features
        features = spotify_client.get_audio_features(feature_ids)
        feature_docs = [parse_audio_features_data(f) for f in features if parse_audio_features_data(f)]

        # Upsert
        db = get_mongo_db()
        upsert_documents(db, settings.MONGO_TRACK_COLLECTION, track_docs)
        upsert_documents(db, settings.MONGO_AUDIO_FEATURES_COLLECTION, feature_docs)

        # Upsert playlist document
        playlist_doc = {
            '_id': meta['id'],
            'name': meta.get('name'),
            'description': meta.get('description'),
            'owner_id': meta.get('owner', {}).get('id'),
            'snapshot_id': meta.get('snapshot_id'),
            'tracks': playlist_order,
            'track_count': len(playlist_order),
            'last_fetched_at': datetime.datetime.now(datetime.timezone.utc),
        }
        res = db[settings.MONGO_PLAYLIST_COLLECTION].update_one(
            {'_id': playlist_id}, {'$set': playlist_doc}, upsert=True
        )
        logger.info(f"Upserted playlist doc (matched={res.matched_count}, modified={res.modified_count})")
        return True

    except Exception as exc:
        logger.exception(f"Critical error during ingestion: {exc}")
        return False