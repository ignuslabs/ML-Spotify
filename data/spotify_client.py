# data/spotify_client.py
"""
SpotifyAPIClient: wrapper around Spotipy for playlist and feature retrieval with
pagination, batching, and basic error handling.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import spotipy
from spotipy.exceptions import SpotifyException

from data.spotify_auth import get_spotify_client

logger = logging.getLogger(__name__)

# Spotify API limits
MAX_TRACKS_PER_REQUEST = 100
MAX_AUDIO_FEATURES_PER_REQUEST = 100
MAX_PLAYLISTS_PER_REQUEST = 50


class SpotifyAPIClient:
    """
    A client wrapper for Spotipy that handles pagination, batching,
    and rate-limit errors for playlist and audio feature endpoints.
    """

    def __init__(self, sp_client: Optional[spotipy.Spotify] = None) -> None:
        """
        Initialize the SpotifyAPIClient.

        Args:
            sp_client: Optional pre-authenticated Spotipy client.
                       If None, will authenticate via get_spotify_client().
        """
        self.sp: spotipy.Spotify = sp_client or get_spotify_client()
        logger.info("SpotifyAPIClient initialized.")

    def get_current_user_playlists(self) -> List[Dict[str, Any]]:
        """
        Fetch all playlists for the authenticated user with pagination.

        Returns:
            List of playlist metadata dicts.
        """
        playlists: List[Dict[str, Any]] = []
        offset = 0
        try:
            while True:
                logger.debug(f"Fetching playlists, offset={offset}")
                response = self.sp.current_user_playlists(
                    limit=MAX_PLAYLISTS_PER_REQUEST, offset=offset
                )
                items = response.get("items", [])
                if not items:
                    break
                playlists.extend(items)
                if response.get("next"):
                    offset += MAX_PLAYLISTS_PER_REQUEST
                    time.sleep(0.1)
                else:
                    break
            logger.info(f"Fetched {len(playlists)} user playlists.")
        except SpotifyException as e:
            logger.error(f"Spotify API error fetching user playlists: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching user playlists: {e}")
        return playlists

    def get_playlist_info(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch basic info for a single playlist.

        Args:
            playlist_id: Spotify playlist ID.

        Returns:
            Playlist metadata dict or None on error.
        """
        try:
            fields = "id,name,description,owner.id,snapshot_id"
            playlist = self.sp.playlist(playlist_id, fields=fields)
            logger.info(f"Fetched playlist info: {playlist.get('name')}")
            return playlist
        except SpotifyException as e:
            logger.error(f"Spotify API error fetching playlist {playlist_id}: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching playlist {playlist_id}: {e}")
        return None

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all tracks for a given playlist with pagination.

        Args:
            playlist_id: Spotify playlist ID.

        Returns:
            List of playlist item dicts (with track data).
        """
        tracks: List[Dict[str, Any]] = []
        offset = 0
        try:
            while True:
                logger.debug(f"Fetching tracks for playlist {playlist_id}, offset={offset}")
                response = self.sp.playlist_items(
                    playlist_id,
                    limit=MAX_TRACKS_PER_REQUEST,
                    offset=offset,
                    fields="items(track(id,name,artists,album,duration_ms,explicit,external_urls)),next",
                )
                items = response.get("items", [])
                if not items:
                    break
                tracks.extend(items)
                if response.get("next"):
                    offset += MAX_TRACKS_PER_REQUEST
                    time.sleep(0.1)
                else:
                    break
            logger.info(f"Fetched {len(tracks)} tracks for playlist {playlist_id}.")
        except SpotifyException as e:
            logger.error(f"Spotify API error fetching tracks: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching tracks: {e}")
        return tracks

    def get_audio_features(
        self, track_ids: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Fetch audio features for a batch of track IDs.

        Args:
            track_ids: List of Spotify track IDs.

        Returns:
            List of audio feature dicts or None for each track.
        """
        if not track_ids:
            return []

        all_features: List[Optional[Dict[str, Any]]] = []
        try:
            for i in range(0, len(track_ids), MAX_AUDIO_FEATURES_PER_REQUEST):
                batch = track_ids[i : i + MAX_AUDIO_FEATURES_PER_REQUEST]
                logger.debug(f"Fetching audio features for batch {i // MAX_AUDIO_FEATURES_PER_REQUEST + 1}")
                features = self.sp.audio_features(tracks=batch)
                all_features.extend(features or [None] * len(batch))
                time.sleep(0.1)
            logger.info(f"Fetched audio features for {len(track_ids)} tracks.")
        except SpotifyException as e:
            logger.error(f"Spotify API error fetching audio features: {e}")
            all_features.extend([None] * len(track_ids))
        except Exception as e:
            logger.exception(f"Unexpected error fetching audio features: {e}")
            all_features.extend([None] * len(track_ids))
        return all_features