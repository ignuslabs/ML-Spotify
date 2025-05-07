# data/spotify_auth.py
"""
Handles Spotify OAuth and provides an authenticated Spotipy client.
"""
import logging
import webbrowser
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from app.core.config import settings

logger = logging.getLogger(__name__)

# Required scopes for playlist and audio features access
SCOPES = "playlist-read-private playlist-read-collaborative user-library-read"

# Cache directory for storing Spotify token
CACHE_DIR = settings.ROOT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = str(CACHE_DIR / ".spotify_token_cache")


def get_spotify_oauth() -> SpotifyOAuth:
    """
    Returns a configured SpotifyOAuth object for PKCE flow.
    """
    return SpotifyOAuth(
        client_id=settings.SPOTIPY_CLIENT_ID,
        client_secret=settings.SPOTIPY_CLIENT_SECRET,
        redirect_uri=str(settings.SPOTIPY_REDIRECT_URI),
        scope=SCOPES,
        cache_path=CACHE_PATH,
        show_dialog=True,
        open_browser=False,
    )


def get_spotify_client() -> spotipy.Spotify:
    """
    Performs OAuth flow (interactive if needed) and returns an authenticated Spotipy client.

    Raises:
        Exception: if authentication or token refresh fails.
    """
    sp_oauth = get_spotify_oauth()
    token_info = sp_oauth.get_cached_token()

    if not token_info:
        # Start interactive authorization
        auth_url = sp_oauth.get_authorize_url()
        print(f"\n--- Spotify Authentication Required ---")
        print(f"Navigate here to authorize: {auth_url}")
        try:
            webbrowser.open(auth_url)
        except webbrowser.Error:
            logger.warning("Unable to open browser automatically.")
        response_url = input("Paste the redirected URL here: ").strip()
        if not response_url:
            logger.error("No redirect URL provided; cancelling authentication.")
            raise Exception("Spotify authentication cancelled by user.")
        code = sp_oauth.parse_response_code(response_url)
        token_info = sp_oauth.get_access_token(code, as_dict=True)
        logger.info("Obtained new Spotify access token.")

    # Refresh token if expired
    if sp_oauth.is_token_expired(token_info):
        logger.info("Refreshing expired Spotify token.")
        token_info = sp_oauth.refresh_access_token(token_info.get("refresh_token"))

    access_token = token_info.get("access_token")
    if not access_token:
        logger.error("No access token available after authentication.")
        raise Exception("Failed to retrieve Spotify access token.")

    sp = spotipy.Spotify(auth=access_token)
    logger.info("Authenticated Spotipy client created successfully.")
    return sp
