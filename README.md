# Playlist‑Recommender

## Project Overview
Build an API‑first service that takes a user’s Spotify playlist ID and returns 10–30 recommended track IDs that blend seamlessly with the playlist. Designed for low latency (< 300 ms p95) at scale.

## Getting Started
1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/playlist-recommender.git
   cd playlist-recommender
2. Install Dependencies
    poetry install --with dev
3. Copy .env.example to .env and fill in your credentials.


playlist-recommender/
├── app/
│   └── core/
│       ├── __init__.py
│       └── config.py
├── data/
├── models/
├── infra/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md