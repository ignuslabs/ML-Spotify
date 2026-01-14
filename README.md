# ML-Spotify (Playlist Recommender)

## Project Overview
ML-Spotify is an API-first recommendation service that takes a Spotify playlist ID
and returns 10–30 track IDs that blend seamlessly with the playlist. The project
combines content-based similarity, collaborative filtering, approximate nearest
neighbor (ANN) search, and learning-to-rank workflows to stay under a 300ms p95
latency budget.

## Quickstart
1. Clone the repo:
   ```bash
   git clone https://github.com/ignuslabs/ML-Spotify.git # or your fork
   cd ML-Spotify
   ```
2. Install dependencies:
   ```bash
   poetry install --with dev
   ```
3. Copy `.env.example` to `.env` and add your Spotify + MongoDB credentials.
4. Ingest a playlist:
   ```bash
   poetry run python data/scripts/ingest.py
   ```

## Common Workflows
### Ingest Spotify data
```bash
poetry run python data/scripts/ingest.py --setup-db
poetry run python data/scripts/ingest.py <playlist_id>
```

### Build the ANN index
```bash
poetry run python data/scripts/build_ann.py --nlist 1000 --hnsw-m 32
```

### Compute playlist centroids
```bash
poetry run python data/features.py
```

## Development Commands
```bash
poetry run pre-commit run --all-files
poetry run ruff check .
poetry run black --check .
poetry run mypy .
poetry run pytest --cov=app --cov=data --cov=models --cov-report=term-missing
```

## Repository Layout
```
app/                 # Core settings
data/                # Spotify ingestion, feature engineering, MongoDB access
models/              # Recommendation and ranking models
.github/workflows/   # CI workflows
pyproject.toml       # Dependencies and tooling
```
