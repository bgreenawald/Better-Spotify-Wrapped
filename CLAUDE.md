# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Better Spotify Wrapped is a Python web application that analyzes Spotify listening history to create enhanced visualizations beyond standard Spotify Wrapped. It combines personal data exports with the Spotify Web API for rich analytics.

## Key Commands

### Running the Application
```bash
# Start the dashboard (default: http://127.0.0.1:8050)
python dashboard/application.py

# Generate wrapped reports
python src/wrapped.py
```

### Development Commands
```bash
# Run all tests
pytest

# Run tests with detailed output
pytest -ra

# Install/sync dependencies (using uv package manager)
uv sync

# Add new dependency
uv add <package>
```

## Architecture Overview

The codebase follows a three-layer architecture:

1. **Data Layer** (`src/io.py`, `src/api/`)
   - Loads Spotify JSON exports from `data/{username}/listening_history/`
   - Enriches data via Spotify Web API with caching
   - API credentials stored in `.env` file

2. **Processing Layer** (`src/preprocessing.py`, `src/metrics/`)
   - `preprocessing.py`: Cleans raw data, adds derived fields (year, month, day)
   - `metrics/metrics.py`: Calculates top tracks, artists, listening stats
   - `metrics/trends.py`: Time-series analysis and trend detection

3. **Presentation Layer** (`dashboard/`)
   - `application.py`: Main Dash app initialization and server
   - `callbacks/callbacks.py`: Interactive callback logic for UI updates
   - `components/`: Reusable UI components (filters, graphs, stats)
   - `layouts/layouts.py`: Page structure and tab definitions

## Key Dependencies

- **Dash + Plotly**: Interactive web dashboard and visualizations
- **Pandas**: Data processing and analysis
- **Spotipy**: Spotify Web API client
- **python-dotenv**: Environment variable management

## Environment Setup

Required `.env` file:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
DATA_DIR=path/to/data
```

## Data Structure

User data expected in: `data/{username}/listening_history/`
- JSON files from Spotify personal data export
- API cache stored in `data/api_cache/`

## Testing Approach

- Integration tests for Dash application components
- Unit tests for metrics calculations
- Test fixtures in `tests/` directory
- Use `pytest` for all testing needs

## Important Patterns

1. **Callback Pattern**: All interactivity in `dashboard/callbacks/callbacks.py` uses Dash callback decorators
2. **Data Caching**: API responses cached to minimize rate limiting
3. **Modular Components**: UI components in `dashboard/components/` are self-contained and reusable
4. **Time Filtering**: Most metrics support date range filtering via pandas datetime operations