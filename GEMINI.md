# Gemini Project Context: Better Spotify Wrapped

This document provides context for the "Better Spotify Wrapped" project, a tool for in-depth analysis and visualization of personal Spotify listening data.

## Project Overview

This is a Python-based data analysis and visualization project. It ingests a user's extended Spotify listening history (the JSON data dump requested from Spotify) into a structured database and provides a web-based dashboard to explore the data.

The project has two main parts:
1.  A **CLI (`bsw`)** for data ingestion and processing.
2.  A **web dashboard** for data visualization.

### Key Technologies

-   **Backend:** Python
-   **CLI:** Click
-   **Database:** DuckDB
-   **Web Framework:** Dash (with Dash Bootstrap Components and Dash Mantine Components)
-   **Data Manipulation:** Pandas
-   **Spotify API Interaction:** Spotipy
-   **Linting/Formatting:** Ruff
-   **Testing:** Pytest

### Architecture

-   **Data Ingestion:** The `src/cli.py` script provides commands to ingest Spotify listening history from JSON files into a DuckDB database (`data/db/music.db` by default). It also includes commands to enrich the data with metadata from the Spotify API (via Spotipy), such as ISRCs, album details, and artist genres.
-   **Database:** The database schema is defined in `DDL.sql`. It uses a star-schema-like model with fact tables for plays (`fact_plays`) and dimension tables for tracks, artists, albums, and users. It also includes tables for genre and mood mapping.
-   **Dashboard:** The `dashboard/` directory contains a Dash application. `application.py` is the main entry point. It queries the DuckDB database to build visualizations. The dashboard is structured with layouts, components, and callbacks.

## Building and Running

### 1. Installation

The project uses `uv` (or `pip`) and a `pyproject.toml` file for dependency management.

To install the project and its development dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Database Setup & Data Ingestion

The core of the application is the DuckDB database. You must first ingest your Spotify listening history.

1.  **Request your data from Spotify:** You need to request your "extended streaming history" from your Spotify account privacy settings.
2.  **Run the ingestion CLI:** Use the `bsw ingest-history` command.

Example:
```bash
# The bsw command is available after installation
bsw ingest-history \
    --user-id your_user_id \
    --db data/db/music.db \
    --history-dir /path/to/your/spotify_data/ \
    --apply-ddl
```
This command will create the database file, apply the schema from `DDL.sql`, and load your listening history.

You can then run other `ingest-*` commands to enrich the data. Use `bsw --help` to see all available commands.

### 3. Running the Web App

Once the database is populated, you can run the web dashboard.

```bash
python dashboard/application.py
```
The application will be available at `http://127.0.0.1:8050/`.

### 4. Running Tests

The project uses `pytest` for testing.

```bash
pytest
```

## Development Conventions

-   **Code Style:** The project uses `ruff` for both linting and formatting. The configuration is in `ruff.toml`. Code style is enforced by a pre-commit hook.
-   **Commits:** Before committing, pre-commit hooks will run `ruff` and `pytest` to ensure code quality and that tests pass.
-   **Database Schema:** Any changes to the database schema should be made in `DDL.sql` and tested with the ingestion scripts.
-   **Dependencies:** Add new dependencies to `pyproject.toml`.
