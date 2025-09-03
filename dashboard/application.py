"""
Module to create and run Dash Spotify Dashboard application.

Updated to use preloaded DuckDB data rather than loading JSON/API at runtime.
"""

import contextlib
import logging
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import duckdb
import pandas as pd
from dash import Dash
from dotenv import load_dotenv

from dashboard.callbacks.callbacks import register_callbacks
from dashboard.conn import get_db_connection
from dashboard.layouts.layouts import create_layout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Only override environment variables in explicit local/test scenarios to prevent overwriting deploy-time env vars
should_override = os.environ.get("ENVIRONMENT", "") in ["local", "dev", "development", "test"]
load_dotenv(override=should_override)


def _load_base_dataframe(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load a minimal, UI-compatible DataFrame from the preloaded database.

    Returns columns used by filters and existing metrics code paths to avoid
    changing downstream logic before full migration to SQL-backed queries.
    """
    sql = """
        SELECT
            p.user_id                         AS user_id,
            p.played_at                        AS ts,
            p.duration_ms                      AS ms_played,
            p.reason_start,
            p.reason_end,
            p.skipped,
            p.incognito_mode,
            t.track_id,
            t.track_name                       AS master_metadata_track_name,
            ar.artist_id                       AS artist_id,
            ar.artist_name                     AS master_metadata_album_artist_name,
            al.album_name                      AS master_metadata_album_album_name
        FROM fact_plays p
        LEFT JOIN dim_tracks t ON t.track_id = p.track_id
        LEFT JOIN bridge_track_artists b
          ON b.track_id = p.track_id AND b.role = 'primary'
        LEFT JOIN dim_artists ar ON ar.artist_id = b.artist_id
        LEFT JOIN dim_albums  al ON al.album_id = t.album_id
    """
    df = conn.execute(sql).df()
    if not df.empty:
        # Synthesize legacy fields expected by callbacks/filters
        df["spotify_track_uri"] = df["track_id"].apply(
            lambda x: f"spotify:track:{x}" if pd.notna(x) else None
        )
        # No podcast episodes in DB-backed path; provide NaNs for compatibility
        df["episode_name"] = pd.NA
        # Placeholder until genre taxonomy is wired in the UI layer
        df["artist_genres"] = [() for _ in range(len(df))]
        # Ensure ts is datetime
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def create_app() -> Dash:
    """Create and configure the Dash application.

    Loads Spotify listening history and API data, merges them, sets up app
    layout and callbacks, and returns the Dash app instance.

    Returns:
        Dash: Configured Dash application.

    Raises:
        EnvironmentError: If DATA_DIR environment variable is not set.
        Exception: Propagated from data loading operations.
    """
    # Path to dashboard assets directory
    assets_path = Path(__file__).parent / "assets"
    # Initialize Dash app with external Bootstrap stylesheet
    app = Dash(
        __name__,
        assets_folder=str(assets_path),
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
    # Allow callbacks to reference components that are rendered dynamically
    # (Tab 2 only mounts one selected chart at a time)
    app.config.suppress_callback_exceptions = True

    # Build base DataFrame from DB for current UI
    try:
        conn = get_db_connection()
        history_df = _load_base_dataframe(conn)
    except Exception:
        logger.exception("Error reading from DuckDB:")
        raise
    finally:
        with contextlib.suppress(Exception):
            conn.close()

    # Initialize app layout and register callbacks
    logger.info("Initializing layout and callbacks...")
    app.layout = create_layout(history_df)
    register_callbacks(app, history_df)
    logger.info("App initialization complete.")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True)
