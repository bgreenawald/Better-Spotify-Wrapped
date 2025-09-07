"""
Module to create and run Dash Spotify Dashboard application.

Updated to use preloaded DuckDB data rather than loading JSON/API at runtime.
"""

import contextlib
import logging
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc  # type: ignore
import duckdb
import pandas as pd
from dash import Dash, dcc, html
from dotenv import load_dotenv

from dashboard.callbacks.callbacks import register_callbacks
from dashboard.components.filters import (
    create_artist_trends_layout,
    create_genre_trends_layout,
    create_track_trends_layout,
)
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
        WITH plays AS (
            SELECT
                ve.user_id AS user_id,
                ve.played_at AS ts,
                ve.duration_ms AS ms_played,
                ve.reason_start,
                ve.reason_end,
                ve.skipped,
                ve.incognito_mode,
                ve.track_id,
                ve.track_name AS master_metadata_track_name,
                REGEXP_REPLACE(TRIM(ve.artist_id), '.*:', '') AS artist_id,
                ve.artist_name AS master_metadata_album_artist_name,
                ve.album_name AS master_metadata_album_album_name
            FROM v_plays_enriched ve
        ), artist_genres_agg AS (
            -- Build formatted artist genres as "Child (Parent1, Parent2)" when parents exist;
            -- otherwise just the genre name. De-duplicate labels per artist.
            WITH parent_map AS (
                SELECT gh.child_genre_id,
                       STRING_AGG(DISTINCT pg.name, ', ' ORDER BY pg.name) AS parent_names
                FROM genre_hierarchy gh
                JOIN dim_genres pg ON pg.genre_id = gh.parent_genre_id
                WHERE COALESCE(pg.active, TRUE)
                GROUP BY gh.child_genre_id
            ), labels AS (
                SELECT DISTINCT COALESCE(TRIM(REGEXP_REPLACE(ag.artist_id, '.*:', '')), '') AS artist_id,
                       CASE
                           WHEN COALESCE(pm.parent_names, '') = '' THEN g.name
                           ELSE (g.name || ' (' || pm.parent_names || ')')
                       END AS label
                FROM artist_genres ag
                JOIN dim_genres g ON g.genre_id = ag.genre_id
                LEFT JOIN parent_map pm ON pm.child_genre_id = g.genre_id
                WHERE COALESCE(g.active, TRUE)
            )
            SELECT artist_id, list(label) AS artist_genres
            FROM labels
            GROUP BY artist_id
        )
        SELECT p.*, aga.artist_genres
        FROM plays p
        LEFT JOIN artist_genres_agg aga ON aga.artist_id = p.artist_id
    """
    df = conn.execute(sql).df()
    if not df.empty:
        # Synthesize legacy fields expected by callbacks/filters
        df["spotify_track_uri"] = df["track_id"].apply(
            lambda x: f"spotify:track:{x}" if pd.notna(x) else None
        )
        # No podcast episodes in DB-backed path; provide NaNs for compatibility
        df["episode_name"] = pd.NA
        # Ensure genres column is present and normalize nulls to empty tuples for consistency
        if "artist_genres" not in df.columns:
            df["artist_genres"] = [() for _ in range(len(df))]
        else:

            def _to_tuple(v):
                try:
                    if v is None:
                        return ()
                    if isinstance(v, list | tuple | set):
                        return tuple(v)
                    # Handle numpy arrays or other iterables (exclude strings/dicts)
                    if hasattr(v, "__iter__") and not isinstance(v, str | bytes | dict):
                        return tuple(v)
                except Exception:
                    pass
                return ()

            df["artist_genres"] = df["artist_genres"].apply(_to_tuple)
        # Ensure ts is datetime
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        # End normalization
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
    base_layout = create_layout(history_df)
    app.layout = dmc.MantineProvider(
        id="mantine-provider",
        defaultColorScheme="light",
        withCssVariables=True,
        theme={
            "primaryColor": "green",
        },
        children=base_layout,
    )

    # Validation layout includes dynamic tab contents to satisfy callback ID checks
    # and prevent "nonexistent object used in Input" errors during chart switching.
    validation_children = [
        create_layout(history_df),
        # Tab 2 dynamic layouts and targets
        html.Div(
            [
                # Genres
                html.Div(
                    [
                        html.H3("Genres Over Time", className="card-title"),
                        # Controls
                        create_genre_trends_layout(history_df),
                        # Targets
                        dcc.Store(id="genre-trends-options"),
                        html.Div(
                            id="genre-trends-container",
                            children=html.Div(id="genre-trends-container-root"),
                        ),
                        # Keep legacy Graph for validation of existing callbacks
                        dcc.Graph(id="genre-trends-graph"),
                        html.Div(id="genre-trends-table"),
                    ]
                ),
                # Artists
                html.Div(
                    [
                        html.H3("Artists Over Time", className="card-title"),
                        create_artist_trends_layout(history_df),
                        dcc.Store(id="artist-trends-options"),
                        html.Div(
                            id="artist-trends-container",
                            children=html.Div(id="artist-trends-container-root"),
                        ),
                        dcc.Graph(id="artist-trends-graph"),
                        html.Div(id="artist-trends-table"),
                    ]
                ),
                # Tracks
                html.Div(
                    [
                        html.H3("Tracks Over Time", className="card-title"),
                        create_track_trends_layout(history_df),
                        dcc.Store(id="track-trends-options"),
                        html.Div(
                            id="track-trends-container",
                            children=html.Div(id="track-trends-container-root"),
                        ),
                        dcc.Graph(id="track-trends-graph"),
                        html.Div(id="track-trends-table"),
                    ]
                ),
                # Listening (monthly)
                html.Div(
                    [
                        html.H3("Listening Over Time", className="card-title"),
                        dcc.Store(id="trends-options"),
                        html.Div(
                            id="trends-container", children=html.Div(id="trends-container-root")
                        ),
                        dcc.Graph(id="trends-graph"),
                    ]
                ),
            ]
        ),
    ]
    # Keep validation layout minimal and avoid duplicating MantineProvider IDs
    app.validation_layout = html.Div(validation_children)
    register_callbacks(app, history_df)
    logger.info("App initialization complete.")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
