"""
Module to create and run Dash Spotify Dashboard application.

Updated to use preloaded DuckDB data rather than loading JSON/API at runtime.
"""

import contextlib
import logging
import os
import sys
from pathlib import Path

# Running this file directly (`python dashboard/application.py`, the command
# both README and CLAUDE.md document) puts dashboard/ on sys.path rather than
# the repository root, so `import dashboard...` below failed with
# ModuleNotFoundError. Put the root on the path first so the documented
# command works; `python -m dashboard.application` was unaffected.
if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.metrics.utils import local_ts_sql

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


def _load_layout_bounds(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load just what the layout needs to build its filter controls.

    The layout uses only three things from this frame: the set of user ids,
    the overall min/max timestamp for the date pickers, and the distinct years
    for the year dropdown. It previously received the entire enriched plays
    table -- 525k rows and ~318 MB resident -- to answer those three
    questions.

    Returning two rows per (user, year) is enough: the global min and max are
    exact, every year is present, and every user appears.

    Args:
        conn: Open DuckDB connection.

    Returns:
        A small DataFrame with `user_id` and `ts` columns.
    """
    local_ts = local_ts_sql("played_at")
    sql = f"""
        SELECT user_id,
               EXTRACT(year FROM {local_ts}) AS yr,
               MIN({local_ts}) AS min_ts,
               MAX({local_ts}) AS max_ts
        FROM fact_plays
        GROUP BY user_id, yr
        ORDER BY user_id, yr
    """
    agg = conn.execute(sql).df()
    if agg.empty:
        return pd.DataFrame(
            {"user_id": pd.Series(dtype=str), "ts": pd.Series(dtype="datetime64[ns]")}
        )

    rows = pd.concat(
        [
            agg[["user_id", "min_ts"]].rename(columns={"min_ts": "ts"}),
            agg[["user_id", "max_ts"]].rename(columns={"max_ts": "ts"}),
        ],
        ignore_index=True,
    )
    rows["ts"] = pd.to_datetime(rows["ts"], errors="coerce")
    return rows.dropna(subset=["ts"]).reset_index(drop=True)


def create_app() -> Dash:
    """Create and configure the Dash application.

    Reads the user ids and timestamp bounds the filters need, builds the
    layout and registers the callbacks. It does not load play data: each
    callback queries DuckDB for what it needs.

    Returns:
        Dash: Configured Dash application.

    Raises:
        Exception: Propagated from opening the database.
    """
    # Path to dashboard assets directory
    assets_path = Path(__file__).parent / "assets"
    # Initialize Dash app with external Bootstrap stylesheet
    app = Dash(
        __name__,
        title="Slopify Wrapped",
        assets_folder=str(assets_path),
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
    # Allow callbacks to reference components that are rendered dynamically
    # (Tab 2 only mounts one selected chart at a time)
    app.config.suppress_callback_exceptions = True

    # Build base DataFrame from DB for current UI
    try:
        conn = get_db_connection()
        history_df = _load_layout_bounds(conn)
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
