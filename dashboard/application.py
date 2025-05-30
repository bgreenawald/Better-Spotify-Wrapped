"""
Module to create and run Dash Spotify Dashboard application.
"""

import logging
import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash
from dotenv import load_dotenv

from dashboard.callbacks.callbacks import register_callbacks
from dashboard.layouts.layouts import create_layout
from src.api.api import load_api_data
from src.io import load_spotify_history
from src.preprocessing import add_api_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file, overriding existing ones
load_dotenv(override=True)


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

    # Retrieve data directory from environment
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        logger.error("DATA_DIR environment variable is not set.")
        raise EnvironmentError("DATA_DIR environment variable is not set.")

    try:
        # Load user listening history
        logger.info("Loading Spotify history...")
        listening_history_path = Path(data_dir) / "listening_history"
        history_df = load_spotify_history(listening_history_path)

        # Load data from Spotify API
        logger.info("Loading Spotify API data...")
        spotify_data = load_api_data()

        # Merge listening history with API data
        logger.info("Merging API data with listening history...")
        history_df = add_api_data(history_df, spotify_data)
    except Exception:
        logger.exception("Error loading data:")
        # Re-raise exception after logging
        raise

    # Initialize app layout and register callbacks
    logger.info("Initializing layout and callbacks...")
    app.layout = create_layout(history_df, spotify_data)
    register_callbacks(app, history_df, spotify_data)
    logger.info("App initialization complete.")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True)
