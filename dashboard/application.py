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

load_dotenv(override=True)


def create_app():
    assets_path = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        assets_folder=str(assets_path),
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
    DATA_DIR = os.getenv("DATA_DIR")
    if not DATA_DIR:
        print("[ERROR] DATA_DIR environment variable is not set.")
        raise EnvironmentError("DATA_DIR environment variable is not set.")

    try:
        print("[INFO] Loading Spotify history...")
        df = load_spotify_history(Path(DATA_DIR) / "listening_history")
        print("[INFO] Loading Spotify API data...")
        spotify_data = load_api_data()
        print("[INFO] Merging API data...")
        df = add_api_data(df, spotify_data)
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        raise

    print("[INFO] Initializing layout and callbacks...")
    app.layout = create_layout(df, spotify_data)
    register_callbacks(app, df, spotify_data)
    print("[INFO] App initialization complete.")
    return app


if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True)
