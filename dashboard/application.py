import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash
from dotenv import load_dotenv

from dashboard.callbacks.callbacks import register_callbacks
from dashboard.layouts.layouts import create_layout
from src.api.api import load_api_data
from src.io import load_spotify_history

load_dotenv()


def create_app():
    assets_path = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        assets_folder=str(assets_path),
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )
    DATA_DIR = Path(os.getenv("DATA_DIR"))

    # Load data
    df = load_spotify_history(DATA_DIR / "listening_history")
    spotify_data = load_api_data()

    # Create layout
    app.layout = create_layout(df)

    # Register callbacks
    register_callbacks(app, df, spotify_data)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=True)
