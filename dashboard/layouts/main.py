import pandas as pd
from dash import html

from dashboard.components.filters import create_filters_section
from dashboard.components.graphs import create_graphs_section


def create_layout(df: pd.DataFrame):
    return html.Div(
        [
            # Header
            html.H1(
                "Spotify Listening History Dashboard",
                style={"textAlign": "center", "marginBottom": "30px"},
            ),
            # Filters
            create_filters_section(df),
            # Graphs
            create_graphs_section(),
            # Stats
            html.Div(id="detailed-stats"),
        ]
    )
