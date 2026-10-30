"""Entry point for the dashboard's callbacks.

The callbacks used to live in one ~2,500-line function. They are now split by
the part of the UI they serve, and this module only wires those parts together.
`register_callbacks` keeps its name and signature so `dashboard.application`
is unaffected.

Where to look:
    shell       the filter panel, tab chrome and light/dark theme
    search      typed search for every filter dropdown
    date_range  the reset button and preset chips, for both tabs
    wrapped     the Wrapped tab's data store and figures
    social      the Social tab's overlap regions and Venn diagram
    trends      the Trends tab's month-by-month charts
    cache       the in-process LRU caches those modules share
    helpers     formatting, axis labels and the clientside chart renderer
"""

from __future__ import annotations

import pandas as pd
from dash import Dash

from dashboard.callbacks import date_range, search, shell, social, trends, wrapped


def register_callbacks(app: Dash, df: pd.DataFrame) -> None:
    """Register all Dash callbacks for the listening dashboard.

    Args:
        app (Dash): Dash application instance.
        df (pd.DataFrame): DataFrame of listening events.
    """
    shell.register(app)
    search.register(app)
    date_range.register(app, df)
    wrapped.register(app, df)
    social.register(app)
    trends.register(app, df)
