"""App-shell callbacks: the filter panel, the tab chrome and the theme."""

from __future__ import annotations

import dash
from dash import Dash, Input, Output, State


def register(app: Dash) -> None:
    """Register the collapse toggle, tab chrome and theme callbacks."""

    @app.callback(
        Output("collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse", "is_open")],
    )
    def toggle_collapse(n_clicks: int, is_open: bool) -> bool:
        """Toggle a collapsible UI component.

        Args:
            n_clicks (int): Number of times the toggle button was clicked.
            is_open (bool): Current open/closed state.

        Returns:
            bool: New open/closed state.
        """
        if n_clicks:
            return not is_open
        return is_open

    # Disable global single-user selector on Social tab
    @app.callback(
        Output("user-id-dropdown", "disabled"),
        Input("main-tabs", "value"),
    )
    def disable_user_dropdown_on_social(tab_value: str):
        return tab_value == "social"

    @app.callback(
        [
            Output("theme-toggle", "checked"),
            Output("app-container", "className"),
            Output("theme-store", "data"),
        ],
        [
            Input("theme-store", "data"),
            Input("theme-toggle", "checked"),
        ],
        prevent_initial_call=False,
    )
    def handle_theme(theme_data, toggle_value):
        """Handle theme initialization and toggling.

        This callback manages both theme initialization from localStorage
        and theme toggling from the toggle switch, preventing callback conflicts.

        Args:
            theme_data (dict): Stored theme data from localStorage.
            toggle_value (bool): Current state of the theme toggle switch.

        Returns:
            tuple: Toggle value, theme class name, icon, and theme data to store.
        """
        ctx = dash.callback_context

        # Determine which input triggered the callback
        if not ctx.triggered:
            # Initial load - use stored theme data
            is_dark = theme_data.get("dark", False) if theme_data else False
        else:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

            if trigger_id == "theme-store":
                # Theme initialization from storage
                is_dark = theme_data.get("dark", False) if theme_data else False
            elif trigger_id == "theme-toggle":
                # Theme toggle from user interaction
                is_dark = toggle_value if toggle_value is not None else False
            else:
                # Fallback
                is_dark = False

        theme_class = "dark-theme" if is_dark else ""
        theme_store_data = {"dark": is_dark}

        return is_dark, theme_class, theme_store_data

    # Update Mantine provider theme for dmc components
    @app.callback(
        Output("mantine-provider", "forceColorScheme"),
        Output("mantine-provider", "theme"),
        Input("theme-store", "data"),
    )
    def set_mantine_theme(theme_data):
        is_dark = bool(theme_data and theme_data.get("dark"))
        scheme = "dark" if is_dark else "light"
        return scheme, {"primaryColor": "green"}
