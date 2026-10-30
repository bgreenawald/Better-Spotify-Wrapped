"""Date-range controls: the reset button, the preset chips and their feedback.

The Wrapped/Trends tabs and the Social tab each carry a full set of these, with
their own component ids and their own idea of "the selected users" -- one
dropdown value against a list of them. The two sets had drifted into six
near-identical callbacks; they are now one machine registered twice.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import dash
import pandas as pd
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

# Preset chip -> how far back from the end of the data it reaches. "ytd" and
# "all" are anchored rather than relative, so they are handled separately.
_RELATIVE_PRESETS = {
    "60d": timedelta(days=60),
    "2y": timedelta(days=2 * 365),
    "5y": timedelta(days=5 * 365),
}

# A selector narrows the full frame to the rows the control should span.
Selector = Callable[[pd.DataFrame, object], pd.DataFrame]


def _as_date(ts) -> datetime:
    """Drop the time of day; these controls pick whole days."""
    return datetime(ts.year, ts.month, ts.day)


def _bounds(df: pd.DataFrame, selector: Selector, selection) -> tuple[datetime, datetime]:
    """The first and last day present for the current selection.

    Falls back to the whole frame when the selection matches nothing, so the
    picker still lands on a usable range instead of raising.
    """
    subset = selector(df, selection)
    if subset.empty:
        subset = df
    return _as_date(subset["ts"].min()), _as_date(subset["ts"].max())


def _single_user(df: pd.DataFrame, user_id) -> pd.DataFrame:
    """Rows for the one user the global dropdown has selected."""
    if not user_id:
        return df
    return df[df.get("user_id") == user_id]


def _many_users(df: pd.DataFrame, users_selected) -> pd.DataFrame:
    """Rows for the several users the Social tab has selected."""
    if not users_selected:
        return df
    return df[df.get("user_id").isin(users_selected)]


def _register_controls(
    app: Dash,
    df: pd.DataFrame,
    *,
    range_id: str,
    source_id: str,
    preset_id: str,
    reset_id: str,
    selection_id: str,
    selector: Selector,
) -> None:
    """Register one tab's reset / preset / manual-change trio.

    Args:
        app: Dash application instance.
        df: Frame holding the `ts` and `user_id` columns the bounds come from.
        range_id: The Mantine date-range picker.
        source_id: Store recording what caused the last change, so a
            programmatic change is not mistaken for the user typing a date.
        preset_id: The preset chip group.
        reset_id: The "reset to full range" button.
        selection_id: The user selector this tab reads.
        selector: Narrows `df` to that selection.
    """

    @app.callback(
        Output(range_id, "value", allow_duplicate=True),
        Output(source_id, "data", allow_duplicate=True),
        Input(reset_id, "n_clicks"),
        State(selection_id, "value"),
        prevent_initial_call=True,
    )
    def reset_date_range(n_clicks: int, selection):
        """Reset the date picker to span the full range of the data."""
        if not n_clicks:
            raise PreventUpdate
        start, end = _bounds(df, selector, selection)
        return [start, end], "reset"

    @app.callback(
        Output(range_id, "value", allow_duplicate=True),
        Output(source_id, "data", allow_duplicate=True),
        Input(preset_id, "value"),
        State(selection_id, "value"),
        prevent_initial_call=True,
    )
    def apply_date_preset(preset: str, selection):
        """Translate a preset chip into an explicit date range."""
        if not preset or preset == "custom":
            raise PreventUpdate
        min_date, end = _bounds(df, selector, selection)
        if preset in _RELATIVE_PRESETS:
            start = end - _RELATIVE_PRESETS[preset]
        elif preset == "ytd":
            start = datetime(end.year, 1, 1)
        elif preset == "all":
            start = min_date
        else:
            raise PreventUpdate
        return [start, end], "preset"

    @app.callback(
        Output(preset_id, "value"),
        Output(source_id, "data", allow_duplicate=True),
        Input(range_id, "value"),
        State(source_id, "data"),
        prevent_initial_call=True,
    )
    def set_preset_on_date_change(value, source):
        """Mark the preset "custom", but only when the user moved the dates."""
        if not value or len(value) != 2:
            raise PreventUpdate
        if source in {"preset", "reset"}:
            # Clear the source so a later manual change is still detected.
            return dash.no_update, None
        return "custom", "manual"


def register(app: Dash, df: pd.DataFrame) -> None:
    """Register the date-range controls for both tabs."""
    _register_controls(
        app,
        df,
        range_id="date-range-mc",
        source_id="date-range-source",
        preset_id="date-range-preset",
        reset_id="reset-date-range",
        selection_id="user-id-dropdown",
        selector=_single_user,
    )
    _register_controls(
        app,
        df,
        range_id="social-date-range-mc",
        source_id="social-date-range-source",
        preset_id="social-date-range-preset",
        reset_id="social-reset-date-range",
        selection_id="social-users-dropdown",
        selector=_many_users,
    )
