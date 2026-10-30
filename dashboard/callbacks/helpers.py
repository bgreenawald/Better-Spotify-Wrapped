"""Formatting, axis and chart-shell helpers shared by the dashboard callbacks."""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, State
from dash.dependencies import ClientsideFunction


def fmt_genres(g):
    """Format a genre value that may be a scalar, list-like, or NaN."""
    if isinstance(g, list | tuple | set):
        return ", ".join(map(str, g))
    # Handle numpy arrays or other iterables (but not strings/dicts)
    try:
        if hasattr(g, "__iter__") and not isinstance(g, str | bytes | dict):
            return ", ".join(map(str, list(g)))
    except (TypeError, ValueError):
        pass  # Not iterable after all; try the scalar paths below.
    try:
        if pd.isna(g):
            return ""
    except (TypeError, ValueError):
        pass  # isna rejects some containers; treat as a value.
    if isinstance(g, str):
        return g
    return str(g) if g is not None else ""


def month_categories(months: pd.Series) -> list[str]:
    """Return every calendar month from the first to the last, inclusive.

    Using only the months present drops silent ones from the axis entirely, so
    a line joins across a gap as though listening had continued. Filling the
    span means a month with no plays shows as zero.

    Args:
        months: Month labels in "YYYY-MM" form; may contain gaps or repeats.

    Returns:
        Sorted "YYYY-MM" labels covering the whole span.
    """
    labels = sorted({str(m) for m in months.dropna().astype(str) if str(m).strip()})
    if not labels:
        return []
    try:
        span = pd.period_range(start=labels[0], end=labels[-1], freq="M")
    except Exception:
        return labels
    return [str(p) for p in span]


def empty_chart(message: str) -> dict:
    """Chart options that render an explicit "nothing here" state.

    Returning None left the previously rendered chart on screen, so a filter
    that matched nothing looked like stale data rather than an empty result.

    Args:
        message: Short explanation to show in the plot area.

    Returns:
        A minimal Highcharts options dict with no series.
    """
    return {
        "chart": {"height": 300},
        "title": {"text": message, "style": {"fontSize": "14px", "fontWeight": "normal"}},
        "credits": {"enabled": False},
        "legend": {"enabled": False},
        "xAxis": {"visible": False},
        "yAxis": {"visible": False},
        "series": [],
    }


def truncate_label(s: str, max_len: int = 28) -> str:
    """Shorten one axis label, with an ellipsis when it is cut."""
    try:
        s = str(s)
    except Exception:
        s = ""
    return (s[: max_len - 1] + "…") if len(s) > max_len else s


def wrap_or_truncate_labels(labels: list[str]) -> list[str]:
    """Shorten a set of axis labels to a width the chart can afford."""
    # Prefer a single-line truncation with a shorter limit to preserve width
    return [truncate_label(x, 28) for x in labels]


def compute_left_margin(labels: list[str]) -> int:
    """Approximate the left margin the longest label needs, in pixels."""
    max_len = max((len(str(x)) for x in labels), default=0)
    px = int(8 + max_len * 6.0)  # 6.0 px per character heuristic
    return max(110, min(px, 210))


def register_highcharts_renderer(
    app: Dash,
    container_id: str,
    options_id: str,
    *,
    rerender_on_tab_content: bool = False,
    function_name: str = "render_single",
) -> None:
    """Wire one options store to its clientside Highcharts container.

    Every chart in the dashboard is rendered the same way: the server callback
    writes Highcharts options to a `dcc.Store`, and this clientside callback
    draws them, merging the current theme over the top. Registering them
    through one function keeps the nine identical blocks from drifting apart.

    Args:
        app: Dash application instance.
        container_id: The div the chart is drawn into.
        options_id: The store holding that chart's Highcharts options.
        rerender_on_tab_content: True for charts inside the Trends tab, whose
            container is re-mounted when the chart selector changes and so must
            redraw on that too.
        function_name: Clientside renderer to use; the Venn diagram has its own.
    """
    inputs = [
        Input("theme-store", "data"),
        Input(options_id, "data"),
        Input("main-tabs", "value"),
    ]
    if rerender_on_tab_content:
        inputs.append(Input("tab-2-content", "children"))
    app.clientside_callback(
        ClientsideFunction(namespace="highcharts", function_name=function_name),
        Output(container_id, "children"),
        *inputs,
        State(container_id, "id"),
        prevent_initial_call=False,
    )
