"""Social tab: overlap between several users' listening.

`compute_social_data` does the shared/exclusive-region work for the selected
users and writes it to `social-data`; `render_social` turns one chosen region
into a Venn diagram and a table of what sits inside it.
"""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, html

from dashboard.callbacks.helpers import register_highcharts_renderer
from dashboard.conn import db_cursor
from src.metrics.social import compute_social_regions


def register(app: Dash) -> None:
    """Register the Social tab's callbacks."""
    # -------------------------- Social tab callbacks --------------------------

    @app.callback(
        Output("social-data", "data"),
        [
            Input("social-users-dropdown", "value"),
            Input("social-date-range-mc", "value"),
            Input("social-mode", "value"),
            Input("social-genre-hide-level0-radio", "value"),
            # Global filters respected
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-genres-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def compute_social_data(
        users,
        date_range_value,
        mode,
        hide_parent_genres,
        excluded_artists,
        excluded_genres,
        excluded_albums,
        excluded_tracks,
        exclude_december,
        remove_incognito,
    ):
        # Validate selection
        users = users or []
        if len(users) < 2:
            return {
                "error": "Select 2–3 users to compare.",
                "users": users,
            }
        if len(users) > 3:
            users = users[:3]

        # Parse date range
        start_date = None
        end_date = None
        if date_range_value and isinstance(date_range_value, list) and len(date_range_value) == 2:
            start_date = pd.to_datetime(date_range_value[0]) if date_range_value[0] else None
            end_date = pd.to_datetime(date_range_value[1]) if date_range_value[1] else None

        # Compute with DuckDB backend
        with db_cursor() as con:
            kwargs = {
                "con": con,
                "users": users,
                "start": pd.to_datetime(start_date) if start_date else None,
                "end": pd.to_datetime(end_date) if end_date else None,
                "mode": mode or "tracks",
                "exclude_december": bool(exclude_december),
                "remove_incognito": bool(remove_incognito),
                "excluded_tracks": excluded_tracks,
                "excluded_artists": excluded_artists,
                "excluded_albums": excluded_albums,
                "excluded_genres": excluded_genres,
                "limit_per_region": 10,
            }
            if (mode or "tracks") == "genres":
                kwargs["hide_parent_genres"] = bool(hide_parent_genres)
            out = compute_social_regions(**kwargs)
        return out

    # Show/hide Social genre parent toggle depending on mode
    @app.callback(
        Output("social-genre-hide-level0-container", "style"),
        Input("social-mode", "value"),
    )
    def toggle_social_genre_parent_visibility(mode):
        if mode == "genres":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("social-venn-options", "data"),
        Output("social-region-lists", "children"),
        Input("social-data", "data"),
        Input("social-selected-region", "data"),
    )
    def render_social(data, selected_region):
        # The clientside renderer merges the active theme over these options,
        # so the values below are only defaults and need not branch on it.
        is_dark = False
        if not data or data.get("error"):
            msg = data.get("error") if data else "Select 2–3 users to compare."
            return None, html.Div(msg)
        user_counts = data.get("user_counts", {})
        if any(user_counts.get(str(u), 0) == 0 for u in data.get("users", [])):
            return None, html.Div(
                "One or more selected users have no plays in range; adjust filters or deselect."
            )

        users = data.get("users", [])
        regions = data.get("regions", {})
        totals = data.get("totals", {})
        user_labels = data.get("user_labels", {})

        def lbl(u):
            return user_labels.get(str(u), str(u))

        series_data: list[dict] = []
        if len(users) == 2:
            u1, u2 = users
            inter_12 = int(totals.get(f"{u1}_{u2}", 0) or 0)
            u1_only = int(totals.get(f"{u1}_only", 0) or 0)
            u2_only = int(totals.get(f"{u2}_only", 0) or 0)
            u1_total = u1_only + inter_12
            u2_total = u2_only + inter_12
            series_data = [
                {"sets": [lbl(u1)], "value": u1_total, "name": lbl(u1), "name_key": f"{u1}_only"},
                {"sets": [lbl(u2)], "value": u2_total, "name": lbl(u2), "name_key": f"{u2}_only"},
                {
                    "sets": [lbl(u1), lbl(u2)],
                    "value": inter_12,
                    "name": f"{lbl(u1)} ∩ {lbl(u2)}",
                    "name_key": f"{u1}_{u2}",
                },
            ]
        else:
            u1, u2, u3 = users
            inter_12 = int(totals.get(f"{u1}_{u2}", 0) or 0)
            inter_13 = int(totals.get(f"{u1}_{u3}", 0) or 0)
            inter_23 = int(totals.get(f"{u2}_{u3}", 0) or 0)
            inter_123 = int(totals.get(f"{u1}_{u2}_{u3}", 0) or 0)
            u1_only = int(totals.get(f"{u1}_only", 0) or 0)
            u2_only = int(totals.get(f"{u2}_only", 0) or 0)
            u3_only = int(totals.get(f"{u3}_only", 0) or 0)
            u1_total = u1_only + inter_12 + inter_13 + inter_123
            u2_total = u2_only + inter_12 + inter_23 + inter_123
            u3_total = u3_only + inter_13 + inter_23 + inter_123
            series_data = [
                {"sets": [lbl(u1)], "value": u1_total, "name": lbl(u1), "name_key": f"{u1}_only"},
                {"sets": [lbl(u2)], "value": u2_total, "name": lbl(u2), "name_key": f"{u2}_only"},
                {"sets": [lbl(u3)], "value": u3_total, "name": lbl(u3), "name_key": f"{u3}_only"},
                {
                    "sets": [lbl(u1), lbl(u2)],
                    "value": inter_12 + inter_123,
                    "name": f"{lbl(u1)} ∩ {lbl(u2)}",
                    "name_key": f"{u1}_{u2}",
                },
                {
                    "sets": [lbl(u1), lbl(u3)],
                    "value": inter_13 + inter_123,
                    "name": f"{lbl(u1)} ∩ {lbl(u3)}",
                    "name_key": f"{u1}_{u3}",
                },
                {
                    "sets": [lbl(u2), lbl(u3)],
                    "value": inter_23 + inter_123,
                    "name": f"{lbl(u2)} ∩ {lbl(u3)}",
                    "name_key": f"{u2}_{u3}",
                },
            ]

        options = {
            "chart": {
                "type": "venn",
                "height": 380,
                "backgroundColor": "#1e1e1e" if is_dark else "white",
            },
            "title": {"text": None},
            "credits": {"enabled": False},
            "tooltip": {"useHTML": True},
            "plotOptions": {
                "series": {"states": {"inactive": {"opacity": 0.25}}, "cursor": "pointer"}
            },
            "series": [{"type": "venn", "data": series_data}],
        }

        def _region_block(title: str, items: list[dict], total: int | None = None):
            rows = []
            for it in items[:10]:
                tooltip = " | ".join(
                    [
                        f"{lbl(u)}: r{it['ranks'].get(u, '-')}, {it['counts'].get(u, 0)} plays"
                        for u in users
                        if u in it.get("ranks", {})
                    ]
                )
                ta = None
                if (data or {}).get("mode") == "genres":
                    ta = it.get("top_artists")
                tail = f" — Top: {', '.join([str(x) for x in ta[:2]])}" if ta else ""
                rows.append(
                    html.Li([html.Span(it.get("name", "")), html.Span(f" ({tooltip}){tail}")])
                )
            if not rows:
                rows = [html.Li("No items in this region")]
            subtitle = None
            if total is not None and total > len(items):
                subtitle = html.Div(
                    f"+ {total - len(items)} more not shown", className="card-subtitle"
                )
            return html.Div(
                [html.H5(title, className="card-title"), subtitle, html.Ul(rows)], className="card"
            )

        blocks = []
        if len(users) == 2:
            u1, u2 = users
            if selected_region:
                key = selected_region
                title = (
                    f"{u1} only"
                    if key == f"{u1}_only"
                    else (f"{u2} only" if key == f"{u2}_only" else f"{u1} ∩ {u2}")
                )
                blocks.append(_region_block(title, regions.get(key, []), totals.get(key)))
            else:
                blocks.append(
                    html.Div(
                        [html.Div("Click a region to show details", className="card-subtitle")],
                        className="card",
                    )
                )
        else:
            u1, u2, u3 = users
            if selected_region:
                key = selected_region
                if key == f"{u1}_only":
                    title = f"{u1} only"
                elif key == f"{u2}_only":
                    title = f"{u2} only"
                elif key == f"{u3}_only":
                    title = f"{u3} only"
                elif key == f"{u1}_{u2}":
                    title = f"{u1} ∩ {u2}"
                elif key == f"{u1}_{u3}":
                    title = f"{u1} ∩ {u3}"
                elif key == f"{u2}_{u3}":
                    title = f"{u2} ∩ {u3}"
                else:
                    title = f"{u1} ∩ {u2} ∩ {u3}"
                blocks.append(_region_block(title, regions.get(key, []), totals.get(key)))
            else:
                blocks.append(
                    html.Div(
                        [html.Div("Click a region to show details", className="card-subtitle")],
                        className="card",
                    )
                )

        mode = (data or {}).get("mode", "tracks")
        consider_caps = {"tracks": 250, "artists": 100, "genres": 50}
        cap = consider_caps.get(mode, 10)
        note = html.Div(
            (
                "Note: region totals are exact. Ranking considers the top "
                f"{cap} items for this mode (Tracks 250, Artists 100, Genres 50) "
                "and the lists show up to 10. Less-specific regions exclude "
                "items already shown in a more-specific one."
            ),
            className="card-subtitle",
        )

        return options, html.Div([note, html.Div(blocks, className="graph-container")])

    # Highcharts Social render + selection
    register_highcharts_renderer(
        app, "social-venn-container", "social-venn-options", function_name="render_venn"
    )
