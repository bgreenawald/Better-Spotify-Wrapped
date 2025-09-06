import dash_mantine_components as dmc  # type: ignore
import pandas as pd
from dash import dcc, html


def create_year_dropdown(df: pd.DataFrame) -> html.Div:
    """Create a dropdown for selecting a year from DataFrame timestamps.

    Args:
        df (pd.DataFrame): DataFrame containing a 'ts' datetime column.

    Returns:
        html.Div: Div containing the year selection dropdown.
    """
    years = sorted(df["ts"].dt.year.unique())
    return html.Div(
        [
            html.Label("Select Year", className="filter-label"),
            dcc.Dropdown(
                id="year-dropdown",
                options=[{"label": str(year), "value": year} for year in years],
                value=years[-1] if years else None,
                placeholder="Select year…",
                clearable=False,
                persistence=True,
                persistence_type="local",
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_wrapped_filters_section(df: pd.DataFrame) -> html.Div:
    """Wrap the year dropdown in a titled filters section.

    Args:
        df (pd.DataFrame): DataFrame used to generate filters.

    Returns:
        html.Div: Div containing a Filters title and section.
    """
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div([create_year_dropdown(df)], className="filters-section"),
        ]
    )


def create_global_settings() -> html.Div:
    """Generate global settings filters for exclusions and toggles.
    Returns:
        html.Div: Div containing exclusion dropdowns and radio items.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Excluded Artists", className="filter-label"),
                            dcc.Dropdown(
                                id="excluded-artists-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search artists…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Select Excluded Genres", className="filter-label"),
                            dcc.Dropdown(
                                id="excluded-genres-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search genres…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Excluded Albums", className="filter-label"),
                            dcc.Dropdown(
                                id="excluded-albums-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search albums…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Select Excluded Tracks", className="filter-label"),
                            dcc.Dropdown(
                                id="excluded-tracks-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search tracks…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Exclude December", className="filter-label"),
                            dcc.RadioItems(
                                id="exclude-december",
                                options=[
                                    {"label": "Yes", "value": True},
                                    {"label": "No", "value": False},
                                ],
                                value=True,
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Remove Incognito", className="filter-label"),
                            dcc.RadioItems(
                                id="remove-incognito",
                                options=[
                                    {"label": "Yes", "value": True},
                                    {"label": "No", "value": False},
                                ],
                                value=True,
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_year_range_filter(df: pd.DataFrame) -> html.Div:
    """Create a date range picker based on min/max dates in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'ts' datetime column.

    Returns:
        html.Div: Div containing a date picker range and reset button.
    """
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    start_date = min_ts.date() if hasattr(min_ts, "date") else min_ts
    end_date = max_ts.date() if hasattr(max_ts, "date") else max_ts

    children: list = [html.Label("Select Date Range", className="filter-label")]

    # Mantine v2 DatePickerInput in range mode (dropdown opens/closes)
    children.append(
        dmc.DatePickerInput(
            id="date-range-mc",
            type="range",
            value=[start_date, end_date],
            minDate=start_date,
            maxDate=end_date,
            allowSingleDateInRange=True,
            numberOfColumns=2,
            firstDayOfWeek=1,
            size="sm",
            variant="filled",
            popoverProps={
                "withinPortal": True,
                "zIndex": 4000,
                "position": "bottom-start",
                "offset": 8,
            },
            persistence=True,
            persistence_type="local",
        )
    )
    # Preset chips
    children.append(
        dmc.SegmentedControl(
            id="date-range-preset",
            data=[
                {"label": "Last 60d", "value": "60d"},
                {"label": "YTD", "value": "ytd"},
                {"label": "All", "value": "all"},
                {"label": "Custom", "value": "custom"},
            ],
            value="custom",
            size="xs",
            radius="xl",
            persistence=True,
            persistence_type="local",
            style={"marginTop": "8px"},
        )
    )

    # Inline reset
    children.append(
        html.Button(
            "Reset Date Range",
            id="reset-date-range",
            className="reset-button",
            n_clicks=0,
        )
    )

    # Track how the date range was changed (preset/reset/manual)
    children.append(dcc.Store(id="date-range-source", storage_type="memory"))

    return html.Div(children, className="filter-item")


def create_social_date_range_filter(df: pd.DataFrame) -> html.Div:
    """Create Social tab date range filter (Mantine + presets).

    Uses separate component IDs to avoid collisions with the Trends tab.
    """
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    start_date = min_ts.date() if hasattr(min_ts, "date") else min_ts
    end_date = max_ts.date() if hasattr(max_ts, "date") else max_ts

    children: list = [html.Label("Select Date Range", className="filter-label")]

    children.append(
        dmc.DatePickerInput(
            id="social-date-range-mc",
            type="range",
            value=[start_date, end_date],
            minDate=start_date,
            maxDate=end_date,
            allowSingleDateInRange=True,
            numberOfColumns=2,
            firstDayOfWeek=1,
            size="sm",
            variant="filled",
            popoverProps={
                "withinPortal": True,
                "zIndex": 4000,
                "position": "bottom-start",
                "offset": 8,
            },
            persistence=True,
            persistence_type="local",
        )
    )
    children.append(
        dmc.SegmentedControl(
            id="social-date-range-preset",
            data=[
                {"label": "Last 60d", "value": "60d"},
                {"label": "YTD", "value": "ytd"},
                {"label": "All", "value": "all"},
                {"label": "Custom", "value": "custom"},
            ],
            value="custom",
            size="xs",
            radius="xl",
            persistence=True,
            persistence_type="local",
            style={"marginTop": "8px"},
        )
    )
    children.append(
        html.Button(
            "Reset Date Range",
            id="social-reset-date-range",
            className="reset-button",
            n_clicks=0,
        )
    )
    children.append(dcc.Store(id="social-date-range-source", storage_type="memory"))

    return html.Div(children, className="filter-item")


def create_genre_trends_layout(_df: pd.DataFrame) -> html.Div:
    """Create the layout for the genre trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame of play history.

    Returns:
        html.Div: Div containing genre filters and sliders.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Genres", className="filter-label"),
                            dcc.Dropdown(
                                id="genre-filter-dropdown",
                                # Options are populated dynamically from tab-2-data
                                options=[],
                                multi=True,
                                placeholder="Filter genres…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Exclude Parent Genres (Level 0)",
                                className="filter-label",
                            ),
                            dcc.RadioItems(
                                id="genre-hide-level0-radio",
                                options=[
                                    {"label": "Yes", "value": True},
                                    {"label": "No", "value": False},
                                ],
                                value=False,
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Number of Top Genres", className="filter-label"),
                            dcc.Slider(
                                id="top-genres-slider",
                                min=3,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(3, 16, 3)},
                                className="slider",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Display Type", className="filter-label"),
                            dcc.RadioItems(
                                id="genre-display-type-radio",
                                options=[
                                    {"label": "Play Count", "value": "play_count"},
                                    {"label": "Percentage", "value": "percentage"},
                                ],
                                value="play_count",
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_artist_trends_layout(df: pd.DataFrame) -> html.Div:  # noqa: ARG001
    """Create the layout for the artist trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame containing play history with artist info.

    Returns:
        html.Div: Div containing artist filters and sliders.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Artists", className="filter-label"),
                            dcc.Dropdown(
                                id="artist-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search artists…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Number of Top Artists", className="filter-label"),
                            dcc.Slider(
                                id="top-artist-slider",
                                min=3,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(3, 16, 3)},
                                className="slider",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Display Type", className="filter-label"),
                            dcc.RadioItems(
                                id="artist-display-type-radio",
                                options=[
                                    {"label": "Play Count", "value": "play_count"},
                                    {
                                        "label": "Unique Tracks",
                                        "value": "unique_tracks",
                                    },
                                    {"label": "Percentage", "value": "percentage"},
                                ],
                                value="play_count",
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_track_trends_layout(df: pd.DataFrame) -> html.Div:  # noqa: ARG001
    """Create the layout for the track trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame containing track play history.

    Returns:
        html.Div: Div containing track filters and sliders.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Tracks", className="filter-label"),
                            dcc.Dropdown(
                                id="track-filter-dropdown",
                                # Options populated via server-side search callback
                                options=[],
                                multi=True,
                                placeholder="Type at least 3 characters to search tracks…",
                                persistence=True,
                                persistence_type="local",
                                className="dropdown",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Number of Top Tracks", className="filter-label"),
                            dcc.Slider(
                                id="top-track-slider",
                                min=3,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(3, 16, 3)},
                                className="slider",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label("Display Type", className="filter-label"),
                            dcc.RadioItems(
                                id="track-display-type-radio",
                                options=[
                                    {"label": "Track Count", "value": "track_count"},
                                    {"label": "Percentage", "value": "percentage"},
                                ],
                                value="track_count",
                                className="radio-group",
                                inputClassName="radio-pill",
                                labelClassName="radio-pill-label",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_monthly_trend_filter() -> html.Div:
    """Create a dropdown for selecting monthly trend metrics.

    Returns:
        html.Div: Div containing the metric selection dropdown.
    """
    options = [
        {"label": "Total Hours", "value": "total_hours"},
        {"label": "Unique Tracks", "value": "unique_tracks"},
        {"label": "Unique Artists", "value": "unique_artists"},
        {"label": "Average Hours per Day", "value": "avg_hours_per_day"},
    ]

    return html.Div(
        [
            html.Label("Select Metric", className="filter-label"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=options,
                value="total_hours",
                placeholder="Select metric…",
                clearable=False,
                persistence=True,
                persistence_type="local",
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_trend_filters_section(df: pd.DataFrame) -> html.Div:
    """Wrap the date range filter in a titled trend filters section.

    Args:
        df (pd.DataFrame): DataFrame used to determine date range.

    Returns:
        html.Div: Div containing a Filters title and trend filter section.
    """
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div([create_year_range_filter(df)], className="filters-section"),
        ]
    )
