from itertools import chain

import pandas as pd
from dash import dcc, html

from src.metrics.trends import get_genre_trends


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


def create_global_settings(df: pd.DataFrame) -> html.Div:
    """Generate global settings filters for exclusions and toggles.

    Args:
        df (pd.DataFrame): DataFrame containing metadata columns.

    Returns:
        html.Div: Div containing exclusion dropdowns and radio items.
    """
    tracks = df["master_metadata_track_name"].dropna().unique().tolist()
    artists = df["master_metadata_album_artist_name"].dropna().unique().tolist()
    albums = df["master_metadata_album_album_name"].dropna().unique().tolist()
    genres = sorted(set(chain.from_iterable(df["artist_genres"].dropna())))

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Excluded Artists", className="filter-label"),
                            dcc.Dropdown(
                                id="excluded-artists-filter-dropdown",
                                options=[{"label": artist, "value": artist} for artist in artists],
                                multi=True,
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
                                options=[{"label": genre, "value": genre} for genre in genres],
                                multi=True,
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
                                options=[{"label": album, "value": album} for album in albums],
                                multi=True,
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
                                options=[{"label": track, "value": track} for track in tracks],
                                multi=True,
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

    return html.Div(
        [
            html.Label("Select Date Range", className="filter-label"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=start_date,
                max_date_allowed=end_date,
                start_date=start_date,
                end_date=end_date,
                className="datepicker",
            ),
            html.Button(
                "Reset Date Range",
                id="reset-date-range",
                className="reset-button",
                n_clicks=0,
            ),
        ],
        className="filter-item",
    )


def create_genre_trends_layout(df: pd.DataFrame) -> html.Div:
    """Create the layout for the genre trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame of play history.

    Returns:
        html.Div: Div containing genre filters and sliders.
    """
    genres_df = get_genre_trends(df, db_path="data/db/music.db")
    genres = sorted(genres_df["genre"].dropna().unique())

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Genres", className="filter-label"),
                            dcc.Dropdown(
                                id="genre-filter-dropdown",
                                options=[
                                    {"label": genre.title(), "value": genre} for genre in genres
                                ],
                                multi=True,
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
                            html.Label("Number of Top Genres", className="filter-label"),
                            dcc.Slider(
                                id="top-genres-slider",
                                min=3,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(3, 16, 3)},
                                className="slider",
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
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_artist_trends_layout(df: pd.DataFrame) -> html.Div:
    """Create the layout for the artist trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame containing play history with artist info.

    Returns:
        html.Div: Div containing artist filters and sliders.
    """
    artists = sorted(df["master_metadata_album_artist_name"].dropna().unique())

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Artists", className="filter-label"),
                            dcc.Dropdown(
                                id="artist-filter-dropdown",
                                options=[{"label": artist, "value": artist} for artist in artists],
                                multi=True,
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
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
        ]
    )


def create_track_trends_layout(df: pd.DataFrame) -> html.Div:
    """Create the layout for the track trends analysis section.

    Args:
        df (pd.DataFrame): DataFrame containing track play history.

    Returns:
        html.Div: Div containing track filters and sliders.
    """
    df_tracks = (
        df.drop_duplicates(
            subset=[
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
            ]
        )
        .dropna(
            subset=[
                "master_metadata_track_name",
                "master_metadata_album_artist_name",
            ]
        )
        .sort_values(
            by="master_metadata_track_name",
            ascending=True,
        )
    )

    options = [
        {
            "label": (
                f"{row.master_metadata_track_name} - {row.master_metadata_album_artist_name}"
            ),
            "value": (
                f"{row.master_metadata_track_name} - {row.master_metadata_album_artist_name}"
            ),
        }
        for row in df_tracks.itertuples()
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Tracks", className="filter-label"),
                            dcc.Dropdown(
                                id="track-filter-dropdown",
                                options=options,
                                multi=True,
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
