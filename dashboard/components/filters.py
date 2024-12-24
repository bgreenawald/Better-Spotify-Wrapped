from datetime import datetime

import pandas as pd
from dash import dcc, html

from src.metrics.trends import (
    get_genre_trends,
)


def create_year_dropdown(df: pd.DataFrame):
    available_years = sorted(df["ts"].dt.year.unique())
    return html.Div(
        [
            html.Label("Select Year", className="filter-label"),
            dcc.Dropdown(
                id="year-dropdown",
                options=[
                    {"label": str(year), "value": year} for year in available_years
                ],
                value=available_years[-1],
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_wrapped_filters_section(df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div(
                [
                    create_year_dropdown(df),
                ],
                className="filters-section",
            ),
        ]
    )


def create_global_settings(df: pd.DataFrame):
    tracks = df["master_metadata_track_name"].dropna().unique().tolist()
    artists = df["master_metadata_album_artist_name"].dropna().unique().tolist()
    albums = df["master_metadata_album_album_name"].dropna().unique().tolist()
    return html.Div(
        [
            html.Div(
                [
                    # Genre Filter
                    html.Div(
                        [
                            html.Label(
                                "Select Excluded Artists", className="filter-label"
                            ),
                            dcc.Dropdown(
                                id="excluded-artists-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {"label": artist, "value": artist}
                                    for artist in artists
                                ],
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Select Excluded Albums", className="filter-label"
                            ),
                            dcc.Dropdown(
                                id="excluded-albums-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {"label": album, "value": album} for album in albums
                                ],
                            ),
                        ],
                        className="filter-item",
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Select Excluded Tracks", className="filter-label"
                            ),
                            dcc.Dropdown(
                                id="excluded-tracks-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {"label": track, "value": track} for track in tracks
                                ],
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
    min_date = datetime(df["ts"].min().year, df["ts"].min().month, df["ts"].min().day)
    max_date = datetime(df["ts"].max().year, df["ts"].max().month, df["ts"].max().day)

    return html.Div(
        [
            html.Label("Select Date Range", className="filter-label"),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
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


def create_genre_trends_layout(df, spotify_data):
    genres_df = get_genre_trends(df, spotify_data)
    genres = sorted(genres_df["genre"].dropna().unique().tolist())
    """Create the layout for the genre trends analysis section"""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Genres", className="filter-label"),
                            dcc.Dropdown(
                                id="genre-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {"label": genre.title(), "value": genre}
                                    for genre in genres
                                ],
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            # Genre Filter
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Number of Top Genres", className="filter-label"
                            ),
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
                    # Display Type Selector
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
            # Top N Genres Slider
        ],
    )


def create_artist_trends_layout(df: pd.DataFrame):
    artists = sorted(df["master_metadata_album_artist_name"].dropna().unique().tolist())
    """Create the layout for the genre trends analysis section"""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Artists", className="filter-label"),
                            dcc.Dropdown(
                                id="artist-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {"label": artist, "value": artist}
                                    for artist in artists
                                ],
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            # Genre Filter
            # Top N Genres Slider
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Number of Top Artists", className="filter-label"
                            ),
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
                    # Display Type Selector
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
        ],
    )


def create_track_trends_layout(df: pd.DataFrame):
    """Create the layout for the track trends analysis section"""
    df_tracks = df.copy()
    df_tracks = df_tracks.drop_duplicates(
        subset=["master_metadata_track_name", "master_metadata_album_artist_name"]
    )
    df_tracks = df_tracks.dropna(
        subset=["master_metadata_track_name", "master_metadata_album_artist_name"]
    )
    df_tracks = df_tracks.sort_values(by="master_metadata_track_name", ascending=True)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Tracks", className="filter-label"),
                            dcc.Dropdown(
                                id="track-filter-dropdown",
                                multi=True,
                                className="dropdown",
                                options=[
                                    {
                                        "label": row.master_metadata_track_name
                                        + " - "
                                        + row.master_metadata_album_artist_name,
                                        "value": row.master_metadata_track_name
                                        + " - "
                                        + row.master_metadata_album_artist_name,
                                    }
                                    for row in df_tracks.itertuples()
                                ],
                            ),
                        ],
                        className="filter-item",
                    ),
                ],
                className="filters-section",
            ),
            # Track Filter
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Number of Top Tracks", className="filter-label"
                            ),
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
                    # Display Type Selector
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
            # Top N Tracks Slider
        ],
    )


def create_monthly_trend_filter() -> html.Div:
    return html.Div(
        [
            html.Label("Select Metric", className="filter-label"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "Total Hours", "value": "total_hours"},
                    {"label": "Unique Tracks", "value": "unique_tracks"},
                    {"label": "Unique Artists", "value": "unique_artists"},
                    {"label": "Average Hours per Day", "value": "avg_hours_per_day"},
                ],
                value="total_hours",
                className="dropdown",
            ),
        ],
        className="filter-item",
    )


def create_trend_filters_section(df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Filters", className="card-title"),
            html.Div(
                [
                    create_year_range_filter(df),
                ],
                className="filters-section",
            ),
        ]
    )
