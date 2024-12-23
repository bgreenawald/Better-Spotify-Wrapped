from datetime import datetime

import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dash_table

from dashboard.components.graphs import create_graph_style
from dashboard.components.stats import create_stats_table
from src.metrics.metrics import (
    get_most_played_artists,
    get_most_played_tracks,
    get_top_albums,
    get_top_artist_genres,
)
from src.metrics.trends import (
    get_artist_trends,
    get_genre_trends,
    get_listening_time_by_month,
    get_track_trends,
)
from src.preprocessing import filter_songs


def register_callbacks(app, df: pd.DataFrame, spotify_data):
    @app.callback(
        Output("collapse", "is_open"),
        [Input("collapse-button", "n_clicks")],
        [State("collapse", "is_open")],
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open

    @app.callback(
        [
            Output("top-tracks-graph", "figure"),
            Output("top-artists-graph", "figure"),
            Output("top-albums-graph", "figure"),
            Output("top-genres-graph", "figure"),
            Output("detailed-stats", "children"),
        ],
        [
            Input("year-dropdown", "value"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_dashboard(
        selected_year,
        exclude_december,
        remove_incognito,
        exluded_tracks,
        excluded_artists,
        excluded_albums,
    ):
        # Check if a valid year is selected
        if not selected_year:
            # Return empty figures if no year is selected
            empty_figure = {"data": [], "layout": create_graph_style()}
            return empty_figure, empty_figure, empty_figure, empty_figure, []

        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.Timestamp(datetime(selected_year, 1, 1)),
            end_date=pd.Timestamp(datetime(selected_year, 12, 31)),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=exluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )

        # Get top tracks
        top_tracks = get_most_played_tracks(filtered_df)
        if top_tracks.empty:
            tracks_fig = {"data": [], "layout": create_graph_style()}
        else:
            tracks_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_tracks["play_count"].head(10),
                        "y": top_tracks["track_name"].head(10),
                        "orientation": "h",
                        "text": top_tracks["play_count"].head(10),
                        "marker": {"color": "#1DB954"},
                    }
                ],
                "layout": {
                    **create_graph_style(),
                    "xaxis": {"gridcolor": "#eee", "title": "Play Count"},
                    "yaxis": {"gridcolor": "#eee", "title": ""},
                },
            }

        # Get top artists
        top_artists = get_most_played_artists(filtered_df)
        if top_artists.empty:
            artists_fig = {"data": [], "layout": create_graph_style()}
        else:
            artists_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_artists["play_count"].head(10),
                        "y": top_artists["artist"].head(10),
                        "orientation": "h",
                        "text": top_artists["play_count"].head(10),
                        "customdata": top_artists["unique_tracks"].head(10),
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": "Artist: %{y}<br>Plays: %{x}<br>Unique Tracks: %{customdata}<extra></extra>",
                    }
                ],
                "layout": {
                    **create_graph_style(),
                    "xaxis": {"gridcolor": "#eee", "title": "Play Count"},
                    "yaxis": {"gridcolor": "#eee", "title": ""},
                },
            }

        # Get top albums
        top_albums = get_top_albums(filtered_df, spotify_data)
        if top_albums.empty:
            albums_fig = {"data": [], "layout": create_graph_style()}
        else:
            albums_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_albums["median_plays"].head(10),
                        "y": top_albums["album_name"].head(10),
                        "orientation": "h",
                        "text": top_albums["median_plays"].round(1).head(10),
                        "customdata": top_albums[
                            ["artist", "tracks_played", "total_tracks"]
                        ]
                        .head(10)
                        .values,
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": "Album: %{y}<br>Median Plays: %{x}<br>Artist: %{customdata[0]}<br>Tracks Played: %{customdata[1]}/{customdata[2]}<extra></extra>",
                    }
                ],
                "layout": {
                    **create_graph_style(),
                    "xaxis": {"gridcolor": "#eee", "title": "Median Plays"},
                    "yaxis": {"gridcolor": "#eee", "title": ""},
                },
            }

        # Get top genres
        top_genres = get_top_artist_genres(filtered_df, spotify_data)
        if top_genres.empty:
            genres_fig = {"data": [], "layout": create_graph_style()}
        else:
            genres_fig = {
                "data": [
                    {
                        "type": "bar",
                        "x": top_genres["play_count"].head(10),
                        "y": top_genres["genre"].head(10),
                        "orientation": "h",
                        "text": [
                            f"{x:.1f}%" for x in top_genres["percentage"].head(10)
                        ],
                        "customdata": top_genres[["percentage", "top_artists"]]
                        .head(10)
                        .values,
                        "marker": {"color": "#1DB954"},
                        "hovertemplate": "Genre: %{y}<br>Tracks: %{x}<br>Percentage: %{customdata[0]:.1f}%<br>Top Artists: %{customdata[1]}<extra></extra>",
                    }
                ],
                "layout": {
                    **create_graph_style(),
                    "xaxis": {"gridcolor": "#eee", "title": "Track Count"},
                    "yaxis": {"gridcolor": "#eee", "title": ""},
                },
            }

        # Create stats table
        stats_table = create_stats_table(filtered_df)

        return tracks_fig, artists_fig, albums_fig, genres_fig, stats_table

    @app.callback(
        Output("trends-graph", "figure"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("metric-dropdown", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_trend_dashboard(
        start_date,
        end_date,
        exclude_december,
        remove_incognito,
        selected_metric,
        excluded_tracks,
        excluded_artists,
        excluded_albums,
    ):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )

        # Calculate monthly statistics
        monthly_stats = get_listening_time_by_month(filtered_df)

        # Create the figure using plotly express
        metric_labels = {
            "total_hours": "Total Listening Hours",
            "unique_tracks": "Unique Tracks",
            "unique_artists": "Unique Artists",
            "avg_hours_per_day": "Average Hours per Day",
        }

        fig = px.line(
            monthly_stats,
            x="month",
            y=selected_metric,
            labels={"month": "Month", selected_metric: metric_labels[selected_metric]},
            title=f"Monthly {metric_labels[selected_metric]}",
        )

        # Update layout
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, sans-serif"},
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            xaxis={"gridcolor": "#eee"},
            yaxis={"gridcolor": "#eee"},
            showlegend=False,
        )

        # Update line color to match Spotify theme
        fig.update_traces(line_color="#1DB954")

        return fig

    @app.callback(
        Output("genre-filter-dropdown", "options"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def update_genre_options(start_date, end_date, exclude_december, remove_incognito):
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )
        trends_df = get_genre_trends(filtered_df, spotify_data)
        genres = trends_df["genre"].unique()
        return [{"label": genre.title(), "value": genre} for genre in genres]

    @app.callback(
        Output("artist-filter-dropdown", "options"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def update_artist_options(start_date, end_date, exclude_december, remove_incognito):
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )
        trends_df = get_artist_trends(filtered_df)
        artists = trends_df["artist"].unique()
        return [{"label": artist.title(), "value": artist} for artist in artists]

    @app.callback(
        Output("track-filter-dropdown", "options"),
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
        ],
    )
    def update_track_options(start_date, end_date, exclude_december, remove_incognito):
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
        )
        trends_df = get_track_trends(filtered_df)
        tracks = trends_df["track"].unique()
        return [
            {
                "label": f"{track} - {trends_df[trends_df['track'] == track]['artist'].iloc[0]}",
                "value": track,
            }
            for track in tracks
        ]

    # Callback to update the graph
    @app.callback(
        [
            Output("genre-trends-graph", "figure"),
            Output("genre-trends-table", "children"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("genre-filter-dropdown", "value"),
            Input("top-genres-slider", "value"),
            Input("genre-display-type-radio", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_genre_trends_graph(
        start_date,
        end_date,
        exclude_december,
        remove_incognito,
        selected_genres,
        top_n,
        display_type,
        excluded_tracks,
        excluded_artists,
        excluded_albums,
    ):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )
        trends_df = get_genre_trends(filtered_df, spotify_data)

        if display_type == "percentage":
            y_column = "percentage"
            y_title = "Percentage of Tracks"
        else:
            y_column = "play_count"
            y_title = "Number of Plays"

        # Overall trends
        overall_trends = get_top_artist_genres(filtered_df, spotify_data)

        if not selected_genres:
            # If no genres selected, show top N genres by average percentage
            avg_by_genre = (
                overall_trends.groupby("genre")[y_column]
                .mean()
                .sort_values(ascending=False)
            )
            selected_genres = avg_by_genre.head(top_n).index.tolist()

        # Filter for selected genres
        plot_df = trends_df[trends_df["genre"].isin(selected_genres)]

        # Create line plot
        fig = px.line(
            plot_df,
            x="month",
            y=y_column,
            color="genre",
            labels={
                "month": "Month",
                y_column: y_title,
                "genre": "Genre",
                "top_artists": "Top Artists",
            },
            hover_data=["top_artists", y_column, "month", "genre"],
            title="Genre Trends Over Time",
        )

        # Update layout
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, sans-serif"},
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            xaxis={"gridcolor": "#eee"},
            yaxis={"gridcolor": "#eee"},
        )

        return fig, [
            dash_table.DataTable(
                overall_trends.to_dict("records"),
                [{"name": i, "id": i} for i in overall_trends.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                page_size=25,
            )
        ]

    @app.callback(
        [
            Output("artist-trends-graph", "figure"),
            Output("artist-trends-table", "children"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("artist-filter-dropdown", "value"),
            Input("top-artist-slider", "value"),
            Input("artist-display-type-radio", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_artist_trends_graph(
        start_date,
        end_date,
        exclude_december,
        remove_incognito,
        selected_artists,
        top_n,
        display_type,
        excluded_tracks,
        excluded_artists,
        excluded_albums,
    ):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )
        trends_df = get_artist_trends(filtered_df)

        if display_type == "percentage":
            y_column = "percentage"
            y_title = "Percentage of Tracks"
        elif display_type == "play_count":
            y_column = "play_count"
            y_title = "Number of Plays"
        else:
            y_column = "unique_tracks"
            y_title = "Unique Tracks"

        # Overall artists
        overall_artists = get_most_played_artists(filtered_df)

        if not selected_artists:
            # If no artists selected, show top N artists by overall
            avg_by_artist = (
                overall_artists.groupby("artist")[y_column]
                .mean()
                .sort_values(ascending=False)
            )
            selected_artists = avg_by_artist.head(top_n).index.tolist()

        # Filter for selected artists
        plot_df = trends_df[trends_df["artist"].isin(selected_artists)]

        # Create line plot
        fig = px.line(
            plot_df,
            x="month",
            y=y_column,
            color="artist",
            labels={
                "month": "Month",
                y_column: y_title,
                "artist": "Artist",
                "top_artists": "Top Artists",
            },
            hover_data=["top_tracks", y_column, "month", "artist"],
            title="Artist Trends Over Time",
        )

        # Update layout
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, sans-serif"},
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            xaxis={"gridcolor": "#eee"},
            yaxis={"gridcolor": "#eee"},
        )

        return fig, [
            dash_table.DataTable(
                overall_artists.to_dict("records"),
                [{"name": i, "id": i} for i in overall_artists.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                page_size=25,
            )
        ]

    @app.callback(
        [
            Output("track-trends-graph", "figure"),
            Output("track-trends-table", "children"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("exclude-december", "value"),
            Input("remove-incognito", "value"),
            Input("track-filter-dropdown", "value"),
            Input("top-track-slider", "value"),
            Input("track-display-type-radio", "value"),
            Input("excluded-tracks-filter-dropdown", "value"),
            Input("excluded-artists-filter-dropdown", "value"),
            Input("excluded-albums-filter-dropdown", "value"),
        ],
    )
    def update_track_trends_graph(
        start_date,
        end_date,
        exclude_december,
        remove_incognito,
        selected_tracks,
        top_n,
        display_type,
        excluded_tracks,
        excluded_artists,
        excluded_albums,
    ):
        # Filter the data based on selections
        filtered_df = filter_songs(
            df,
            start_date=pd.to_datetime(start_date),
            end_date=pd.to_datetime(end_date),
            exclude_december=exclude_december,
            remove_incognito=remove_incognito,
            excluded_tracks=excluded_tracks,
            excluded_artists=excluded_artists,
            excluded_albums=excluded_albums,
        )
        trends_df = get_track_trends(filtered_df)

        if display_type == "percentage":
            y_column = "percentage"
            y_title = "Percentage of Plays"
        else:
            y_column = "play_count"
            y_title = "Number of Plays"

        # Overall tracks
        overall_tracks = get_most_played_tracks(filtered_df)

        if not selected_tracks:
            # If no tracks selected, show top N tracks by overall
            avg_by_track = (
                overall_tracks.groupby("track_name")[y_column]
                .mean()
                .sort_values(ascending=False)
            )
            selected_tracks = avg_by_track.head(top_n).index.tolist()

        # Filter for selected tracks
        plot_df = trends_df[trends_df["track"].isin(selected_tracks)]

        # Create line plot
        fig = px.line(
            plot_df,
            x="month",
            y=y_column,
            color="track",
            labels={
                "month": "Month",
                y_column: y_title,
                "track": "Track",
                "artist": "Artist",
            },
            hover_data=["artist", y_column, "month", "track"],
            title="Track Trends Over Time",
        )

        # Update layout
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Segoe UI, sans-serif"},
            margin={"t": 50, "b": 30, "l": 30, "r": 30},
            xaxis={"gridcolor": "#eee"},
            yaxis={"gridcolor": "#eee"},
        )

        return fig, [
            dash_table.DataTable(
                overall_tracks.to_dict("records"),
                [{"name": i, "id": i} for i in overall_tracks.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                page_size=25,
            )
        ]
