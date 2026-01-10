import pandas as pd
from dash import html

MS_PER_HOUR: int = 1000 * 60 * 60


def format_stat(value: int | float) -> str:
    """Format a numeric statistic for display.

    Args:
        value (int or float): Numeric value to format.

    Returns:
        str: Number formatted with comma separators and one decimal
            place for floats.
    """
    if isinstance(value, float):
        # Format floats with one decimal and comma separators.
        return f"{value:,.1f}"
    # Format integers with comma separators.
    return f"{value:,}"


def create_stats_table(filtered_df: pd.DataFrame) -> html.Table:
    """Create a Dash HTML table summarizing listening statistics.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing listening data
            with the following columns:
            - 'ms_played': milliseconds played.
            - 'ts': timestamps of play events.
            - 'master_metadata_track_name': track names.
            - 'master_metadata_album_artist_name': artist names.

    Returns:
        html.Table: Dash HTML Table component displaying metrics and values.
    """
    # Sum total milliseconds played.
    total_ms = filtered_df["ms_played"].sum()
    # Convert milliseconds to hours.
    total_hours = total_ms / MS_PER_HOUR

    # Count unique tracks and artists.
    unique_tracks = filtered_df["master_metadata_track_name"].nunique()
    unique_artists = filtered_df["master_metadata_album_artist_name"].nunique()

    # Count unique listening days.
    unique_days = filtered_df["ts"].dt.date.nunique()

    # Calculate average daily listening time in hours.
    average_daily_hours = total_hours / unique_days if unique_days > 0 else 0.0

    # Determine the day with the most listening time.
    daily_ms_played = filtered_df.groupby(filtered_df["ts"].dt.date)["ms_played"].sum()
    most_active_day = daily_ms_played.idxmax().strftime("%Y-%m-%d")

    # Prepare statistics dictionary.
    stats = {
        "Total Listening Time": f"{format_stat(total_hours)} hours",
        "Unique Tracks": format_stat(unique_tracks),
        "Unique Artists": format_stat(unique_artists),
        "Average Daily Listening Time": f"{format_stat(average_daily_hours)} hours",
        "Most Active Day": most_active_day,
    }

    # Build Dash HTML table.
    return html.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody(
                [html.Tr([html.Td(metric), html.Td(value)]) for metric, value in stats.items()]
            ),
        ],
        className="stats-table",
    )
