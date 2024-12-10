import pandas as pd
from dash import html


def format_stat(value):
    """Format statistics for display"""
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def create_stats_table(filtered_df: pd.DataFrame):
    stats = {
        "Total Listening Time": f"{filtered_df['ms_played'].sum() / (1000 * 60 * 60):.1f} hours",
        "Unique Tracks": len(filtered_df["master_metadata_track_name"].unique()),
        "Unique Artists": len(
            filtered_df["master_metadata_album_artist_name"].unique()
        ),
        "Average Daily Listening Time": f"{(filtered_df['ms_played'].sum() / (1000 * 60 * 60)) / len(filtered_df['ts'].dt.date.unique()):.1f} hours",
        "Most Active Day": filtered_df.groupby(filtered_df["ts"].dt.date)["ms_played"]
        .sum()
        .idxmax()
        .strftime("%Y-%m-%d"),
    }

    return html.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
            html.Tbody(
                [
                    html.Tr([html.Td(metric), html.Td(value)])
                    for metric, value in stats.items()
                ]
            ),
        ],
        className="stats-table",
    )
