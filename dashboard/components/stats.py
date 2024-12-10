import pandas as pd
from dash import html


def create_stats_table(filtered_df: pd.DataFrame):
    return html.Div(
        [
            html.H3("Detailed Statistics"),
            html.Table(
                [
                    html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td("Total Listening Time"),
                                    html.Td(
                                        f"{filtered_df['ms_played'].sum() / (1000 * 60 * 60):.1f} hours"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Unique Tracks"),
                                    html.Td(
                                        len(
                                            filtered_df[
                                                "master_metadata_track_name"
                                            ].unique()
                                        )
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Unique Artists"),
                                    html.Td(
                                        len(
                                            filtered_df[
                                                "master_metadata_album_artist_name"
                                            ].unique()
                                        )
                                    ),
                                ]
                            ),
                        ]
                    ),
                ],
                style={"width": "100%", "textAlign": "left"},
            ),
        ],
        style={"marginTop": "30px"},
    )
