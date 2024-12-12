from src.io import load_spotify_history
from src.metrics.metrics import get_most_played_tracks
from src.preprocessing import filter_songs


def main():
    for i in range(2023, 2025):
        df = load_spotify_history("listening_history")
        df_i = filter_songs(df, i, remove_incognito=False)
        top_songs_i = get_most_played_tracks(df_i)
        top_songs_i.to_excel(f"tmp/top_songs_{i}.xlsx")


if __name__ == "__main__":
    main()
