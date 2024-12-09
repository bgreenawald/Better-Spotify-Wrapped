from src.io import load_spotify_history
from src.metrics import get_most_played_tracks
from src.preprocessing import filter_songs


def main():
    df = load_spotify_history("listening_history")
    df_2024 = filter_songs(df, 2024)
    top_songs_2024 = get_most_played_tracks(df_2024)
    top_songs_2024.to_excel("top_songs_2024.xlsx")


if __name__ == "__main__":
    main()
