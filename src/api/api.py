import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

load_dotenv(override=True)


@dataclass
class SpotifyData:
    """Container for all Spotify-related data"""

    tracks: Dict[str, Dict]  # track_id -> track_data
    artists: Dict[str, Dict]  # artist_id -> artist_data
    albums: Dict[str, Dict]  # album_id -> album_data


class SpotifyDataCollector:
    def __init__(
        self,
        client_id: str = os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret: str = os.getenv("SPOTIFY_CLIENT_SECRET"),
    ):
        """
        Initialize the Spotify data collector with API credentials and cache directory.

        Args:
            client_id: Spotify API client ID
            client_secret: Spotify API client secret
        """
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )
        )

        self.data_dir = Path(os.getenv("DATA_DIR"))
        self.api_cache_dir = Path("data/api/cache")
        self._setup_cache_directories()

        # API batch limits
        self.TRACK_BATCH_SIZE = 50
        self.ALBUM_BATCH_SIZE = 20
        self.ARTIST_BATCH_SIZE = 20

    def _setup_cache_directories(self):
        """Create cache directories if they don't exist."""
        for subdir in ["tracks", "artists", "albums"]:
            (self.api_cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _get_cached_ids(self, cache_type: str) -> Set[str]:
        """Get set of all cached IDs for a given type."""
        cache_path = self.api_cache_dir / cache_type
        return {f.stem for f in cache_path.glob("*.json")}

    def _load_cache(self, cache_type: str, item_id: str) -> Dict:
        """Load item from cache."""
        cache_path = self.api_cache_dir / cache_type / f"{item_id}.json"
        with open(cache_path, "r") as f:
            return json.load(f)

    def _save_cache(self, cache_type: str, items: List[Dict]):
        """Save multiple items to cache."""
        for item in items:
            item_id = item["id"]
            cache_path = self.api_cache_dir / cache_type / f"{item_id}.json"
            with open(cache_path, "w") as f:
                json.dump(item, f, indent=4, sort_keys=True)

    def _chunk_list(self, lst: List[str], chunk_size: int) -> List[List[str]]:
        """Split list into chunks of specified size."""
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def fetch_tracks(self, track_ids: List[str]) -> List[Dict]:
        """
        Fetch track information for multiple tracks, using cache when available.
        Only fetches uncached tracks from the API.
        """
        cached_ids = self._get_cached_ids("tracks")
        uncached_ids = [tid for tid in track_ids if tid not in cached_ids]

        # Load cached tracks
        tracks = [
            self._load_cache("tracks", tid) for tid in track_ids if tid in cached_ids
        ]

        # Fetch uncached tracks in batches
        for id_batch in tqdm(self._chunk_list(uncached_ids, self.TRACK_BATCH_SIZE)):
            batch_tracks = self.sp.tracks(id_batch)["tracks"]
            self._save_cache("tracks", batch_tracks)
            tracks.extend(batch_tracks)
            time.sleep(0.2)  # Rate limiting

        return tracks

    def fetch_artists(self, artist_ids: List[str]) -> List[Dict]:
        """
        Fetch artist information for multiple artists, using cache when available.
        Only fetches uncached artists from the API.
        """
        cached_ids = self._get_cached_ids("artists")
        uncached_ids = [aid for aid in artist_ids if aid not in cached_ids]

        # Load cached artists
        artists = [
            self._load_cache("artists", aid) for aid in artist_ids if aid in cached_ids
        ]

        # Fetch uncached artists in batches
        for id_batch in tqdm(self._chunk_list(uncached_ids, self.ARTIST_BATCH_SIZE)):
            batch_artists = self.sp.artists(id_batch)["artists"]
            self._save_cache("artists", batch_artists)
            artists.extend(batch_artists)
            time.sleep(0.2)

        return artists

    def fetch_albums(self, album_ids: List[str]) -> List[Dict]:
        """
        Fetch album information for multiple albums, using cache when available.
        Only fetches uncached albums from the API.
        """
        cached_ids = self._get_cached_ids("albums")
        uncached_ids = [aid for aid in album_ids if aid not in cached_ids]

        # Load cached albums
        albums = [
            self._load_cache("albums", aid) for aid in album_ids if aid in cached_ids
        ]

        # Fetch uncached albums in batches
        for id_batch in tqdm(self._chunk_list(uncached_ids, self.ALBUM_BATCH_SIZE)):
            batch_albums = self.sp.albums(id_batch)["albums"]
            self._save_cache("albums", batch_albums)
            albums.extend(batch_albums)
            time.sleep(0.2)

        return albums


def _list_to_dict(lst: List[Dict], key: str) -> Dict[str, Dict]:
    return {item[key]: item for item in lst}


def load_api_data() -> SpotifyData:
    collector = SpotifyDataCollector()
    with open(collector.data_dir / "track_ids.txt", "r") as f:
        track_ids = [line.strip() for line in f]
    with open(collector.data_dir / "artist_ids.txt", "r") as f:
        artist_ids = [line.strip() for line in f]
    with open(collector.data_dir / "album_ids.txt", "r") as f:
        album_ids = [line.strip() for line in f]

    tracks = collector.fetch_tracks(track_ids)
    artists = collector.fetch_artists(artist_ids)
    albums = collector.fetch_albums(album_ids)

    return SpotifyData(
        _list_to_dict(tracks, "id"),
        _list_to_dict(artists, "id"),
        _list_to_dict(albums, "id"),
    )


# Example usage
if __name__ == "__main__":
    # Load credentials from environment variables
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    collector = SpotifyDataCollector(client_id, client_secret)

    # Example: Load IDs from files
    with open(collector.data_dir / "track_ids.txt", "r") as f:
        track_ids = [line.strip() for line in f]
    with open(collector.data_dir / "artist_ids.txt", "r") as f:
        artist_ids = [line.strip() for line in f]
    with open(collector.data_dir / "album_ids.txt", "r") as f:
        album_ids = [line.strip() for line in f]

    # Fetch data
    print("Fetching tracks...")
    tracks = collector.fetch_tracks(track_ids)
    print("Fetching artists...")
    artists = collector.fetch_artists(artist_ids)
    print("Fetching albums...")
    albums = collector.fetch_albums(album_ids)

    # Convert to DataFrames if needed
    tracks_df = pd.DataFrame(tracks)
    tracks_df.to_csv(collector.data_dir / "tracks.csv", index=False)

    artists_df = pd.DataFrame(artists)
    artists_df.to_csv(collector.data_dir / "artists.csv", index=False)

    albums_df = pd.DataFrame(albums)
    albums_df.to_csv(collector.data_dir / "albums.csv", index=False)
