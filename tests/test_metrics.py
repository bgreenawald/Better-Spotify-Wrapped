import pandas as pd
import pytest
from src.metrics import metrics

@pytest.fixture
def sample_df():
    data = {
        'master_metadata_track_name': ['Song A', 'Song B', 'Song A', 'Song C'],
        'master_metadata_album_artist_name': ['Artist 1', 'Artist 2', 'Artist 1', 'Artist 3'],
        'spotify_track_uri': ['uri:1', 'uri:2', 'uri:1', 'uri:3'],
        'artist_id': ['id1', 'id2', 'id1', 'id3'],
        'artist_genres': [('pop',), ('rock',), ('pop',), ('jazz',)],
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_spotify_data():
    # Minimal mock for the SpotifyData object used in metrics.py
    class MockSpotifyData:
        def __init__(self):
            self.tracks = {
                '1': {'album': {'id': 'a1'}, 'track_number': 1, 'artists': [{'id': 'id1'}, {'id': 'id2'}]},
                '2': {'album': {'id': 'a2'}, 'track_number': 1, 'artists': [{'id': 'id2'}]},
                '3': {'album': {'id': 'a3'}, 'track_number': 1, 'artists': [{'id': 'id3'}]},
            }
            self.albums = {
                'a1': {'album_type': 'album', 'total_tracks': 1, 'name': 'Album 1', 'artists': [{'name': 'Artist 1'}], 'release_date': '2020-01-01'},
                'a2': {'album_type': 'album', 'total_tracks': 1, 'name': 'Album 2', 'artists': [{'name': 'Artist 2'}], 'release_date': '2021-01-01'},
                'a3': {'album_type': 'album', 'total_tracks': 1, 'name': 'Album 3', 'artists': [{'name': 'Artist 3'}], 'release_date': '2022-01-01'},
            }
            self.artists = {
                'id1': {'name': 'Artist 1', 'genres': ['pop']},
                'id2': {'name': 'Artist 2', 'genres': ['rock']},
                'id3': {'name': 'Artist 3', 'genres': ['jazz']},
            }
    return MockSpotifyData()

def test_get_most_played_tracks(sample_df):
    result = metrics.get_most_played_tracks(sample_df)
    assert not result.empty
    assert 'track_name' in result.columns
    assert 'artist' in result.columns
    assert 'play_count' in result.columns
    assert result.iloc[0]['play_count'] >= result.iloc[-1]['play_count']

def test_get_top_albums(sample_df, mock_spotify_data):
    # Add matching track IDs to sample_df for compatibility
    sample_df = sample_df.copy()
    sample_df['spotify_track_uri'] = ['spotify:track:1', 'spotify:track:2', 'spotify:track:1', 'spotify:track:3']
    result = metrics.get_top_albums(sample_df, mock_spotify_data)
    assert isinstance(result, pd.DataFrame)
    assert 'album_name' in result.columns
    assert 'median_plays' in result.columns
    assert result['median_plays'].min() >= 0

def test_get_top_artist_genres(sample_df, mock_spotify_data):
    # Add matching track IDs to sample_df for compatibility
    sample_df = sample_df.copy()
    sample_df['spotify_track_uri'] = ['spotify:track:1', 'spotify:track:2', 'spotify:track:1', 'spotify:track:3']
    result = metrics.get_top_artist_genres(sample_df, mock_spotify_data)
    assert isinstance(result, pd.DataFrame)
    assert 'genre' in result.columns
    assert 'play_count' in result.columns
    assert result['play_count'].min() >= 0

def test_get_most_played_artists(sample_df):
    result = metrics.get_most_played_artists(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert 'artist' in result.columns
    assert 'play_count' in result.columns
    assert 'unique_tracks' in result.columns
    assert result['play_count'].min() >= 0
