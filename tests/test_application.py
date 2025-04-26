import pytest
from dash import Dash
from dashboard.application import create_app

@pytest.fixture
def dash_app():
    app = create_app()
    return app

def test_app_starts(dash_duo, dash_app):
    dash_duo.start_server(dash_app)
    dash_duo.wait_for_element('.dashboard-title', timeout=10)
    assert dash_duo.find_element('.dashboard-title').text == 'Spotify Listening History'
