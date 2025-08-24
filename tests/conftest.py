import os

import pytest
from webdriver_manager.chrome import ChromeDriverManager


@pytest.fixture(autouse=True, scope="session")
def setup_chrome_environment(tmp_path_factory):
    """Set up Chrome profile and ChromeDriver for test session.

    This fixture automatically runs for all tests and ensures that:
    1. Each test session uses a unique Chrome profile directory
    2. The correct ChromeDriver version is available
    """
    # Create isolated Chrome profile
    chrome_data_dir = tmp_path_factory.mktemp("chrome_profile")
    os.environ["CHROME_USER_DATA_DIR"] = str(chrome_data_dir)

    # Ensure correct ChromeDriver is available
    chromedriver_path = ChromeDriverManager().install()
    os.environ["CHROMEDRIVER_PATH"] = chromedriver_path

    # Add to PATH for Selenium to find it
    path_dir = os.path.dirname(chromedriver_path)
    current_path = os.environ.get("PATH", "")
    if path_dir not in current_path:
        os.environ["PATH"] = f"{path_dir}:{current_path}"

    return chrome_data_dir
