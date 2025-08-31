import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def setup_chrome_environment(tmp_path_factory):
    """Set up Chrome profile and ChromeDriver for test session.

    This fixture automatically runs for all tests and ensures that:
    1. Each test session uses a unique Chrome profile directory
    2. The correct ChromeDriver version is available (when not in CI)
    """
    # Check if running in CI environment and skip Chrome setup if so
    if os.getenv("CI"):
        pytest.skip("Skipping Chrome setup in CI environment")

    # Create isolated Chrome profile
    chrome_data_dir = tmp_path_factory.mktemp("chrome_profile")
    os.environ["CHROME_USER_DATA_DIR"] = str(chrome_data_dir)

    # Attempt ChromeDriver management if webdriver_manager is available
    try:
        from webdriver_manager.chrome import ChromeDriverManager

        # Install and configure ChromeDriver
        chromedriver_path = ChromeDriverManager().install()
        os.environ["CHROMEDRIVER_PATH"] = chromedriver_path

        # Add to PATH for Selenium to find it
        path_dir = os.path.dirname(chromedriver_path)
        current_path = os.environ.get("PATH", "")
        if path_dir not in current_path:
            os.environ["PATH"] = f"{path_dir}:{current_path}"
    except ImportError:
        # webdriver_manager not available, let Selenium Manager handle driver
        pass

    return chrome_data_dir
