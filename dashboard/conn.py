import logging
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

# Only override environment variables in explicit local/test scenarios to prevent overwriting deploy-time env vars
should_override = os.environ.get("ENVIRONMENT", "") in ["local", "dev", "development", "test"]
load_dotenv(override=should_override)

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_db_path() -> str:
    """Get the path to the DuckDB database from environment or default.

    Returns:
        str: Path to the DuckDB database file.
    """
    db_path = os.getenv("MUSIC_DB", "data/db/music.db")

    # Expand environment variables and user home before checking
    expanded_path = os.path.expanduser(os.path.expandvars(db_path))

    # Check if it's an in-memory database (skip existence check for in-memory DBs)
    is_memory = (
        expanded_path == ":memory:"
        or ":memory:" in expanded_path.lower()
        or expanded_path.lower() == "memory"
    )
    if not is_memory and not Path(expanded_path).exists():
        logger.error(
            "DuckDB database file not found at resolved path '%s' (expected file-backed DuckDB database)",
            expanded_path,
        )
        raise OSError(
            f"DuckDB database file not found at resolved path '{expanded_path}' (expected file-backed DuckDB database)"
        )

    return expanded_path


def get_db_connection() -> duckdb.DuckDBPyConnection:
    """Establish a connection to the DuckDB database.

    Returns:
        duckdb.DuckDBPyConnection: Connection object to the DuckDB database.
    """
    db_path = get_db_path()
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        logger.info("Connected to DuckDB database at %s", db_path)
        return conn
    except Exception as e:
        logger.error("Failed to connect to DuckDB database: %s", e)
        raise
