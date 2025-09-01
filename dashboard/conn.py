import logging
import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_db_path() -> str:
    """Get the path to the DuckDB database from environment or default.

    Returns:
        str: Path to the DuckDB database file.
    """
    db_path = os.getenv("MUSIC_DB", "data/db/music.db")

    # Resolve database path (preloaded); default aligns with CLI
    if not Path(db_path).exists():
        logger.error("DuckDB database not found at %s", db_path)
        raise OSError(f"DuckDB database not found at {db_path}")

    return db_path


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
