import logging

from aim2.data.database_builder import build_database
from aim2.data.database_downloader import _download_file, download_database_data
from aim2.utils.config import ensure_dirs
from aim2.utils.logging_cfg import setup_logging

logger = logging.getLogger(__name__)

def main():
    """
    Main function to trigger the download of database data.
    """
    setup_logging()
    ensure_dirs()

    # 1. Download data if needed
    download_database_data()

    # 2. Build the database from the downloaded files
    build_database()

if __name__ == "__main__":
    main()