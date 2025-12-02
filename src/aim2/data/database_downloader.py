import logging
import requests
from tqdm import tqdm

from aim2.utils.config import REFERENCE_FILES_URLS, ensure_dirs, DATA_DIR

logger = logging.getLogger(__name__)

def _download_file(url: str, dest_path: str):
    """
    Downloads a file from a URL to a specified destination path.
    """

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        progress_bar = tqdm(desc=f"downloading {dest_path.name}", total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logger.error("ERROR, something went wrong during the download")
        else:
            logger.info(f"Downloaded {dest_path} successfully.")
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}. Error: {e}")

def download_database_data():
    """
    Ensures all necessary reference files are downloaded to the reference directory.
    """
    ensure_dirs()
    
    logger.info("Checking for reference files...")
    # Check if all needed files exist
    missing_files = [filename for filename in REFERENCE_FILES_URLS if not (DATA_DIR / filename).exists()]
    if not missing_files:
        logger.info("All reference files are already present.")
        return

    # Download missing files only
    for filename in missing_files:
        url = REFERENCE_FILES_URLS[filename]
        if not url:
            logger.warning(f"No URL found for {filename}, skipping download.")
            continue
        dest_path = DATA_DIR / filename
        _download_file(url, dest_path)
    logger.info("All missing reference files have been downloaded.")
