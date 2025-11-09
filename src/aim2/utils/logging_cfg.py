import logging
import sys
from .config import LOGS_DIR # Import the logs directory path
from .config import ensure_dirs

def setup_logging(level=logging.INFO):
    """
    Configures logging for the application to output to both console and a file.
    """
    ensure_dirs()  # Ensure that the logs directory exists before creating the log file

    # Define the log file path
    log_file = LOGS_DIR / "app.log"

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs if this function is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a handler for console output (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Create a handler for file output
    file_handler = logging.FileHandler(log_file, mode='a') # 'a' for append
    file_handler.setFormatter(formatter)
    # Set the level for this handler to WARNING, so only warnings and errors are logged to the file
    file_handler.setLevel(logging.WARNING)
    root_logger.addHandler(file_handler)
    
    # Suppress httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)