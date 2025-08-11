import logging
import sys

def setup_logging(level=logging.INFO):
    """
    Configures logging for the application.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    
    # Suppress httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
