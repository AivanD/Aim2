import os
import logging
from huggingface_hub import snapshot_download
from aim2.utils.config import MODELS_DIR
from aim2.utils.logging_cfg import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Models to Download ---
# Add or remove model identifiers as needed
MODELS_TO_DOWNLOAD = [
    # Main LLM for NER/RE
    "kosbu/Llama-3.3-70B-Instruct-AWQ",                 # this is the 4-bit quantized version of the 70B model, which is much smaller and faster to download than the full FP8 version
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",      # this is the full FP8 version of the 70B model, which is much larger and slower to download than the 4-bit quantized version, but may provide better performance for some tasks
    # SapBERT for entity normalization
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
]

def download_hf_model(model_id: str):
    """Downloads a Hugging Face model using snapshot_download."""
    logger.info(f"Downloading Hugging Face model: {model_id}...")
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=MODELS_DIR,
            max_workers=15, 
        )
        logger.info(f"Successfully downloaded or verified Hugging Face model: {model_id}")
    except Exception as e:
        logger.error(f"Failed to download Hugging Face model {model_id}. Error: {e}")

def main():
    """
    Main function to download all required models.
    """
    logger.info(f"Ensuring models directory exists at: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for model_name in MODELS_TO_DOWNLOAD:
        download_hf_model(model_name)

    logger.info("All model downloads attempted.")

if __name__ == "__main__":
    main()