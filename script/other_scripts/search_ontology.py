import logging
from pathlib import Path

# Add the project root to the Python path to allow imports from aim2

from aim2.llm.models import load_sapbert
from aim2.postprocessing.ontology_normalizer import SapbertNormalizer
from aim2.utils.logging_cfg import setup_logging

def search_ontologies():
    """
    Interactively asks for a term and searches for the best match across all
    loaded ontologies using SapBERT.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # 1. Load SapBERT model
    try:
        logger.info("Loading SapBERT model... This may take a moment.")
        sapbert_model = load_sapbert()
        logger.info("SapBERT model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SapBERT model: {e}")
        logger.error("Please ensure you have run the setup to download the model.")
        return

    # 2. Initialize the normalizer, which will load all .pkl caches
    try:
        normalizer = SapbertNormalizer(sapbert_model)
    except Exception as e:
        logger.error(f"Failed to initialize SapbertNormalizer: {e}")
        return

    if not normalizer.ontology_caches or all(v is None for v in normalizer.ontology_caches.values()):
        logger.error("No ontology caches were loaded. Please run the cache_ontology_embeddings.py script first.")
        return

    logger.info("Ready to search. Type a word or phrase and press Enter.")
    logger.info("Type 'exit' or 'quit' to end the script.")

    # 3. Start interactive search loop
    while True:
        try:
            search_term = input("\nEnter a term to search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not search_term:
            continue

        if search_term.lower() in ['exit', 'quit']:
            print("Exiting.")
            break

        print(f"\n--- Searching for '{search_term}' ---")

        # 4. Iterate through each loaded ontology and find the best match
        for ontology_key, cache in normalizer.ontology_caches.items():
            if cache is None:
                print(f"\n[{ontology_key.upper()}]")
                print("  Cache not loaded.")
                continue

            # Use a low threshold to always get the top match
            match = normalizer._find_best_match(search_term, ontology_key, threshold=0.0)

            print(f"\n[{ontology_key.upper()}]")
            if match:
                print(f"  Best Match: '{match['normalized_name']}'")
                print(f"  Ontology ID: {match['ontology_id']}")
                print(f"  Score: {match['score']:.4f}")
            else:
                print("  No match found.")
        print("\n" + "="*40)


if __name__ == "__main__":
    search_ontologies()