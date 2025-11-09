import logging
import time
from typing import List, Dict, Any
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import re

from aim2.utils.config import NCBI_API_KEY

logger = logging.getLogger(__name__)

# New Base URL for NCBI Datasets API
DATASETS_BASE_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2"

def normalize_species_with_ncbi(processed_results: List[Dict[str, Any]], MAX_ATTEMPTS=5) -> List[Dict[str, Any]]:
    """
    Normalizes species names using the NCBI Datasets API to fetch a Taxonomy ID and canonical name.
    It resolves abbreviated genus names within the document context before querying and caches results.
    """
    logger.info("Starting species normalization with NCBI Datasets API...")

    # In-memory cache to avoid redundant API calls
    species_cache = {}

    # Prepare request headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if NCBI_API_KEY:
        headers["api-key"] = NCBI_API_KEY
        logger.info("Using NCBI API Key for normalization.")
        # With an API key, we can go faster (up to 10 rps)
        request_delay = 0.1
    else:
        logger.warning("NCBI_API_KEY not found. Rate limiting to 3 rps.")
        # Without an API key, be respectful (max 3 rps)
        request_delay = 0.4

    # --- Pass 1: Build an abbreviation map from the entire document ---
    abbreviation_map = {}
    all_species_entities = [
        entity for result in processed_results for entity in result.get("species", [])
    ]

    # Find all full names first (e.g., "Beta vulgaris")
    full_names = {
        entity['name'] for entity in all_species_entities 
        if re.match(r'^[A-Z][a-z]+ [a-z]+$', entity['name'])
    }

    # Create a map from abbreviations to full names
    for full_name in full_names:
        parts = full_name.split()
        if len(parts) == 2:
            genus, species_epithet = parts
            abbreviation = f"{genus[0]}. {species_epithet}"
            abbreviation_map[abbreviation.lower()] = full_name

    # --- Pass 2: Normalize entities using the map ---
    for result in processed_results:
        if "species" not in result or not result["species"]:
            continue

        for species_entity in result["species"]:
            original_name = species_entity.get("name")
            if not original_name:
                continue

            # Skip if already normalized
            if species_entity.get("taxonomy_id"):
                continue

            # Use the full name from our map if the original name is an abbreviation
            name_to_query = abbreviation_map.get(original_name.lower(), original_name)

            # Check cache first
            if name_to_query in species_cache:
                cached_result = species_cache[name_to_query]
                if cached_result:
                    species_entity.update(cached_result)
                    logger.debug(f"Cache hit for '{name_to_query}'.")
                continue

            for attempt in range(MAX_ATTEMPTS):
                try:
                    # Use the new NCBI Datasets API endpoint
                    search_url = f"{DATASETS_BASE_URL}/taxonomy/taxon_suggest/{urllib.parse.quote(name_to_query)}"
                    search_response = requests.get(search_url, headers=headers)
                    search_response.raise_for_status()

                    # --- Proactive Rate Limit Check on first successful call ---
                    
                    # The header is 'X-RateLimit-Limit' for v2 API
                    limit_header = search_response.headers.get('X-RateLimit-Limit')
                    # Unauthenticated is 3 or 5, authenticated is 10. Check if it's low.
                    if limit_header and int(limit_header) <= 5:
                        logger.warning("Low rate limit detected. The provided NCBI API key may be invalid. "
                                        "Falling back to unauthenticated rate (~3 rps).")
                        headers.pop("api-key", None)
                        request_delay = 0.4 # Slow down for all subsequent requests

                    search_data = search_response.json()

                    suggestions = search_data.get("sci_name_and_ids", [])
                    
                    if suggestions:
                        top_suggestion = suggestions[0]
                        tax_id = top_suggestion.get("tax_id")
                        
                        if tax_id:
                            update_data = {
                                "taxonomy_id": int(tax_id),
                            }
                            species_entity.update(update_data)
                            species_cache[name_to_query] = update_data # Cache success
                            logger.debug(f"Normalized '{original_name}' -> '{name_to_query}' (TaxID: {tax_id})")
                        else:
                            logger.warning(f"Found suggestion for '{name_to_query}' but it lacked a tax_id.")
                            species_cache[name_to_query] = None # Cache failure
                    else:
                        logger.warning(f"Could not find NCBI Taxonomy entry for species: '{name_to_query}'")
                        species_cache[name_to_query] = None # Cache failure

                    # Be respectful to the API
                    time.sleep(request_delay)
                    break # Success or non-retriable failure, exit retry loop

                except requests.exceptions.HTTPError as http_err:
                    # Handle invalid API key specifically
                    if  http_err.response.status_code in [400, 401, 403]:
                        logger.error(f"NCBI API key is invalid or unauthorized. Status: {http_err.response.status_code}. "
                                     "Falling back to unauthenticated requests at a slower rate.")
                        headers.pop("api-key", None)
                        request_delay = 0.4 # Slow down for unauthenticated access
                        continue # Retry the same request immediately without the key
                    elif http_err.response.status_code == 404:
                        logger.warning(f"Could not find NCBI entry for species: '{name_to_query}' (404 Not Found)")
                        species_cache[name_to_query] = None # Cache failure
                        break
                    elif http_err.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_ATTEMPTS - 1:
                        logger.warning(f"Server error for '{name_to_query}' (attempt {attempt + 1}/{MAX_ATTEMPTS}). Retrying in 5s...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error(f"HTTP error querying NCBI for '{name_to_query}': {http_err}")
                        break
                except Exception as e:
                    logger.error(f"An unexpected error occurred while normalizing species '{original_name}': {e}")
                    break

    logger.info("Species normalization complete.")
    return processed_results