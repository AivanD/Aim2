import logging
import time
from typing import List, Dict, Any
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import re

logger = logging.getLogger(__name__)

# Base URL for NCBI E-utilities
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def normalize_species_with_ncbi(processed_results: List[Dict[str, Any]], MAX_ATTEMPTS=5) -> List[Dict[str, Any]]:
    """
    Normalizes species names using the NCBI Taxonomy API to fetch a Taxonomy ID and canonical name.
    It resolves abbreviated genus names within the document context before querying.
    """
    logger.info("Starting species normalization with NCBI Taxonomy...")

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

            for attempt in range(MAX_ATTEMPTS):
                try:
                    # Step 1: Search for the TaxID using esearch
                    search_url = f"{EUTILS_BASE}esearch.fcgi?db=taxonomy&term={urllib.parse.quote(name_to_query)}&retmode=json"
                    search_response = requests.get(search_url)
                    search_response.raise_for_status()
                    search_data = search_response.json()

                    id_list = search_data.get("esearchresult", {}).get("idlist", [])
                    
                    if id_list:
                        tax_id = id_list[0] # Take the first result
                        
                        # # Step 2: Fetch the details for that TaxID using efetch
                        # fetch_url = f"{EUTILS_BASE}efetch.fcgi?db=taxonomy&id={tax_id}&retmode=xml"
                        # fetch_response = requests.get(fetch_url)
                        # fetch_response.raise_for_status()
                        
                        # # Parse the XML response to get the scientific name
                        # root = ET.fromstring(fetch_response.content)
                        # scientific_name_element = root.find(".//ScientificName")
                        
                        # if scientific_name_element is not None:
                        #     normalized_name = scientific_name_element.text
                        species_entity["taxonomy_id"] = int(tax_id)
                        #     species_entity["normalized_name"] = normalized_name
                        #     logger.debug(f"Normalized '{original_name}' -> '{name_to_query}' to '{normalized_name}' (TaxID: {tax_id})")
                        # else:
                        #     logger.warning(f"Found TaxID {tax_id} for '{name_to_query}' but could not fetch scientific name.")
                    else:
                        logger.warning(f"Could not find NCBI Taxonomy entry for species: '{name_to_query}'")

                    # Be respectful to the API (max 3 requests/sec without API key)
                    time.sleep(0.4)
                    break # Success, exit retry loop

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 404:
                        logger.warning(f"Could not find NCBI entry for species: '{name_to_query}' (404 Not Found)")
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