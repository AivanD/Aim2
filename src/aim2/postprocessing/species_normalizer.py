import logging
import sqlite3
import time
from typing import List, Dict, Any
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import re

from aim2.utils.config import DATABASE_FILE, NCBI_API_KEY

logger = logging.getLogger(__name__)

# New Base URL for NCBI Datasets API
DATASETS_BASE_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2"

def normalize_species_with_ncbi_local(processed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes species names using the local SQLite database to fetch a Taxonomy ID.
    It resolves abbreviated genus names and caches results.
    """
    species_norm_start_time = time.time()
    logger.info("Starting species normalization with local NCBI database...")

    if not DATABASE_FILE.exists():
        logger.error(f"Local database not found at {DATABASE_FILE}. Cannot perform local normalization.")
        return processed_results

    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # In-memory cache to avoid redundant DB queries
        species_cache = {}

        # --- Pass 1: Build an abbreviation map from the entire document ---
        abbreviation_map = {}
        all_species_entities = [entity for result in processed_results for entity in result.get("species", [])]
        full_names = {entity['name'] for entity in all_species_entities if re.match(r'^[A-Z][a-z]+ [a-z]+$', entity['name'])}
        for full_name in full_names:
            parts = full_name.split()
            if len(parts) == 2:
                genus, species_epithet = parts
                abbreviation = f"{genus[0]}. {species_epithet}"
                abbreviation_map[abbreviation.lower()] = full_name

        # --- Pass 2: Collect all unique, un-normalized, un-cached names to query ---
        names_to_query = set()
        for result in processed_results:
            for species_entity in result.get("species", []):
                if species_entity.get("taxonomy_id"): continue
                original_name = species_entity.get("name")
                if not original_name: continue
                
                name_to_query = abbreviation_map.get(original_name.lower(), original_name)
                if name_to_query not in species_cache:
                    names_to_query.add(name_to_query)

        # --- Pass 3: Perform batch normalization against the local DB ---
        if names_to_query:
            for name in names_to_query:
                # OPTION 1: Original simpler queries (may return non-species ranks)
                # # Query for scientific name first for higher accuracy (any rank)
                # cursor.execute("SELECT tax_id FROM taxonomy_names WHERE name = ? AND name_class = 'scientific name' COLLATE NOCASE LIMIT 1", (name,))
                # tax_id_result = cursor.fetchone()
                
                # # If not found, check other name classes (e.g., common name) (any rank)
                # if not tax_id_result:
                #     cursor.execute("SELECT tax_id FROM taxonomy_names WHERE name = ? COLLATE NOCASE LIMIT 1", (name,))
                #     tax_id_result = cursor.fetchone()

                # OPTION 2: Improved queries to ensure species rank only
                # Query for scientific name first for higher accuracy, ensuring it's a species rank
                query = """
                    SELECT T1.tax_id 
                    FROM taxonomy_names AS T1 
                    JOIN taxonomy_nodes AS T2 ON T1.tax_id = T2.tax_id 
                    WHERE T1.name = ? 
                      AND T2.rank = 'species' 
                      AND T1.name_class = 'scientific name' 
                    COLLATE NOCASE 
                    LIMIT 1
                """
                cursor.execute(query, (name,))
                tax_id_result = cursor.fetchone()
                
                # If not found, check other name classes (e.g., common name), still filtering for species rank
                if not tax_id_result:
                    fallback_query = """
                        SELECT T1.tax_id 
                        FROM taxonomy_names AS T1 
                        JOIN taxonomy_nodes AS T2 ON T1.tax_id = T2.tax_id 
                        WHERE T1.name = ? 
                          AND T2.rank = 'species' 
                        COLLATE NOCASE 
                        LIMIT 1
                    """
                    cursor.execute(fallback_query, (name,))
                    tax_id_result = cursor.fetchone()

                if tax_id_result:
                    tax_id = tax_id_result[0]
                    # Check if the tax_id has been merged into a new one
                    cursor.execute("SELECT new_tax_id FROM taxonomy_merged WHERE old_tax_id = ?", (tax_id,))
                    merged_result = cursor.fetchone()
                    if merged_result:
                        final_tax_id = merged_result[0]
                        logger.debug(f"Resolved merged TaxID {tax_id} -> {final_tax_id} for '{name}'")
                    else:
                        final_tax_id = tax_id
                    
                    species_cache[name] = {"taxonomy_id": final_tax_id}
                    logger.debug(f"Normalized '{name}' (TaxID: {final_tax_id})")
                else:
                    species_cache[name] = None # Cache failure
                    logger.warning(f"Could not find local NCBI Taxonomy entry for species: '{name}'")

        # --- Pass 4: Apply cached results to all entities ---
        for result in processed_results:
            if "species" not in result or not result["species"]: continue
            for species_entity in result["species"]:
                if species_entity.get("taxonomy_id"): continue
                original_name = species_entity.get("name")
                if not original_name: continue
                
                name_to_query = abbreviation_map.get(original_name.lower(), original_name)
                cached_result = species_cache.get(name_to_query)
                if cached_result:
                    species_entity.update(cached_result)

    except sqlite3.Error as e:
        logger.error(f"Database error during local species normalization: {e}")
    finally:
        if conn:
            conn.close()

    logger.info(f"Local species normalization complete in {time.time() - species_norm_start_time:.2f} seconds.")
    return processed_results

def normalize_species_with_ncbi(processed_results: List[Dict[str, Any]], MAX_ATTEMPTS=5, BATCH_SIZE=10) -> List[Dict[str, Any]]:
    """
    Normalizes species names using the NCBI Datasets API to fetch a Taxonomy ID and canonical name.
    It resolves abbreviated genus names within the document context before querying and caches results.
    """
    species_norm_start_time = time.time()
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

    # --- Pass 2: Collect all unique, un-normalized, un-cached names to query ---
    names_to_query = set()
    for result in processed_results:
        for species_entity in result.get("species", []):
            if species_entity.get("taxonomy_id"):
                continue
            original_name = species_entity.get("name")
            if not original_name:
                continue
            
            name_to_query = abbreviation_map.get(original_name.lower(), original_name)
            if name_to_query not in species_cache:
                names_to_query.add(name_to_query)

    # --- Pass 3: Perform batch normalization ---
    if names_to_query:
        name_list = list(names_to_query)
        for i in range(0, len(name_list), BATCH_SIZE):
            batch = name_list[i:i + BATCH_SIZE]
            batch_str = ",".join(batch)
            
            for attempt in range(MAX_ATTEMPTS):
                try:
                    search_url = (
                        f"{DATASETS_BASE_URL}/taxonomy/taxon/{urllib.parse.quote(batch_str)}"
                        "?returned_content=TAXIDS&ranks=SPECIES"
                    )
                    search_response = requests.get(search_url, headers=headers)
                    search_response.raise_for_status()

                    search_data = search_response.json()
                    
                    # Process successful response
                    found_taxa = {node['query'][0]: node['taxonomy']['tax_id'] 
                                  for node in search_data.get('taxonomy_nodes', []) 
                                  if node.get('taxonomy') and node['taxonomy'].get('tax_id')}

                    # Update cache for all names in the batch
                    for name in batch:
                        if name in found_taxa:
                            tax_id = found_taxa[name]
                            update_data = {"taxonomy_id": int(tax_id)}
                            species_cache[name] = update_data
                            logger.debug(f"Normalized '{name}' (TaxID: {tax_id})")
                        else:
                            species_cache[name] = None # Cache failure
                            logger.warning(f"Could not find NCBI Taxonomy entry for species: '{name}'")
                    
                    time.sleep(request_delay)
                    break # Success, exit retry loop for this batch

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code in [400, 401, 403]:
                        logger.error(f"NCBI API key is invalid or unauthorized. Status: {http_err.response.status_code}. "
                                     "Falling back to unauthenticated requests at a slower rate.")
                        headers.pop("api-key", None)
                        request_delay = 0.4
                        continue # Retry the same batch immediately without the key
                    elif http_err.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_ATTEMPTS - 1:
                        logger.warning(f"Server error for batch (attempt {attempt + 1}/{MAX_ATTEMPTS}). Retrying in 5s...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error(f"HTTP error querying NCBI for batch: {http_err}")
                        # Cache failure for the entire batch on unrecoverable error
                        for name in batch:
                            species_cache[name] = None
                        break
                except Exception as e:
                    logger.error(f"An unexpected error occurred while normalizing a batch: {e}")
                    for name in batch:
                        species_cache[name] = None
                    break

    # --- Pass 4: Apply cached results to all entities ---
    for result in processed_results:
        if "species" not in result or not result["species"]:
            continue
        for species_entity in result["species"]:
            if species_entity.get("taxonomy_id"):
                continue
            original_name = species_entity.get("name")
            if not original_name:
                continue
            
            name_to_query = abbreviation_map.get(original_name.lower(), original_name)
            cached_result = species_cache.get(name_to_query)
            if cached_result:
                species_entity.update(cached_result)

    logger.info(f"Species normalization complete in {time.time() - species_norm_start_time:.2f} seconds.")
    return processed_results