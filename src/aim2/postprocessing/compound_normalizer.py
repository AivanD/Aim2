import logging
import time
from typing import List, Dict, Any
import requests

logger = logging.getLogger(__name__)

# Base URL for PubChem PUG REST API
API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def normalize_compounds_with_pubchem(processed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes compound names using the PubChem REST API to fetch CID and SMILES.

    This function iterates through a list of processed results, finds the 'compounds'
    list in each, and for each compound, it queries PubChem to find a canonical
    ID (CID) and SMILES string.

    Args:
        processed_results: A list of dictionaries, where each dictionary represents
                           the extracted entities for a passage, already processed
                           by span_adder.

    Returns:
        The same list of dictionaries with compound entities updated with 'CID'
        and 'SMILES' where found.
    """
    for result in processed_results:
        if "compounds" not in result or not result["compounds"]:
            continue

        for compound in result["compounds"]:
            original_name = compound.get("name")
            if not original_name:
                continue

            try:
                # Step 1: Get CID from compound name
                cid_url = f"{API_BASE}/compound/name/{original_name}/cids/JSON"
                response = requests.get(cid_url)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                data = response.json()
                cids = data.get("IdentifierList", {}).get("CID")

                if cids:
                    first_cid = cids[0]
                    compound['CID'] = first_cid
                    
                    # Step 2: Get SMILES from CID
                    smiles_url = f"{API_BASE}/compound/cid/{first_cid}/property/IsomericSMILES/JSON"
                    smiles_response = requests.get(smiles_url)
                    smiles_response.raise_for_status()
                    
                    smiles_data = smiles_response.json()
                    properties = smiles_data.get("PropertyTable", {}).get("Properties", [])
                    
                    if properties:
                        compound['SMILES'] = properties[0].get("SMILES")
                        logger.debug(f"Normalized '{original_name}' to CID: {first_cid}")
                else:
                    logger.warning(f"Could not find PubChem entry for compound: '{original_name}'")

                # Be respectful to the API and avoid rate limiting (5 requests/sec)
                time.sleep(0.3)

            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 404:
                    logger.warning(f"Could not find PubChem entry for compound: '{original_name}' (404 Not Found)")
                else:
                    logger.error(f"HTTP error querying PubChem for '{original_name}': {http_err}")
            except Exception as e:
                logger.error(f"Error querying PubChem for '{original_name}': {e}")

    return processed_results