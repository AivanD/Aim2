import logging
import time
from typing import List, Dict, Any
import requests
import urllib.parse

logger = logging.getLogger(__name__)

# Base URL for PubChem PUG REST API
API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"      # to retreive CID and SMILES
SMILES_TO_CLASS = "https://npclassifier.gnps2.org/classify" # to retreive NP_class and NP_superclass
Classyfire = "https://structure.gnps2.org/classyfire"        # to retreive Classyfire classification using SMILES
ClassyFire_Wishart = "http://classyfire.wishartlab.com"

def get_np_class(processed_results: List[Dict[str, Any]], MAX_ATTEMPTS=10) -> List[Dict[str, Any]]:
    """
    fetches the npclass for a given compound using its SMILES

    This function iterates throguh a list of processed results, finds compounds list in each,
    and for each compound, it queries NPCLASSIFIER to find the canonical superclass and class.

    Args:
        processed_results: A list of dictionaries, where each dictionary represents
                           the extracted entities for a passage, already processed
                           by span_adder.
    Returns:
        The same list of disctionaries with compound entities uppdated with 'NP_class' and 
        'NP_superclass'.
    """
    for result in processed_results:
        if "compounds" not in result or not result["compounds"]:
            continue

        for compound in result["compounds"]:
            if 'SMILES' not in compound:
                continue

            original_name = compound.get("name")
            if not original_name:
                continue
            
            for attempt in range(MAX_ATTEMPTS):
                try:
                    np_class_url = f"{SMILES_TO_CLASS}?smiles={urllib.parse.quote(compound['SMILES'])}"
                    np_class_response = requests.get(np_class_url)
                    np_class_response.raise_for_status()

                    np_class_data = np_class_response.json()
                    compound['Natural_product_class'] = {
                        "Np_class": np_class_data.get("class_results", []),
                        "Np_superclass": np_class_data.get("superclass_results", [])
                    }

                    time.sleep(0.3)
                    break
                except requests.exceptions.HTTPError as http_err:
                    # if server error, wait 10s. break because of compounds like Iron, they dont have an np class
                    if http_err.response.status_code in [500, 502, 503, 504] and attempt < MAX_ATTEMPTS - 1:
                        logger.warning(f"Server error for '{original_name}' (attempt {attempt + 1}/{MAX_ATTEMPTS}). Skipping in 10s...")
                        time.sleep(10)
                        break
                    else:
                        logger.error(f"Error querying NPCLASSIFIER for '{original_name}': {http_err}")
                        break # Final attempt failed or non-retriable error
                except Exception as e:
                    logger.error(f"Error querying NPCLASSIFIER for '{original_name}': {e}")
                    break # Non-HTTP error, break loop
    return processed_results

def normalize_compounds_with_pubchem(processed_results: List[Dict[str, Any]], MAX_ATTEMPTS=10) -> List[Dict[str, Any]]:
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
            
            for attempt in range(MAX_ATTEMPTS):
                try:
                    # Step 1: Get CID from compound name
                    cid_url = f"{API_BASE}/compound/name/{urllib.parse.quote(original_name)}/cids/JSON"
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
                        smiles_properties = smiles_data.get("PropertyTable", {}).get("Properties", [])
                        
                        if smiles_properties:
                            compound['SMILES'] = smiles_properties[0].get("SMILES")
                            logger.debug(f"Normalized '{original_name}' to CID: {first_cid}")

                    else:
                        logger.warning(f"Could not find PubChem entry for compound: '{original_name}'")

                    # Be respectful to the API and avoid rate limiting (5 requests/sec)
                    time.sleep(0.3)
                    break

                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 404:
                        logger.warning(f"Could not find PubChem entry for compound: '{original_name}' (404 Not Found)")
                        break
                    # if server error, wait 10s
                    elif http_err.response.status_code in [500, 502, 503, 504] and attempt < MAX_ATTEMPTS - 1:
                        logger.warning(f"Server error for '{original_name}' (attempt {attempt + 1}/{MAX_ATTEMPTS}). Retrying in 10s...")
                        time.sleep(10)
                        continue
                    else:
                        logger.error(f"HTTP error querying PubChem for '{original_name}': {http_err}")
                        break
                except Exception as e:
                    logger.error(f"Error querying PubChem for '{original_name}': {e}")
                    break

    return processed_results

def get_classyfire_classification(smiles: str, MAX_ATTEMPTS=10) -> Dict[str, Any]:
    """
    Fetches Classyfire classification for a given SMILES string.

    This function queries the Classyfire API to retrieve the classification
    information for a compound represented by its SMILES string.

    Args:
        smiles: The SMILES string of the compound.
        MAX_ATTEMPTS: Maximum number of retry attempts for transient errors.
    Returns:
        A dictionary containing the Classyfire classification data.
    """
    # copy the structure of get_np_class()
    return NotImplementedError  # Placeholder for future implementation
    for attempt in range(MAX_ATTEMPTS):
        try:
            classyfire_url = f"{Classyfire}?smiles={urllib.parse.quote(smiles)}"    # TODO: Switch to wisharts lab rather than gnps2 (latter doenst work)
            response = requests.get(classyfire_url)
            response.raise_for_status()

            classification_data = response.json()
            return classification_data

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                logger.warning(f"Classyfire classification not found for SMILES: '{smiles}' (404 Not Found)")
                return {}
            elif http_err.response.status_code in [500, 502, 503, 504] and attempt < MAX_ATTEMPTS - 1:
                logger.warning(f"Server error for SMILES '{smiles}' (attempt {attempt + 1}/{MAX_ATTEMPTS}). Retrying in 10s...")
                time.sleep(10)
                continue
            else:
                logger.error(f"HTTP error querying Classyfire for SMILES '{smiles}': {http_err}")
                return {}
        except Exception as e:
            logger.error(f"Error querying Classyfire for SMILES '{smiles}': {e}")
            return {}
    return {}