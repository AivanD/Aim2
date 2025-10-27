import logging
import time
from typing import List, Dict, Any
import requests
import urllib.parse
import subprocess
import tempfile
import os
import csv

logger = logging.getLogger(__name__)

# Base URL for PubChem PUG REST API
API_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"      # to retreive CID and SMILES
SMILES_TO_CLASS = "https://npclassifier.gnps2.org/classify" # to retreive NP_class and NP_superclass

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
    logger.info("Fetching NP classes for compounds...")
    for result in processed_results:
        if "compounds" not in result or not result["compounds"]:
            continue

        for compound in result["compounds"]:
            # If the compound is already classified by an ontology (e.g., ChemOnt), skip it.
            if compound.get("ontology_id"):
                continue
            
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
    logger.info("NP class fetching complete.")
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
    logger.info("Normalizing compounds with PubChem...")
    for result in processed_results:
        if "compounds" not in result or not result["compounds"]:
            continue

        for compound in result["compounds"]:
            # If the compound is already classified by an ontology (e.g., ChemOnt), skip it.
            if compound.get("ontology_id"):
                continue

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
                        smiles_url = f"{API_BASE}/compound/cid/{first_cid}/property/IsomericSMILES,InChIKey/JSON"
                        smiles_response = requests.get(smiles_url)
                        smiles_response.raise_for_status()
                        
                        smiles_data = smiles_response.json()
                        properties = smiles_data.get("PropertyTable", {}).get("Properties", [])

                        if properties:
                            compound['SMILES'] = properties[0].get("SMILES")
                            compound['InChIKey'] = properties[0].get("InChIKey")
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
    logger.info("PubChem normalization complete.")
    return processed_results

def classify_with_classyfire_local(processed_results: List[Dict[str, Any]], jar_path='external_tools/Classyfire/Classyfire_2024.jar') -> List[Dict[str, Any]]:
    """
    Classifies compounds using a local ClassyFire .jar file.

    Args:
        processed_results: The list of processed entity dictionaries.
        jar_path: The file path to the ClassyFire .jar file.

    Returns:
        The processed results with ClassyFire classifications added.
    """
    compounds_to_classify = []
    for result in processed_results:
        for compound in result.get("compounds", []):
            if compound.get("SMILES") and compound.get("InChIKey") and not compound.get("Classyfire"):
                compounds_to_classify.append(compound)

    if not compounds_to_classify:
        logger.info("No compounds to classify with ClassyFire.")
        return processed_results

    classy_start_time = time.time()
    logger.info(f"Classifying {len(compounds_to_classify)} unmerged compounds with local ClassyFire. This may time some time...")

    with tempfile.TemporaryDirectory() as tempdir:
        input_path = os.path.join(tempdir, "input.csv")
        output_path = os.path.join(tempdir, "output.tsv")

        # 1. Write input file for the JAR
        with open(input_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['inchikey', 'smiles'])  # Header
            for compound in compounds_to_classify:
                writer.writerow([compound['InChIKey'], compound['SMILES']])

        # 2. Run the JAR file
        try:
            command = ["java", "-jar", jar_path, input_path, output_path]
            subprocess.run(command, check=True, capture_output=True, text=True)
            classy_end_time = time.time()
            logger.info(f"Classyfire_Local took {classy_end_time - classy_start_time}seconds.")
        except FileNotFoundError:
            logger.error(f"Java command not found. Please ensure Java is installed and in your PATH.")
            return processed_results
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running ClassyFire JAR file: {e.stderr}")
            return processed_results
        
        # 3. Parse the output file and create a mapping
        classification_map = {}
        try:
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    # The 'sid' column from the output contains the InChIKey
                    classification_map[row['sid']] = {
                        "Kingdom": row.get('kingdom'),
                        "Superclass": row.get('superklass'),
                        "Class": row.get('klass'),
                        "Subclass": row.get('subklass')
                    }
        except FileNotFoundError:
            logger.error(f"ClassyFire output file not found at: {output_path}")
            return processed_results

        # 4. Update compounds with classification data
        for compound in compounds_to_classify:
            if compound['InChIKey'] in classification_map:
                compound['Classyfire'] = classification_map[compound['InChIKey']]

    logger.info("ClassyFire classification complete.")
    return processed_results