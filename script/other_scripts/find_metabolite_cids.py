import pandas as pd
import sqlite3
import logging
import os
import re
from tqdm import tqdm

from aim2.utils.config import DATABASE_FILE
from aim2.utils.logging_cfg import setup_logging

# --- Configuration ---
# TODO: Update these paths as needed
INPUT_EXCEL_FILE = 'Metabolite-Classes.xlsx'
OUTPUT_EXCEL_FILE = 'Metabolite-Classes_with_CIDs.xlsx'
# excel file columns
METABOLITE_COLUMN_NAME = 'Metabolite Name'
MANUAL_CID_COLUMN_NAME = 'PubChem CID' # The column with manually added CIDs
# added excel file columns
OUTPUT_CID_COLUMN_NAME = 'Found PubChem CID'
COMPARISON_COLUMN_NAME = 'CID Match' # The new column for the comparison result

# Greek letter mapping (lowercase and uppercase)
GREEK_TO_ASCII = {
    'α': 'alpha', 'Α': 'Alpha',
    'β': 'beta', 'Β': 'Beta',
    'γ': 'gamma', 'Γ': 'Gamma',
    'δ': 'delta', 'Δ': 'Delta',
    'ε': 'epsilon', 'Ε': 'Epsilon',
    'ζ': 'zeta', 'Ζ': 'Zeta',
    'η': 'eta', 'Η': 'Eta',
    'θ': 'theta', 'Θ': 'Theta',
    'ι': 'iota', 'Ι': 'Iota',
    'κ': 'kappa', 'Κ': 'Kappa',
    'λ': 'lambda', 'Λ': 'Lambda',
    'μ': 'mu', 'Μ': 'Mu',
    'ν': 'nu', 'Ν': 'Nu',
    'ξ': 'xi', 'Ξ': 'Xi',
    'ο': 'omicron', 'Ο': 'Omicron',
    'π': 'pi', 'Π': 'Pi',
    'ρ': 'rho', 'Ρ': 'Rho',
    'σ': 'sigma', 'ς': 'sigma', 'Σ': 'Sigma',
    'τ': 'tau', 'Τ': 'Tau',
    'υ': 'upsilon', 'Υ': 'Upsilon',
    'φ': 'phi', 'Φ': 'Phi',
    'χ': 'chi', 'Χ': 'Chi',
    'ψ': 'psi', 'Ψ': 'Psi',
    'ω': 'omega', 'Ω': 'Omega',
}


def normalize_greek_letters(name: str) -> str:
    """Replace Greek letters with their ASCII equivalents."""
    for greek, ascii_equiv in GREEK_TO_ASCII.items():
        name = name.replace(greek, ascii_equiv)
    return name


def extract_name_variants(metabolite_name: str) -> list[str]:
    """
    Extract all possible name variants from a metabolite name.
    
    Handles formats like:
    - "Glycerol trioctanoate (Trioctanoin / Tricaprylin)"
    - "Glycerol tripropanoate (Tripropionin)"
    - "5-(3',5'-dihydroxyphenyl)-γ-valerolactone 3-O-glucuronide"
    
    Returns a list of variants to try, in order of priority.
    """
    variants = []
    
    # First, try the original name
    variants.append(metabolite_name.strip())
    
    # Try with Greek letters normalized
    normalized = normalize_greek_letters(metabolite_name)
    if normalized != metabolite_name:
        variants.append(normalized.strip())
    
    # Check for parenthetical synonyms: "Main Name (Synonym1 / Synonym2)"
    match = re.match(r'^([^(]+)\s*\(([^)]+)\)\s*$', metabolite_name)
    if match:
        main_name = match.group(1).strip()
        synonyms_str = match.group(2)
        
        # Add the main name without parentheses
        variants.append(main_name)
        variants.append(normalize_greek_letters(main_name))
        
        # Split synonyms by "/" or ";" and add each
        synonyms = re.split(r'\s*[/;]\s*', synonyms_str)
        for syn in synonyms:
            syn = syn.strip()
            if syn:
                variants.append(syn)
                variants.append(normalize_greek_letters(syn))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique_variants.append(v)
    
    return unique_variants


def find_cid_for_metabolite(cursor, metabolite_name: str) -> tuple[int | None, str | None]:
    """
    Try to find a CID for the metabolite name, attempting multiple variants.
    
    Returns:
        tuple: (found_cid, matched_variant) or (None, None) if not found
    """
    variants = extract_name_variants(metabolite_name)
    
    for variant in variants:
        cursor.execute(
            "SELECT cid FROM pubchem_synonyms WHERE synonym = ? COLLATE NOCASE LIMIT 1",
            (variant,)
        )
        result = cursor.fetchone()
        if result:
            return result[0], variant
    
    return None, None


def find_cids_from_database(input_excel_path: str, output_excel_path: str):
    """
    Reads an Excel file with metabolite names, finds their PubChem CIDs using the local database,
    compares them to a manually provided CID column, and saves the results to a new Excel file.

    The input Excel file must have a column named 'Metabolite Name'.
    """
    logger = logging.getLogger(__name__)
    setup_logging()

    # 1. Validate paths
    if not os.path.exists(DATABASE_FILE):
        logger.error(f"Local database not found at '{DATABASE_FILE}'. Please run the database build script first.")
        return
    if not os.path.exists(input_excel_path):
        logger.error(f"Input Excel file not found at '{input_excel_path}'.")
        return

    # 2. Read the Excel file
    try:
        df = pd.read_excel(input_excel_path)
        logger.info(f"Successfully loaded {len(df)} rows from '{input_excel_path}'.")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        return

    if METABOLITE_COLUMN_NAME not in df.columns:
        logger.error(f"The required column '{METABOLITE_COLUMN_NAME}' was not found in the Excel file.")
        return
    
    perform_comparison = MANUAL_CID_COLUMN_NAME in df.columns
    if not perform_comparison:
        logger.warning(f"Column '{MANUAL_CID_COLUMN_NAME}' not found. Skipping CID comparison.")

    # 3. Connect to the database and process rows
    conn = None
    found_count = 0
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        logger.info("Connected to the local database.")

        # Prepare lists to store the results
        found_cids = []
        matched_variants = []  # Track which variant matched
        comparison_results = []

        # Use tqdm for a progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Finding CIDs"):
            metabolite_name = row[METABOLITE_COLUMN_NAME]
            found_cid = None
            matched_variant = None
            
            if not (pd.isna(metabolite_name) or not isinstance(metabolite_name, str) or not metabolite_name.strip()):
                # Try to find CID using multiple variants
                found_cid, matched_variant = find_cid_for_metabolite(cursor, metabolite_name)

                if found_cid:
                    found_count += 1
                    if matched_variant != metabolite_name:
                        logger.debug(f"Found CID for '{metabolite_name}' using variant: '{matched_variant}'")
                else:
                    logger.warning(f"Could not find CID for: '{metabolite_name}'")
            
            found_cids.append(found_cid)
            matched_variants.append(matched_variant)

            # Compare with manual CID if possible
            if perform_comparison:
                manual_cid = row[MANUAL_CID_COLUMN_NAME]
                if pd.isna(manual_cid):
                    comparison_results.append("No Manual CID")
                elif found_cid is None:
                    comparison_results.append("Not Found in DB")
                elif int(found_cid) == int(manual_cid):
                    comparison_results.append("Match")
                else:
                    comparison_results.append("Mismatch")

        # Add the new columns to the DataFrame
        df[OUTPUT_CID_COLUMN_NAME] = found_cids
        df['Matched Variant'] = matched_variants  # Optional: see which variant matched
        if perform_comparison:
            df[COMPARISON_COLUMN_NAME] = comparison_results

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    # 4. Save the updated DataFrame to a new Excel file
    try:
        df.to_excel(output_excel_path, index=False)
        logger.info(f"Found CIDs for {found_count}/{len(df)} metabolites.")
        logger.info(f"Processing complete. Results saved to '{output_excel_path}'.")
    except Exception as e:
        logger.error(f"Failed to save output Excel file: {e}")


if __name__ == "__main__":
    # Before running, make sure to update the INPUT_EXCEL_FILE and OUTPUT_EXCEL_FILE paths above.
    find_cids_from_database(INPUT_EXCEL_FILE, OUTPUT_EXCEL_FILE)