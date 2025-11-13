import json
import pandas as pd
import os
import re
import argparse

from aim2.utils.config import PROCESSED_RE_OUTPUT_DIR

def process_json_file(json_file_path):
    """
    Reads a JSON file containing a list of relations, extracts specific fields,
    and returns them as a list of records.

    Args:
        json_file_path (str): The path to the input JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary is a record.
              Returns an empty list if the file cannot be processed.
    """
    # Extract the PMCID from the filename
    basename = os.path.basename(json_file_path)
    match = re.search(r'PMC(\d+)', basename)
    pmcid = match.group(1) if match else None
    pmcid = f"PMC{pmcid}" if pmcid else "Unknown"

    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'.")
        return []

    # Process the data
    records = []
    for relation in data:
        record = {
            "PMCID": pmcid,
            "subject_name": relation.get("subject_name"),
            "subject_cid": relation.get("subject_cid"),
            "predicate": relation.get("predicate"),
            "object_name": relation.get("object_name"),
            "object_alt_names": str(relation.get("object_alt_names")) if relation.get("object_alt_names") is not None else None,
            "object_ontology_id": relation.get("object_ontology_id"),
            "category": relation.get("category")
        }
        records.append(record)
    return records

def process_all_json_to_excel(input_dir, output_excel_path):
    """
    Processes all valid JSON files in a directory and compiles them into a single Excel file.

    Args:
        input_dir (str): The directory containing the JSON files.
        output_excel_path (str): The path for the output Excel file.
    """
    all_records = []
    file_count = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".json") and "no_relationships" not in filename:
            file_path = os.path.join(input_dir, filename)
            records = process_json_file(file_path)
            if records:
                all_records.extend(records)
                file_count += 1

    # Create a DataFrame and save to Excel
    if all_records:
        df = pd.DataFrame(all_records)
        
        # Ensure columns are in the desired order
        column_order = [
            "PMCID", "subject_name", "subject_cid", "predicate", 
            "object_name", "object_alt_names", "object_ontology_id", "category"
        ]
        df = df[column_order]
        
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"Successfully created '{output_excel_path}' with {len(all_records)} records from {file_count} files.")
    else:
        print("No records found in any of the JSON files in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert relation JSON files from a directory to a single Excel file.")
    parser.add_argument("-o", "--output", default="Eval.xlsx", help="Path for the output Excel file (default: Eval.xlsx).")
    args = parser.parse_args()

    process_all_json_to_excel(PROCESSED_RE_OUTPUT_DIR, args.output)