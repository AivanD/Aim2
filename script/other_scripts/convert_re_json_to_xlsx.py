import json
import pandas as pd
import os
import re
import argparse

def json_to_excel(json_file_path, output_excel_path):
    """
    Reads a JSON file containing a list of relations, extracts specific fields,
    and saves them to an Excel file.

    Args:
        json_file_path (str): The path to the input JSON file.
        output_excel_path (str): The path for the output Excel file.
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
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'.")
        return

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

    # Create a DataFrame and save to Excel
    if records:
        df = pd.DataFrame(records)
        
        # Ensure columns are in the desired order
        column_order = [
            "PMCID", "subject_name", "subject_cid", "predicate", 
            "object_name", "object_alt_names", "object_ontology_id", "category"
        ]
        df = df[column_order]
        
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"Successfully created '{output_excel_path}' with {len(records)} records.")
    else:
        print("No records found in the JSON file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert relation JSON to an Excel file.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    args = parser.parse_args()

    output_filename = "Eval.xlsx"
    json_to_excel(args.json_file, output_filename)