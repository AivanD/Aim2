import pandas as pd
import json
import os
import sys
import logging

from aim2.utils.config import RE_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging

def convert_jsonl_to_xlsx(input_file_path, output_file_path):
    """
    Reads a JSONL file, converts it to a pandas DataFrame, and saves it as an XLSX file.
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(input_file_path):
        logger.error(f"Input file not found: {input_file_path}")
        return

    logger.info(f"Reading data from {input_file_path}...")
    
    data = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_file_path}: {e}")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return

    if not data:
        logger.warning("No data found in the input file. The output file will be empty.")
        # Create an empty file to indicate completion
        pd.DataFrame().to_excel(output_file_path, index=False)
        return

    df = pd.DataFrame(data)
    
    logger.info(f"Writing {len(df)} records to {output_file_path}...")
    try:
        df.to_excel(output_file_path, index=False, engine='openpyxl')
        logger.info("Conversion successful.")
    except Exception as e:
        logger.error(f"Failed to write to Excel file: {e}")

def main():
    """
    Main function to define file paths and run the conversion.
    """
    setup_logging()
    
    # Define the directory and filenames
    annotated_dir = os.path.join(RE_OUTPUT_DIR, 'annotated')
    input_filename = 'PMC7384185.jsonl'
    output_filename = 'PMC7384185.xlsx'
    
    input_path = os.path.join(annotated_dir, input_filename)
    output_path = os.path.join(annotated_dir, output_filename)
    
    # Ensure the output directory exists
    os.makedirs(annotated_dir, exist_ok=True)
    
    convert_jsonl_to_xlsx(input_path, output_path)

if __name__ == "__main__":
    main()