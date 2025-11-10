import json
import os
import argparse
import logging
from collections import defaultdict


from aim2.utils.config import PROCESSED_RE_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging

def view_relations_by_subject(filepath: str):
    """
    Reads a simplified JSON relation file, lets the user select a subject,
    and prints all relations for that subject.
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            relations = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filepath}. The file might be empty or malformed.")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the file: {e}")
        return

    if not relations or not isinstance(relations, list):
        logger.warning("No relations found in the file.")
        return

    # Group relations by subject
    relations_by_subject = defaultdict(list)
    for rel in relations:
        subject_name = rel.get("subject_name")
        if subject_name:
            relations_by_subject[subject_name].append(rel)

    if not relations_by_subject:
        logger.warning("Could not find any subjects in the provided file.")
        return

    subjects = sorted(relations_by_subject.keys())

    # Main loop to allow user to select subjects
    while True:
        print("\nPlease select a subject to view its relations:")
        for i, subject in enumerate(subjects):
            print(f"  {i + 1}: {subject}")
        print("\nEnter 'exit' to quit.")

        choice = input("Your choice (number): ").strip().lower()

        if choice == 'exit':
            break

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(subjects):
                selected_subject = subjects[choice_idx]
                print(f"\n--- Relations for Subject: '{selected_subject}' ---")
                for rel in relations_by_subject[selected_subject]:
                    predicate = rel.get('predicate', 'N/A')
                    obj_name = rel.get('object_name', 'N/A')
                    print(f"  - Predicate: {predicate}, Object: {obj_name}")
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'exit'.")

def main():
    """Main function to parse arguments and run the script."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Interactively view relations from a simplified JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filename",
        type=str,
        help="The name of the simplified .json file in the 'output/re/processed/' directory."
    )
    args = parser.parse_args()

    # The user's active file is in `output/re/processed`, so we'll use that directory.
    input_file = os.path.join(PROCESSED_RE_OUTPUT_DIR, args.filename)

    view_relations_by_subject(input_file)

if __name__ == "__main__":
    # Add the project root to sys.path to allow imports from src
    import sys
    # This is a common pattern: go up from script/ to the project root.
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(project_root) == 'script':
        project_root = os.path.dirname(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    main()