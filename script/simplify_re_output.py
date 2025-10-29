import json
import os
import argparse
import logging
import sys

from aim2.utils.config import RE_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging

def simplify_re_output(input_path: str, output_path: str):
    """
    Reads a JSON file with extracted relations and creates a simplified copy
    containing only the subject name, predicate, object name, justification, and context.
    """
    logger = logging.getLogger(__name__)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {input_path}")
        return

    if "relations" not in data or not isinstance(data["relations"], list):
        logger.warning(f"Input file {input_path} does not contain a 'relations' list.")
        return

    simplified_relations = []
    for relation in data["relations"]:
        try:
            subject = relation.get("subject", {})
            object_entity = relation.get("object", {})

            simplified_relation = {
                "subject_name": subject.get("name"),
                "subject_alt_names": subject.get("alt_names"),
                "predicate": relation.get("predicate"),
                "object_name": object_entity.get("name"),
                "object_alt_names": object_entity.get("alt_names"),
                "justification": relation.get("justification"),
                "context": relation.get("context")
            }
            simplified_relations.append(simplified_relation)
        except AttributeError:
            logger.warning(f"Skipping a malformed relation object: {relation}")
            continue

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_relations, f, indent=2)
        logger.info(f"Successfully created simplified output at: {output_path}")
    except IOError as e:
        logger.error(f"Failed to write to output file {output_path}: {e}")


def main():
    """Main function to parse arguments and run the script."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Simplify a relation extraction JSON output file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filename",
        type=str,
        help="The name of the .json file in the 'output/re/' directory to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Optional: The name of the output file. Defaults to '[filename]_simple.json'."
    )
    args = parser.parse_args()

    input_file = os.path.join(RE_OUTPUT_DIR, args.filename)

    if args.output:
        output_file = os.path.join(RE_OUTPUT_DIR, args.output)
    else:
        base_name = os.path.splitext(args.filename)[0]
        output_file = os.path.join(RE_OUTPUT_DIR, f"{base_name}_simple.json")

    simplify_re_output(input_file, output_file)


if __name__ == "__main__":
    main()