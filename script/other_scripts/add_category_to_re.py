import os
import json
import logging
from aim2.utils.config import RAW_RE_OUTPUT_DIR, PROCESSED_NER_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging

def create_entity_category_map(processed_ner_data: dict) -> dict:
    """Creates a mapping from entity name to its category."""
    category_map = {}
    for category, entities in processed_ner_data.items():
        if isinstance(entities, list):
            for entity in entities:
                if 'name' in entity:
                    # Use lowercased name for case-insensitive matching
                    category_map[entity['name'].lower()] = category
    return category_map

def main():
    """
    Updates existing raw relation extraction JSON files to include the 'category'
    field for each relation.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting to update raw RE files with category information...")

    # Ensure the directories exist
    if not os.path.isdir(RAW_RE_OUTPUT_DIR) or not os.path.isdir(PROCESSED_NER_OUTPUT_DIR):
        logger.error("Required directories 'output/re/raw' or 'output/ner/processed' not found.")
        return

    for filename in os.listdir(RAW_RE_OUTPUT_DIR):
        if not filename.endswith('.json') or '_no_relationships' in filename:
            continue

        raw_re_path = os.path.join(RAW_RE_OUTPUT_DIR, filename)
        processed_ner_path = os.path.join(PROCESSED_NER_OUTPUT_DIR, filename)

        if not os.path.exists(processed_ner_path):
            logger.warning(f"Skipping {filename}: Corresponding processed NER file not found at {processed_ner_path}")
            continue

        try:
            # Load the processed NER data to get categories
            with open(processed_ner_path, 'r', encoding='utf-8') as f:
                processed_ner_data = json.load(f)
            
            # Create the name -> category map
            entity_to_category = create_entity_category_map(processed_ner_data)

            # Load the raw RE data
            with open(raw_re_path, 'r', encoding='utf-8') as f:
                re_data = json.load(f)

            if "relations" not in re_data:
                logger.warning(f"Skipping {filename}: No 'relations' key found.")
                continue

            relations_updated = 0
            # Update each relation
            for relation in re_data.get("relations", []):
                # Skip if category already exists
                if "category" in relation:
                    continue

                object_entity_name = relation.get("object_entity", {}).get("name", "").lower()
                if object_entity_name in entity_to_category:
                    relation["category"] = entity_to_category[object_entity_name]
                    relations_updated += 1
                else:
                    logger.warning(f"Could not find category for object entity '{object_entity_name}' in {filename}")

            # Overwrite the file with the updated data
            if relations_updated > 0:
                with open(raw_re_path, 'w', encoding='utf-8') as f:
                    json.dump(re_data, f, indent=2)
                logger.info(f"Updated {relations_updated} relations in {filename}")
            else:
                logger.info(f"No updates needed for {filename} (relations may already have categories).")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    logger.info("Finished updating raw RE files.")

if __name__ == "__main__":
    main()