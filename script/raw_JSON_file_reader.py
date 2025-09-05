import json

from aim2.utils.config import RAW_OUTPUT_DIR
from aim2.entities_types.entities import CustomExtractedEntities

def main():
    # read user input for a raw json file path
    raw_json_file = input("Enter the filename of the raw JSON file: ").strip()
    raw_json_file_path = f"{RAW_OUTPUT_DIR}/{raw_json_file}.json"

    try:
        with open(raw_json_file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            print("Successfully read the raw JSON file.")
    except Exception as e:
        print(f"Error reading the raw JSON file: {e}")
        return
    
    # merge all the entity labels from the raw data
    merged_entities = {}
    for entry in raw_data:
        # ensure that the entry is a valid CustomExtractedEntities object (in case user edited the raw json file)
        try:
            _ = CustomExtractedEntities.model_validate(entry)
        except Exception as e:
            print(f"Invalid entry in raw JSON file: {e}")
            continue

        for category, entities in entry.items():
            if category not in merged_entities:
                merged_entities[category] = set()
            if entities: # Ensure the list of entities is not empty
                for entity in entities:
                    # Each entity is a Pydantic model (or dict) with a 'name' attribute
                    merged_entities[category].add(entity['name'])


    # print the merged entities
    for category, entities in sorted(merged_entities.items()):
        print(f"Category: {category}")
        for entity_name in sorted(list(entities)):
            print(f" - {entity_name}")

if __name__ == "__main__":
    main()
