import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def _get_deduplication_key(entity: Dict[str, Any], keys_to_try: List[str]) -> str:
    """
    Gets the first available key from the entity to use for deduplication.
    Falls back to the lowercased entity name if no other key is found.
    """
    for key in keys_to_try:
        if entity.get(key):
            return str(entity[key])
    return entity.get("name", "").lower()

def merge_and_deduplicate(processed_results: List[Dict[str, Any]], abbreviation_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Merges entities from multiple passages into a single document-level dictionary
    and deduplicates them based on canonical IDs or names.

    When a duplicate is found, its spans are merged, and its name is added to
    an 'alt_names' list if it's different from the canonical name.
    """
    logger.info("Starting entity merging and deduplication for the entire document...")

    merged_entities_store = defaultdict(dict)

    deduplication_keys_map = {
        "species": ["taxonomy_id"],
        "compounds": ["ontology_id", "CID"],
        "pathways": ["ontology_id"],
        "anatomical_structures": ["ontology_id"],
        "molecular_traits": ["ontology_id"],
        "plant_traits": ["ontology_id"],
        "experimental_conditions": ["ontology_id"],
        "genes": [],  # Will default to name
        "human_traits": [], # Will default to name
    }

    for passage in processed_results:
        for entity_type, entities in passage.items():
            if not entities:
                continue
            
            keys_to_try = deduplication_keys_map.get(entity_type, [])

            for entity in entities:
                dedupe_key = _get_deduplication_key(entity, keys_to_try)
                
                if not dedupe_key:
                    continue

                if dedupe_key in merged_entities_store[entity_type]:
                    existing_entity = merged_entities_store[entity_type][dedupe_key]
                    
                    # Merge spans
                    if "spans" in entity and entity["spans"]:
                        if "spans" not in existing_entity:
                            existing_entity["spans"] = []
                        existing_entity["spans"].extend(entity["spans"])
                    
                    # Add alternative name if it's different
                    new_name = entity.get("name")
                    if new_name and new_name.lower() != existing_entity.get("name", "").lower():
                        if "alt_names" not in existing_entity or existing_entity["alt_names"] is None:
                            existing_entity["alt_names"] = []
                        if new_name not in existing_entity["alt_names"]:
                            existing_entity["alt_names"].append(new_name)
                else:
                    # This is the first time we see this entity (based on its ID).
                    # We store it as is, keeping its original in-text name.
                    merged_entities_store[entity_type][dedupe_key] = entity

    # Convert the store back to the final list format
    final_merged_entities = {}
    for entity_type, entities_dict in merged_entities_store.items():
        final_merged_entities[entity_type] = list(entities_dict.values())
        for entity in final_merged_entities[entity_type]:
            # Clean up spans
            if "spans" in entity and entity["spans"]:
                unique_spans = sorted(list(set(tuple(span) for span in entity["spans"])))
                entity["spans"] = [list(span) for span in unique_spans]
            # Clean up alt_names
            if "alt_names" in entity and entity["alt_names"]:
                entity["alt_names"] = sorted(list(set(entity["alt_names"])))

    logger.info("Entity merging and deduplication complete.")
    return final_merged_entities