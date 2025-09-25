import re
import logging

from aim2.entities_types.entities import CustomExtractedEntities
from aim2.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# A helper function to process each entity list
def _process_entity_list(entity_list, category_name, passage_text, passage_offset, output_data):
    """
    (Helper function) (add_spans_to_entities) Processes a list of entity objects by identifying all occurrences of each entity's name within a given passage of text.
    For each entity, it adds the character spans (start and end indices) of all matches to the output data structure under the specified category.
    Args:
        entity_list (list): List of entity objects, each with a 'name' attribute.
        category_name (str): The key under which entities and their spans are stored in the output_data dictionary.
        passage_text (str): The text passage in which to search for entity names.
        passage_offset (int): The offset to add to each span's start and end indices, typically representing the passage's position in a larger text.
        output_data (dict): The data structure to which entities and their spans are appended. Must contain a list for the given category_name.
    Returns:
        None: The function modifies output_data in place by appending entities and their spans.
    """
    
    # Get unique entity names from the list to avoid duplicate processing
    unique_entity_names = {entity.name for entity in entity_list}

    for entity_name in unique_entity_names:
        spans = []
        # Find all occurrences of this entity name in the passage, ignoring case
        try:
            # Use word boundaries (\b) to match whole words only
            pattern = r'\b' + re.escape(entity_name) + r'\b'
            for match in re.finditer(pattern, passage_text, re.IGNORECASE):
                start, end = match.span()
                spans.append((passage_offset + start, passage_offset + end))
        except re.error as e:
            logger.error(f"Regex error for entity '{entity_name}': {e}")
            continue

        # Only add the entity to the output if it was actually found in the text (i.e., has spans)
        if spans:
            new_entity_obj = {
                "name": entity_name,
                "spans": spans
            }
            output_data[category_name].append(new_entity_obj)
        else:
            # This is where hallucinated entities are caught and discarded.
            logger.warning(
                f"Discarding entity '{entity_name}' (category: {category_name}) "
                f"because it was not found in the passage text."
            )


def add_spans_to_entities(
    entities: CustomExtractedEntities,
    passage_text: str,
    passage_offset: int
) -> dict:
    """
    Adds span information to extracted entities by processing lists of metabolites, pathways, and species.
    Args:
        entities (CustomExtractedEntities): An object containing lists of extracted entities (metabolites, pathways, species).
        passage_text (str): The text of the passage from which entities were extracted.
        passage_offset (int): The offset to apply to entity spans within the passage text.
    Returns:
        dict: A dictionary with keys for each entity category, each containing a list of entities with added span information.
    """
    output_data = {
        "compounds": [],
        "pathways": [],
        "genes": [],
        "anatomical_structures": [],
        "species": [],
        "experimental_conditions": [],
        # "natural_product_classes": [],
        "molecular_traits": [],
        "plant_traits": [],
        "human_traits": [],
    }

    for category in output_data.keys():
        _process_entity_list(getattr(entities, category), category, passage_text, passage_offset, output_data)

    return output_data