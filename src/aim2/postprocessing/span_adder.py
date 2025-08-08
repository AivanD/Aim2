import re
from aim2.entities_types.entities import CustomExtractedEntities


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
    
    # A map to quickly find an entity object by its name to append spans
    entity_map = {}

    # First, create the unique entity entries in the map
    for entity in entity_list:
        if entity.name not in entity_map:
            new_entity_obj = {
                "name": entity.name,
                "spans": []  # Initialize with an empty list for spans
            }
            output_data[category_name].append(new_entity_obj)
            entity_map[entity.name] = new_entity_obj

    # Now, find all occurrences and populate the spans
    for entity_name, entity_obj in entity_map.items():
        for match in re.finditer(re.escape(entity_name), passage_text, re.IGNORECASE):
            start, end = match.span()
            span = (passage_offset + start, passage_offset + end)
            entity_obj["spans"].append(span)


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
        dict: A dictionary with keys "metabolites", "pathways", and "species", each containing a list of entities with added span information.
    """
    output_data = {
        "metabolites": [],
        "pathways": [],
        "species": []
    }

    _process_entity_list(entities.metabolites, "metabolites", passage_text, passage_offset, output_data)
    _process_entity_list(entities.pathways, "pathways", passage_text, passage_offset, output_data)
    _process_entity_list(entities.species, "species", passage_text, passage_offset, output_data)

    return output_data