import os
import logging
import re
import json
import pandas as pd

from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, NER_OUTPUT_DIR
from aim2.xml.xml_parser import parse_xml
from aim2.utils.logging_cfg import setup_logging
from aim2.entities_types.entities import CustomExtractedEntities

def find_spans(passage_text, passage_offset, entity_text):
    """Finds all occurrences of entity_text as a whole word in the passage and returns their spans."""
    spans = []
    try:
        # use word boundaries only for single-word entities
        if " " in entity_text or "(" in entity_text:
            pattern = re.escape(entity_text)
        else:
            pattern = r'\b' + re.escape(entity_text) + r'\b'
        for match in re.finditer(pattern, passage_text, re.IGNORECASE):
            start, end = match.span()
            spans.append([passage_offset + start, passage_offset + end])
    except re.error as e:
        logging.error(f"Regex error for entity '{entity_text}': {e}")
    return spans

def write_reference_file(output_path, filename, annotated_entities):
    """Writes the set of annotated entities to a reference text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"--- Annotated Entities for {filename} ---\n")
        if annotated_entities:
            sorted_entities = sorted(list(annotated_entities))
            for entity_type, entity_name in sorted_entities:
                f.write(f"{entity_type}: {entity_name}\n")
        else:
            f.write("No entities annotated yet.\n")
    print(f"Reference file updated: {output_path}")

def main():
    ensure_dirs()
    setup_logging()
    
    logger = logging.getLogger(__name__)

    # Get the list of valid entity types from the Pydantic model
    entity_types = list(CustomExtractedEntities.model_fields.keys())

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith('.xml'):
            continue

        # directories
        input_path = os.path.join(INPUT_DIR, filename)
        ANNOTATION_OUTPUT_DIR = os.path.join(NER_OUTPUT_DIR, 'annotated')
        if not os.path.exists(ANNOTATION_OUTPUT_DIR):
            os.makedirs(ANNOTATION_OUTPUT_DIR)
        output_excel_path = os.path.join(ANNOTATION_OUTPUT_DIR, filename.replace('.xml', '.xlsx'))
        output_ref_path = os.path.join(ANNOTATION_OUTPUT_DIR, filename.replace('.xml', '_reference.txt'))

        # get the passages to annotate for entities
        passages_w_offsets, _, _ = parse_xml(input_path, False)
        
        all_annotations_for_file = []
        annotated_entities = set() # To track entities already annotated in this session

        # --- Load existing annotations if available ---
        if os.path.exists(output_excel_path):
            try:
                logger.info(f"Loading existing annotations from {output_excel_path}")
                df_existing = pd.read_excel(output_excel_path)
                # Ensure 'Ontology ID' column exists
                if 'Ontology ID' not in df_existing.columns:
                    df_existing['Ontology ID'] = ''
                
                # Fill NaN values only in the 'Ontology ID' column and ensure it's a string type
                df_existing['Ontology ID'] = df_existing['Ontology ID'].fillna('').astype(str)

                all_annotations_for_file = df_existing.to_dict('records')

                for record in all_annotations_for_file:
                    entity_type = record.get('entity type')
                    entity_name = record.get('entity')
                    if entity_type and entity_name:
                        annotated_entities.add((entity_type, str(entity_name).lower()))
                
                logger.info(f"Loaded {len(all_annotations_for_file)} existing annotations.")
                # --- Generate reference file on load ---
                write_reference_file(output_ref_path, filename, annotated_entities)

            except Exception as e:
                logger.error(f"Could not load existing annotations from {output_excel_path}. Error: {e}. Starting fresh.")
                all_annotations_for_file = []
                annotated_entities = set()

        print(f"\n--- Annotating file: {filename} ---")

        # This outer loop is now just for displaying passages to the user for context
        counter = 0
        for passage, offset in passages_w_offsets:
            print("\n" + "="*80)
            print(f"Passage {counter} (Offset: {offset}):\n{passage}")
            print("="*80)
            
            # Loop to add multiple entities while viewing the current passage
            while True:
                # Ask user to select an entity type
                print("\nSelect an entity type to annotate (will search all passages):")
                for i, etype in enumerate(entity_types):
                    print(f"  {i+1}: {etype}")
                print("Enter 'next' for next passage, 'remove' to delete an annotation, or 'exit' to quit and save.")

                type_input = input("Your choice (number): ").strip().lower()

                if type_input == 'exit':
                    print("Exiting annotation for this file.")
                    # Write reference file before exiting
                    write_reference_file(output_ref_path, filename, annotated_entities)
                    # Sort and save before exiting completely
                    if all_annotations_for_file:
                        all_annotations_for_file.sort(key=lambda x: (x['start'], x['end']))
                        df = pd.DataFrame(all_annotations_for_file)
                        df.to_excel(output_excel_path, index=False)
                        logger.info(f"Annotations for {filename} saved to {output_excel_path}")
                    return
                if type_input == 'next':
                    # Write unique annotated entities to a reference file, overwriting it each time
                    write_reference_file(output_ref_path, filename, annotated_entities)
                    counter += 1    
                    break
                
                if type_input == 'remove':
                    if not annotated_entities:
                        print("No annotations to remove yet.")
                        continue

                    print("\n--- Remove an Annotation ---")
                    # 1. Ask for entity type to remove
                    print("Select the entity type of the annotation to remove:")
                    for i, etype in enumerate(entity_types):
                        print(f"  {i+1}: {etype}")
                    
                    remove_type_input = input("Your choice (number): ").strip()
                    try:
                        remove_choice_idx = int(remove_type_input) - 1
                        if not 0 <= remove_choice_idx < len(entity_types):
                            raise ValueError
                        type_to_remove = entity_types[remove_choice_idx]
                    except (ValueError, IndexError):
                        print("Invalid choice. Returning to main menu.")
                        continue

                    # 2. Ask for entity text to remove
                    text_to_remove = input(f"Enter the exact entity text to remove for type '{type_to_remove}': ").strip()
                    if not text_to_remove:
                        print("Entity text cannot be empty. Returning to main menu.")
                        continue
                    
                    entity_tuple_to_remove = (type_to_remove, text_to_remove.lower())

                    # 3. Check if it exists and remove it
                    if entity_tuple_to_remove in annotated_entities:
                        # Remove from the tracking set
                        annotated_entities.remove(entity_tuple_to_remove)
                        
                        # Filter the main list to remove all occurrences
                        original_count = len(all_annotations_for_file)
                        all_annotations_for_file = [
                            ann for ann in all_annotations_for_file 
                            if not (ann['entity type'] == type_to_remove and ann['entity'].lower() == text_to_remove.lower())
                        ]
                        removed_count = original_count - len(all_annotations_for_file)

                        print(f"Successfully removed {removed_count} occurrence(s) of '{text_to_remove}'.")
                        
                        # Update the reference file immediately
                        write_reference_file(output_ref_path, filename, annotated_entities)
                    else:
                        print(f"Annotation '{text_to_remove}' of type '{type_to_remove}' not found.")
                    
                    continue

                try:
                    choice_idx = int(type_input) - 1
                    if not 0 <= choice_idx < len(entity_types):
                        raise ValueError
                    selected_type = entity_types[choice_idx]
                except ValueError:
                    print("Invalid choice. Please enter a number from the list.")
                    continue

                # Ask user for the entity text
                entity_text = input(f"Enter the text for '{selected_type}': ").strip()
                if not entity_text:
                    print("Entity text cannot be empty.")
                    continue

                # Prevent re-annotating the same entity text with the same type
                if (selected_type, entity_text.lower()) in annotated_entities:
                    print(f"'{entity_text}' has already been annotated as '{selected_type}'. Skipping.")
                    continue

                # Ask user for the Ontology ID (optional)
                ontology_id = input(f"Enter the Ontology ID for '{entity_text}' (optional, press Enter to skip): ").strip()

                total_occurrences = 0
                # Search for the entity in ALL passages
                for p_text, p_offset in passages_w_offsets:
                    spans = find_spans(p_text, p_offset, entity_text)
                    if spans:
                        total_occurrences += len(spans)
                        for start, end in spans:
                            annotation_row = {
                                "entity type": selected_type,
                                "entity": entity_text,
                                "start": start,
                                "end": end, 
                                "Ontology ID": ontology_id
                            }
                            all_annotations_for_file.append(annotation_row)
                
                if total_occurrences > 0:
                    print(f"Annotation added for '{entity_text}' with {total_occurrences} occurrence(s) across the document.")
                    annotated_entities.add((selected_type, entity_text.lower()))
                else:
                    print(f"Warning: Could not find '{entity_text}' in any passage. Entity not added.")

        # Save all annotations for the file to an Excel file after iterating through all passages
        if all_annotations_for_file:
            # Sort all collected annotations by start and end position
            all_annotations_for_file.sort(key=lambda x: (x['start'], x['end']))
            df = pd.DataFrame(all_annotations_for_file)
            # Remove duplicates just in case loading and re-annotating caused any
            df.drop_duplicates(subset=['entity type', 'entity', 'start', 'end'], keep='last', inplace=True)
            df.to_excel(output_excel_path, index=False)
            logger.info(f"Final annotations for {filename} saved to {output_excel_path}")
            # Also write the final reference file here
            write_reference_file(output_ref_path, filename, annotated_entities)
        else:
            logger.info(f"No annotations were made for {filename}.")

if __name__ == "__main__":
    main()