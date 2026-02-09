import os
import shutil
import xml.etree.ElementTree as ET
import spacy
import logging

from aim2.utils.config import TARDC_INPUT_MALFORMED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_xml(file_path, for_sentences=False, nlp=None):
    """
    Parses an XML file and returns a list of sentences using scispacy's TRANSFORMER model (en_core_sci_scibert).
    Can also use en_core_sci_lg for non-transformer models.
    Args:
        file_path (str): The path to the XML file.
        for_sentences (bool): If True, enables sentence splitting and abbreviation detection.
    Returns:
        list: A list of sentences extracted from the XML file.
    """
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        logger.error(f"XML ParseError in {file_path}: {e}. Moving file to malformed directory.")
        if not os.path.exists(TARDC_INPUT_MALFORMED_DIR):
            os.makedirs(TARDC_INPUT_MALFORMED_DIR)
        shutil.move(file_path, os.path.join(TARDC_INPUT_MALFORMED_DIR, os.path.basename(file_path)))
        return None, None, None, None # Return None to signal an error

    root = tree.getroot()

    for document in root.findall('document'):
        all_passages = []       # passages along with their offsets
        all_sentences = []      # sentences from all passages
        abbreviations_dict = {}  # abbreviations from all passages
        all_annotations = []    # annotations from all passages
        found_any_annotations = False

        # document id (for debugging purposes)
        doc_id_element = document.find('id')
        if doc_id_element is not None:
            doc_id = doc_id_element.text
        else:
            doc_id = "Unknown ID"
        logger.info(f"Processing document ID: {doc_id}")

        # get all passages in the document
        for passage in document.findall('passage'):
            # skip passages under REF and title elements
            section_type_element = passage.find("infon[@key='section_type']")
            # ADJUST THIS to only include the relevant section types
            if section_type_element is not None and section_type_element.text.upper() not in ['ABSTRACT', 'INTRO', 'RESULTS', 'DISCUSS', 'CONCL', 'FIG']:
                continue
            type_element = passage.find("infon[@key='type']")
            # Adjust this to only include the relevant types
            if type_element is not None and type_element.text.upper() not in ['ABSTRACT', 'PARAGRAPH', 'FIG_TITLE_CAPTION', 'FIG_CAPTION']:
                continue

            # all passages have a text element. Save it into the list so we can batch process it
            text_element = passage.find('text')
            offset_element = passage.find('offset')
            if text_element is not None and offset_element is not None:
                all_passages.append((text_element.text, (int(offset_element.text))))

                # ------------------ PUBTATOR ANNOTATIONS EXTRACTION ------------------
                # extract annotations for this passage
                passage_annotations = {
                    "compounds": set(),
                    "species": set(),
                    "genes": set(),
                }

                for annotation in passage.findall('annotation'):
                    # <infon key="type">Chemical</infon> or <infon key="type">Gene</infon> or <infon key="type">Species</infon>
                    type_infon = annotation.find("infon[@key='type']")
                    if type_infon is None:
                        continue

                    # Chemical, Gene, or Species
                    annotation_type = type_infon.text
                    target_set = None

                    # map XML types to internal keys
                    if annotation_type == 'Chemical':
                        target_set = passage_annotations["compounds"]
                    elif annotation_type == 'Gene':
                        target_set = passage_annotations["genes"]
                    elif annotation_type == 'Species':
                        target_set = passage_annotations["species"]
                    
                    # extract the text
                    if target_set is not None:
                        text_node = annotation.find('text')
                        if text_node is not None and text_node.text:
                            target_set.add(text_node.text)
                            found_any_annotations = True

                # Convert sets to lists for the final output
                all_annotations.append({k: list(v) for k, v in passage_annotations.items()})

                # ------------------ END OF PUBTATOR ANNOTATIONS EXTRACTION ------------------
                
            else: 
                continue

        # use nlp.pipe with batch processing and using multi core (multi core is only support if you dont need abbreviation)
        # for transformers, leave n_process at 1. For non-transformers, use any nproc processes
        if for_sentences and nlp is not None:
            # extracting only the text from all_passages for nlp.pipe

            for doc, passage_offset in nlp.pipe(all_passages, as_tuples=True, n_process=1, batch_size=(min(len(all_passages), 128))):
                for sent in doc.sents:
                    sentence_offset = passage_offset + sent.start_char
                    all_sentences.append((sent.text, sentence_offset))
                for abrv in doc._.abbreviations:
                    abbreviations_dict[abrv.text] = str(abrv._.long_form)
        else:
            logger.warning("Sentence extraction and abbreviation detection skipped due to missing NLP model or for_sentences=False.")
            all_sentences = []
            abbreviations_dict = {}
    final_annotations = all_annotations if found_any_annotations else None
    return all_passages, all_sentences, abbreviations_dict, final_annotations