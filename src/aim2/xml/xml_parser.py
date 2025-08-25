import xml.etree.ElementTree as ET
import spacy
import logging
from scispacy.abbreviation import AbbreviationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_xml(file_path, for_sentences=False):
    """
    Parses an XML file and returns a list of sentences using scispacy's TRANSFORMER model (en_core_sci_scibert).
    Can also use en_core_sci_lg for non-transformer models.
    Args:
        file_path (str): The path to the XML file.
        for_sentences (bool): If True, enables sentence splitting and abbreviation detection.
    Returns:
        list: A list of sentences extracted from the XML file.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    for document in root.findall('document'):
        all_passages = []       # passages along with their offsets
        all_sentences = []      # sentences from all passages
        abbreviations_dict = {}  # abbreviations from all passages

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
            if section_type_element is not None and section_type_element.text.upper() not in ['ABSTRACT', 'INTRO', 'RESULTS', 'DISCUSS', 'CONCL']:
                continue
            type_element = passage.find("infon[@key='type']")
            # Adjust this to only include the relevant types
            if type_element is not None and type_element.text.upper() not in ['ABSTRACT', 'PARAGRAPH']:
                continue

            # all passages have a text element. Save it into the list so we can batch process it
            text_element = passage.find('text')
            offset_element = passage.find('offset')
            if text_element is not None and offset_element is not None:
                all_passages.append((text_element.text, (int(offset_element.text))))
            else: 
                continue

        # use nlp.pipe with batch processing and using multi core (multi core is only support if you dont need abbreviation)
        # for transformers, leave n_process at 1. For non-transformers, use any nproc processes
        if for_sentences:
            # if torch.cuda.is_available():
            #     logger.info(f"Using GPU ({torch.cuda.get_device_name(0)}) for processing with {torch.cuda.device_count()} CUDA devices.")
            #     nlp = spacy.load("en_core_sci_scibert", disable=["tagger", "ner", "lemmatizer"]) # just needed parser
            # else:
            logger.info("Using CPU for processing...")
            nlp = spacy.load("en_core_sci_lg", disable=["tagger", "ner", "lemmatizer"])
            nlp.add_pipe("abbreviation_detector")

            # extracting only the text from all_passages for nlp.pipe

            for doc, passage_offset in nlp.pipe(all_passages, as_tuples=True, n_process=1, batch_size=(min(len(all_passages), 128))):
                for sent in doc.sents:
                    sentence_offset = passage_offset + sent.start_char
                    all_sentences.append((sent.text, sentence_offset))
                for abrv in doc._.abbreviations:
                    abbreviations_dict[abrv.text] = str(abrv._.long_form)

    return all_passages, all_sentences, abbreviations_dict
