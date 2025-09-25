import argparse
import os
import logging
import sys

# Add the project root to the Python path to allow importing from 'aim2'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from aim2.xml.xml_parser import parse_xml
    from aim2.utils.config import INPUT_DIR
    from aim2.utils.logging_cfg import setup_logging
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Make sure the project is installed correctly (`pip install -e .`). Details: {e}")
    sys.exit(1)

def find_sentence_for_span(sentences_w_offsets: list, start_offset: int, end_offset: int):
    """
    Finds and prints the sentence that contains the given character span from a pre-parsed list.

    Args:
        sentences_w_offsets (list): A list of tuples, where each tuple is (sentence_text, sentence_start_offset).
        start_offset (int): The starting character offset of the span.
        end_offset (int): The ending character offset of the span.
    """
    logger = logging.getLogger(__name__)
    found_sentence = False
    for sentence_text, sentence_start in sentences_w_offsets:
        sentence_end = sentence_start + len(sentence_text)
        # Check if the provided span is fully contained within the current sentence's span
        if sentence_start <= start_offset and end_offset <= sentence_end:
            print("\n--- Found Sentence ---")
            print(sentence_text)
            print(f"Span ({start_offset}, {end_offset}) is within sentence bounds ({sentence_start}, {sentence_end}).")
            found_sentence = True
            break

    if not found_sentence:
        logger.warning(f"Could not find a sentence containing the span ({start_offset}, {end_offset}).")

def main():
    """Main function to parse arguments and run the script."""
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Find sentences for given word/spans from an XML file in the input directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("filename", type=str, help="The name of the .xml file in the 'input/' directory.")
    
    args = parser.parse_args()
    
    input_path = os.path.join(INPUT_DIR, args.filename)

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Parsing '{args.filename}' to find sentences... This may take a moment.")
    # The parse_xml function with for_sentences=True handles sentence segmentation and provides global offsets.
    _, sentences_w_offsets, _ = parse_xml(input_path, for_sentences=True)

    if not sentences_w_offsets:
        logger.warning("No sentences were extracted from the document.")
        return
        
    logger.info("Parsing complete. You can now enter spans to find sentences.")

    while True:
        try:
            user_input = input("\nEnter start and end offsets (e.g., '5533 5559') or type 'exit' to quit: ").strip()
            if user_input.lower() == 'exit':
                break

            parts = user_input.split()
            if len(parts) != 2:
                logger.error("Invalid input. Please provide two numbers separated by a space.")
                continue

            start = int(parts[0])
            end = int(parts[1])

            find_sentence_for_span(sentences_w_offsets, start, end)

        except ValueError:
            logger.error("Invalid input. Please enter numeric offsets.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()