import logging
import os
import logging
import warnings
from pydantic import BaseModel
import torch

from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, MODELS_DIR
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_openai_model, load_local_model_via_outlines
from aim2.llm.prompt import make_prompt
from aim2.entities_types.entities import CustomExtractedEntities

warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

class Country(BaseModel):
    name: str
    capital: str

def main():
    ensure_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)

    # load the model to use
    model = load_local_model_via_outlines()
    logger.info("Model loaded successfully.")

    logger.info("Starting the XML processing...")

    # process each files in the input folder
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.xml'):
            # define the input file and output file
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename.replace('.xml', '.json'))

            # define a prompt list for batching
            prompts = []

            # log the processing of the file
            logger.info(f"Processing file: {filename}")

            # parse the XML file to get the list of passages and sentences. True for sentences
            passages, sentences = parse_xml(input_path, False)

            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages)} passages and {len(sentences)} sentences from {filename}")

            # process each passage
            for passage in passages:
                # create a prompt for the passage
                prompt = make_prompt(passage)
                prompts.append(prompt)

            # # save the results to the output file
            # with open(output_path, 'w') as f:
            #     for result in results:
            #         f.write(result.json(indent=2) + '\n')
            # logger.info(f"Results saved to {output_path}")

                # generate entities for each prompt


if __name__ == "__main__":
    main()
    