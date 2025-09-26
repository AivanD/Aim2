import logging
import os
import logging
import warnings
import json
from vllm import SamplingParams
import time
import asyncio
from openai import RateLimitError
import re

from aim2.postprocessing.compound_normalizer import get_np_class, normalize_compounds_with_pubchem
from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, PO_OBO, PECO_OBO, TO_OBO, GO_OBO, RAW_OUTPUT_DIR, PROCESSED_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_sapbert, groq_inference, groq_inference_async, load_openai_model, load_local_model_via_outlines, load_local_model_via_outlinesVLLM
from aim2.llm.prompt import make_prompt
from aim2.entities_types.entities import CustomExtractedEntities, SimpleExtractedEntities
from aim2.postprocessing.span_adder import add_spans_to_entities
from aim2.data.ontology import load_ontology

warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

def _parse_openai_retry_after(error_message: str) -> float:
    """Parses the retry-after time from an OpenAI API error message."""
    match = re.search(r"Please try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1)) + 0.1  # Add a small buffer
    return 10.0  # Default to 10 seconds if parsing fails

async def process_passage(semaphore, passage_text, model=None):
    """Helper function to process a single passage with semaphore and retry logic."""
    async with semaphore:
        for attempt in range(5):  # Retry up to 5 times
            try:
                # OPTION 1: OPENAI inference
                prompt = make_prompt(passage_text)
                openai_schema = SimpleExtractedEntities().schemic_schema()
                result = await model(
                    model_input=prompt,
                    response_format=openai_schema,
                    max_tokens=1024,
                    temperature=1e-67,
                )
                return result

                # OPTION 2: GROQ inference (async)
                # prompt = make_prompt(passage_text)
                # result = await groq_inference_async(prompt)
                # return result
            
            except RateLimitError as e:
                wait_time = _parse_openai_retry_after(str(e))
                logging.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/5)")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing a passage: {e}")
                return None # Or handle as appropriate

        logging.error("Passage failed after multiple retries due to rate limiting.")
        return None

async def amain():
    ensure_dirs()
    setup_logging()
    
    logger = logging.getLogger(__name__)

    # load the models to use
    try:
        sapbert_model = load_sapbert()
        model = load_openai_model()     # for OPENAI or Local model
        # model = load_local_model_via_outlinesVLLM()
        logger.info(f"Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # load the plant-ontology, peco, and trait_ontology (for future use)
    try:
        # "plant ontology" which has 2 namespaces: 'plant_anatomy' and 'plant_structure_development_stage'
        plant_terms_dict, po_graph = load_ontology(PO_OBO)
        logger.info(f"Plant ontology loaded successfully from {PO_OBO}.")
        # "experimental condition" ontology which has 1 namespace: 'plant_experimental_conditions_ontology'
        peco_terms_dict, peco_graph = load_ontology(PECO_OBO)
        logger.info(f"PECO ontology loaded successfully from {PECO_OBO}.")
        # "plant trait" ontology which has 1 namespace: 'plant_trait_ontology'
        to_terms_dict, to_graph = load_ontology(TO_OBO)
        logger.info(f"Trait ontology loaded successfully from {TO_OBO}.")
        # "gene ontology" ontology which has 3 namespaces: molecular function, biological process, cellular component
        go_terms_dict, go_graph = load_ontology(GO_OBO)
        logger.info(f"Whole Gene Ontology loaded successfully from {GO_OBO}.")
    
        ontologies = {
            "po_graph": po_graph,
            "peco_graph": peco_graph,
            "to_graph": to_graph,
            "go_graph": go_graph,
        }

    except Exception as e:
        logger.error(f"Error loading ontology: {e}")

    logger.info("Starting the XML processing...")
    # process each files in the input folder
    start_time = time.time()
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.xml'):
            # define the input file and output file
            input_path = os.path.join(INPUT_DIR, filename)
            raw_output_path = os.path.join(RAW_OUTPUT_DIR, filename.replace('.xml', '.json'))
            processed_output_path = os.path.join(PROCESSED_OUTPUT_DIR, filename.replace('.xml', '.json'))
            output_path = os.path.join(OUTPUT_DIR, filename.replace('.xml', '.json'))

            # define a prompt list for batching
            prompts = []

            # log the processing of the file
            logger.info(f"Processing file: {filename}")

            # parse the XML file to get the list of passages w/ offsets and sentences. Set True for sentences and abbreviations
            passages_w_offsets, sentences_w_offsets, abbreviations = parse_xml(input_path, False)

            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages_w_offsets)} passages and {len(sentences_w_offsets)} sentences from {filename}")

            # process each passage. processing each sentence would be costly.
            # only process if there is no raw output file yet
            if not os.path.exists(raw_output_path):   
                # limit concurrency to 3 requests at a time. Adjust as needed. 1 = sequential
                semaphore = asyncio.Semaphore(1)
                tasks = []      # for async api calls
                # define a list of results (raw results from the model)
                raw_result_list = [] 
                for passage_text, passage_offset in passages_w_offsets:
                    # create a prompt for the passage
                    prompt = make_prompt(passage_text)
                    prompts.append(prompt)

                    # API (async)
                    task = process_passage(semaphore, passage_text, model)
                    tasks.append(task)

                # wait for all tasks to complete and get their results
                logger.info(f"Waiting for {len(tasks)} tasks to complete...")
                results = await asyncio.gather(*tasks)

                for result in results:
                    if result is None:
                        continue  # Skip if there was an error processing this passage
                    # parse the json_string result into a pydantic object
                    # TODO: add custom validators in entities.py later to ensure outputs are aligning to what is expected ESPECIALLY FOR LITERALS.
                    extracted_entities = SimpleExtractedEntities().model_validate_json(result)

                    # add the entities to the raw result list to be saved into a file
                    raw_result_list.append(extracted_entities.model_dump())
                
                # OPTION 3: LOCAL MODEL via outlines+VLLM (batching)
                # results = model.batch(
                #     model_input=prompts,
                #     output_type=CustomExtractedEntities,
                #     sampling_params=SamplingParams(temperature=1e-67, max_tokens=1024),
                # )

                # validate each one
                # for result in results:
                #     extracted_entities = CustomExtractedEntities.model_validate_json(result[0])
                #     # add the entities to the raw result list to be saved into a file
                #     raw_result_list.append(extracted_entities.model_dump())

                # save the results to the output file
                with open(raw_output_path, 'w') as f:
                    json.dump(raw_result_list, f, indent=2)
                logger.info(f"Raw results saved to {raw_output_path}")

            # Post-processing:
            # TODO: Currently, each passage have their own set of extracted entities so the JSON has x items (x = # of passages)
            # implement a way to normalize, dedupe and merge all extracted entites. Thus this will give us the extracted entities for the whole paper
            # rather than for each passage. 
            # do span_adder, normalization, then dedupe then merge

            processed_result_list = []

            # 1. add spans to each of the extracted entities in the raw_result_list
            # read the raw results from the raw output file
            with open(raw_output_path, 'r') as f:
                raw_result_list = json.load(f)

            # ensure valid objects (just in case user edited the raw json file)
            for raw_result, (passage_text, passage_offset) in zip(raw_result_list, passages_w_offsets):
                try:
                    extracted_entities = CustomExtractedEntities.model_validate(raw_result)
                except Exception as e:
                    logger.error(f"Invalid raw result object in {raw_output_path}: {e}")
        
                # add spans to each of the extracted entities in the raw_result_list
                extracted_entities_w_spans = add_spans_to_entities(extracted_entities, passage_text, passage_offset)
                processed_result_list.append(extracted_entities_w_spans)

            # 2. normalize
            # - use ChemOnt to normalize compounds first for classes
            # - use PUBCHEM to normalize compounds  for molecular compounds
            # - use NP_CLASSIFIER to get the superclass/class

            try:
                processed_result_list = normalize_compounds_with_pubchem(processed_result_list)
                processed_result_list = get_np_class(processed_result_list)
            except Exception as e:
                logger.error(f"An unexpected error occurred while normalizing compounds with PubChem: {e}")
            # 3. dedupe
            # 4. merge
            
            # save the processed results to the output file
            with open(processed_output_path, 'w') as f:
                json.dump(processed_result_list, f, indent=2)
            logger.info(f"Processed results saved to {processed_output_path}")

    end_time = time.time()
    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(amain())
    