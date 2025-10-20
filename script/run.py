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

from aim2.postprocessing.compound_normalizer import classify_with_classyfire_local, get_np_class, normalize_compounds_with_pubchem
from aim2.postprocessing.merger import merge_and_deduplicate
from aim2.postprocessing.ontology_normalizer import SapbertNormalizer
from aim2.postprocessing.species_normalizer import normalize_species_with_ncbi
from aim2.preprocessing.pairing import find_entity_pairs, rank_passages_for_pair
from aim2.relation_types.relations import ExtractedRelations, Relation, SimpleRelation
from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import ensure_dirs, INPUT_DIR, PO_OBO, PECO_OBO, TO_OBO, GO_OBO, CHEMONT_OBO, RAW_NER_OUTPUT_DIR, EVAL_NER_OUTPUT_DIR, PROCESSED_NER_OUTPUT_DIR, RE_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_sapbert, groq_inference, groq_inference_async, load_openai_model, load_local_model_via_outlines, load_local_model_via_outlinesVLLM
from aim2.llm.prompt import make_prompt, make_re_prompt, make_re_prompt_body_only
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

async def process_passage_for_ner(semaphore, body, model=None):
    """Helper function to process a single passage with semaphore and retry logic."""
    async with semaphore:
        for attempt in range(5):  # Retry up to 5 times
            try:
                # # OPTION 1: OPENAI inference
                # prompt = make_prompt(body)
                # openai_schema = SimpleExtractedEntities().schemic_schema()
                # result = await model(
                #     model_input=prompt,
                #     response_format=openai_schema,
                #     max_tokens=2048,
                #     temperature=1e-67,
                # )
                # await asyncio.sleep(0.5)
                # return result

                # OPTION 2: GROQ inference (async)
                result = await groq_inference_async(body, task='ner')
                await asyncio.sleep(0.5)
                return result
            
            except RateLimitError as e:
                wait_time = _parse_openai_retry_after(str(e))
                logging.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/5)")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing a passage: {e}")
                return None # Or handle as appropriate

        logging.error("Passage failed after multiple retries due to rate limiting.")
        return None

async def process_pair_for_re(semaphore, body, model=None):
    """Helper function to process a single entity pair with semaphore and retry logic."""
    async with semaphore:
        for attempt in range(5):
            try:
                # # OPTION 1: OPENAI inference
                prompt = make_re_prompt(body[0], body[1], body[2], body[3])
                openai_schema = SimpleRelation.schemic_schema()
                result = await model(
                    model_input=prompt,
                    response_format=openai_schema,
                    max_tokens=256, # Smaller max tokens for this focused task
                    temperature=1e-67,
                )
                await asyncio.sleep(0.5)
                # OPTION 2: GROQ inference (async)
                # prompt_body = make_re_prompt_body_only(body[0], body[1], body[2], body[3])
                # result = await groq_inference_async(prompt_body, task='re')
                return result
            except RateLimitError as e:
                wait_time = _parse_openai_retry_after(str(e))
                logging.warning(f"RE Rate limit hit. Retrying in {wait_time:.2f}s (Attempt {attempt + 1}/5)")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logging.error(f"An unexpected error occurred during relation extraction for a pair: {e}")
                return None
        logging.error("Relation extraction for a pair failed after multiple retries.")
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
        
    # Define custom thresholds for normalization 
    # TODO: finetune these
    normalization_thresholds = {
        "compounds": 0.90,
        "pathways": 0.80,
        "anatomical_structures": 0.90,
        "plant_traits": 0.75, # Plant traits can be more descriptive and varied
        "molecular_traits": 0.85,
        "experimental_conditions": 0.88,
    }

    # Initialize the SapbertNormalizaer with the model and custom thresholds
    try: 
        normalizer = SapbertNormalizer(sapbert_model, thresholds=normalization_thresholds)
    except Exception as e:
        logger.error(f"Error initializing SapbertNormalizer: {e}")
        return
    
    # load the plant-ontology, peco, and trait_ontology (for future use)
    # try:
    #     # "plant ontology" which has 2 namespaces: 'plant_anatomy' and 'plant_structure_development_stage'
    #     plant_terms_dict, po_graph = load_ontology(PO_OBO)
    #     logger.info(f"Plant ontology loaded successfully from {PO_OBO}.")
    #     # "experimental condition" ontology which has 1 namespace: 'plant_experimental_conditions_ontology'
    #     peco_terms_dict, peco_graph = load_ontology(PECO_OBO)
    #     logger.info(f"PECO ontology loaded successfully from {PECO_OBO}.")
    #     # "plant trait" ontology which has 1 namespace: 'plant_trait_ontology'
    #     to_terms_dict, to_graph = load_ontology(TO_OBO)
    #     logger.info(f"Trait ontology loaded successfully from {TO_OBO}.")
    #     # "gene ontology" ontology which has 3 namespaces: molecular function, biological process, cellular component
    #     go_terms_dict, go_graph = load_ontology(GO_OBO)
    #     logger.info(f"Whole Gene Ontology loaded successfully from {GO_OBO}.")
    #     # "chemical ontology" ontology which has 1 namespace: 'chemont'
    #     chemont_terms_dict, chemont_graph = load_ontology(CHEMONT_OBO)
    #     logger.info(f"ChemOnt loaded successfully from {CHEMONT_OBO}.")
        
    #     ontologies = {
    #         "po_graph": po_graph,
    #         "peco_graph": peco_graph,
    #         "to_graph": to_graph,
    #         "go_graph": go_graph,
    #         "chemont_graph": chemont_graph
    #     }

    # except Exception as e:
    #     logger.error(f"Error loading ontology: {e}")

    logger.info("Starting the XML processing...")
    # process each files in the input folder
    start_time = time.time()
    for filename in os.listdir(INPUT_DIR):
        start_ner_time = time.time()
        if filename.endswith('.xml'):
            # define the input file and output file
            input_path = os.path.join(INPUT_DIR, filename)
            raw_ner_output_path = os.path.join(RAW_NER_OUTPUT_DIR, filename.replace('.xml', '.json'))
            eval_ner_output_path = os.path.join(EVAL_NER_OUTPUT_DIR, filename.replace('.xml', '.json')) # New path for evaluation file
            processed_ner_output_path = os.path.join(PROCESSED_NER_OUTPUT_DIR, filename.replace('.xml', '.json'))

            # define a prompt list for batching
            prompts_ner = []

            # log the processing of the file
            logger.info(f"Processing file: {filename}")

            # parse the XML file to get the list of passages w/ offsets and sentences. Set True for sentences and abbreviations
            passages_w_offsets, sentences_w_offsets, abbreviations = parse_xml(input_path, False)

            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages_w_offsets)} passages and {len(sentences_w_offsets)} sentences from {filename}")

            # process each passage. processing each sentence would be costly.
            # only process if there is no raw output file yet
            if not os.path.exists(raw_ner_output_path):   
                # limit concurrency to 3 requests at a time. Adjust as needed. 1 = sequential
                semaphore = asyncio.Semaphore(3)
                tasks = []      # for async api calls
                # define a list of results (raw results from the model)
                raw_result_list = [] 
                # for sentence_text, sentence_offset in sentences_w_offsets:
                for passage_text, passage_offset in passages_w_offsets:
                    # create a prompt for the passage and add to the list (for local inference)
                    prompt = make_prompt(passage_text)
                    prompts_ner.append(prompt)

                    # API (async)
                    task = process_passage_for_ner(semaphore, passage_text, model)
                    tasks.append(task)

                # wait for all tasks to complete and get their results
                logger.info(f"Waiting for {len(tasks)} tasks to complete...")
                results = await asyncio.gather(*tasks)

                # # OPTION 3: LOCAL MODEL via outlines+VLLM (batching)
                # results = model.batch(
                #     model_input=prompts_ner,
                #     output_type=SimpleExtractedEntities,
                #     sampling_params=SamplingParams(temperature=1e-67, max_tokens=1024),
                # )

                for result in results:
                    if result is None:
                        continue  # Skip if there was an error processing this passage
                    # parse the json_string result into a pydantic object
                    # TODO: add custom validators in entities.py later to ensure outputs are aligning to what is expected ESPECIALLY FOR LITERALS.
                    extracted_entities = SimpleExtractedEntities().model_validate_json(result)      # if using OPENAI or GROQ
                    # extracted_entities = SimpleExtractedEntities().model_validate_json(result[0])  # if using local model

                    # add the entities to the raw result list to be saved into a file
                    raw_result_list.append(extracted_entities.model_dump())

                # save the results to the output file
                with open(raw_ner_output_path, 'w') as f:
                    json.dump(raw_result_list, f, indent=2)
                logger.info(f"Raw results saved to {raw_ner_output_path}")
                end_ner_time = time.time()
                logger.info(f"Ner processing time for {filename}: {end_ner_time - start_ner_time:.2f} seconds")

            # Post-processing:
            # TODO: Currently, each passage have their own set of extracted entities so the JSON has x items (x = # of passages)
            # implement a way to normalize, dedupe and merge all extracted entites. Thus this will give us the extracted entities for the whole paper
            # rather than for each passage. 
            # do span_adder, normalization, then dedupe then merge
            if not os.path.exists(processed_ner_output_path):
                start_ner_post_time = time.time()

                processed_result_list = []

                # 1. add spans to each of the extracted entities in the raw_result_list
                # read the raw results from the raw output file
                with open(raw_ner_output_path, 'r') as f:
                    raw_result_list = json.load(f)

                # ensure valid objects (just in case user edited the raw json file)
                for raw_result, (passage_text, passage_offset) in zip(raw_result_list, passages_w_offsets):
                    try:
                        extracted_entities = CustomExtractedEntities.model_validate(raw_result)
                    except Exception as e:
                        logger.error(f"Invalid raw result object in {raw_ner_output_path}: {e}")
            
                    # add spans to each of the extracted entities in the raw_result_list
                    extracted_entities_w_spans = add_spans_to_entities(extracted_entities, passage_text, passage_offset)
                    processed_result_list.append(extracted_entities_w_spans)

                # --- SAVE INTERMEDIATE FILE FOR NER EVALUATION if it doesnt exists ---
                if not os.path.exists(eval_ner_output_path):
                    with open(eval_ner_output_path, 'w') as f:
                        json.dump(processed_result_list, f, indent=2)
                logger.info(f"NER evaluation file saved to {eval_ner_output_path}")

                # 2. normalize
                # - use ChemOnt to normalize compounds first for classes
                # - use PUBCHEM to normalize compounds  for molecular compounds
                # - use NP_CLASSIFIER to get the superclass/class
                # - use existing ontology for pathways, molecular traits, plant traits, anatomical structures, experimental conditions
                # - use NCBI tax for species.
                # - human traits are not normalized
                # - genes are not normalized

                try:
                    # 1. Normalize against ontologies first. This will classify compound classes via ChemOnt.
                    processed_result_list = normalizer.normalize_entities(processed_result_list)

                    # 2. For compounds NOT classified by ChemOnt, try to find a CID and SMILES via PubChem.
                    processed_result_list = normalize_compounds_with_pubchem(processed_result_list)

                    # 3. For compounds with a SMILES and InChIkey strings, get further classification.
                    processed_result_list = get_np_class(processed_result_list)
                    processed_result_list = classify_with_classyfire_local(processed_result_list)

                    # 4. For species, use NCBI taxonomy to get the taxon id
                    processed_result_list = normalize_species_with_ncbi(processed_result_list)

                except Exception as e:
                    logger.error(f"An unexpected error occurred while normalizing: {e}")
                
                # 3. Deduplicate and merge entities for the entire document
                try: 
                    final_entities = merge_and_deduplicate(processed_result_list)
                except Exception as e:
                    logger.error(f"An unexpected error occurred while merging: {e}")
                
                # save the processed results to the output file
                with open(processed_ner_output_path, 'w') as f:
                    json.dump(final_entities, f, indent=2)
                logger.info(f"Processed results saved to {processed_ner_output_path}")
                end_ner_post_time = time.time()
                logger.info(f"Ner post-processing time for {filename}: {end_ner_post_time - start_ner_post_time:.2f} seconds")

            # Relation Extraction
            start_re_time = time.time()
            re_output_path = os.path.join(RE_OUTPUT_DIR, filename.replace('.xml', '.json'))
            final_entities = None
            if not os.path.exists(re_output_path):
                # Run relation extraction if there is no RE output file yet using the entities from NER
                with open(processed_ner_output_path, 'r') as f:
                    final_entities = json.load(f)
                
                # 1. Find entity pairs that co-occur in the paragraph
                entity_pairs = find_entity_pairs(final_entities)
                if not entity_pairs:
                    logger.info(f"No entity pairs found in {filename}. Skipping relation extraction.")
                    continue

                # Initialize a list to hold all extracted relations
                tasks = []
                pair_details = [] # To hold details needed after async calls
                # prompts list for offline batching
                prompts_re = []

                # Limit concurrency to 3 to avoid rate limits.
                re_semaphore = asyncio.Semaphore(3)

                # 2. filter and rank
                for compound, other_entity, category in entity_pairs:
                    ranked_passages = rank_passages_for_pair(compound, other_entity, passages_w_offsets)

                    if not ranked_passages:
                        continue

                    # 3. Select top 1-3 passages as context
                    top_passages_text = [p[0] for p in ranked_passages[:3]]
                    context_str = "\n\n---\n\n".join(top_passages_text)
                    
                    # prompt_re = make_re_prompt(compound, other_entity, category, top_passages_text)
                    # prompts_re.append(prompt_re)
                    pair_details.append({"compound": compound, "other_entity": other_entity, "context": context_str})

                    # API (async). set Model = none for Groq
                    task = process_pair_for_re(re_semaphore, (compound, other_entity, category, top_passages_text), model)
                    tasks.append(task)
                
                # Execute all API calls concurrently
                logger.info(f"Starting relation extraction for {len(tasks)} pairs...")
                re_results = await asyncio.gather(*tasks)

                # # OPTION 3: LOCAL MODEL via outlines+VLLM (batching)
                # re_results = model.batch(
                #     model_input=prompts_re,
                #     output_type=SimpleRelation,
                #     sampling_params=SamplingParams(temperature=1e-67, max_tokens=1024),
                # )

                # 4. Process results and build final relation objects
                all_relations = ExtractedRelations()
                for i, result_json in enumerate(re_results):
                    if result_json is None:
                        continue
                    
                    try:
                        # simple_relation = SimpleRelation.model_validate_json(result_json[0])   # if using local model
                        simple_relation = SimpleRelation.model_validate_json(result_json)
                        if simple_relation.predicate == "No_Relationship":
                            continue

                        details = pair_details[i]
                        full_relation = Relation(
                            subject=details["compound"],
                            object=details["other_entity"],
                            predicate=simple_relation.predicate,
                            justification=simple_relation.justification,
                            context=details["context"]
                        )
                        all_relations.relations.append(full_relation)
                    except Exception as e:
                        logger.error(f"Failed to validate or process RE result: {e}\nResult was: {result_json}")
            
                # 5. Save all found relations to a file
                with open(re_output_path, 'w') as f:
                    f.write(all_relations.model_dump_json(indent=2))
                logger.info(f"Saved {len(all_relations.relations)} relations to {re_output_path}")

            end_re_time = time.time()
            logger.info(f"Relation extraction time for {filename}: {end_re_time - start_re_time:.2f} seconds")  

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(amain())
    