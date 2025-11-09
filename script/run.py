import logging
import os
import logging
import sys
import warnings
import json
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import time
import asyncio
from tqdm.asyncio import tqdm
from openai import RateLimitError
import re
import pydantic

from aim2.postprocessing.compound_normalizer import classify_with_classyfire_local, get_np_class, normalize_compounds_with_pubchem
from aim2.postprocessing.merger import merge_and_deduplicate, merge_entities_by_abbreviation
from aim2.postprocessing.ontology_normalizer import SapbertNormalizer
from aim2.postprocessing.species_normalizer import normalize_species_with_ncbi
from aim2.preprocessing.pairing import find_entity_pairs, rank_passages_for_pair_enhanced, select_best_sentences_from_paragraphs
from aim2.relation_types.relations import ExtractedRelations, Relation, SimpleRelation, SelfEvaluationResult
from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import GPT_MODEL_NER, GPT_MODEL_RE_EVAL, GROQ_MODEL, GROQ_MODEL_RE_EVAL, PROCESSED_RE_OUTPUT_DIR, RAW_RE_OUTPUT_DIR, ensure_dirs, INPUT_DIR, PO_OBO, PECO_OBO, TO_OBO, GO_OBO, CHEMONT_OBO, RAW_NER_OUTPUT_DIR, EVAL_NER_OUTPUT_DIR, PROCESSED_NER_OUTPUT_DIR, RE_OUTPUT_DIR
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_sapbert, groq_inference, groq_inference_async, load_openai_client_async, load_groq_client_async, load_local_model_via_outlines, load_local_model_via_outlinesVLLM, gpt_inference_async
from aim2.llm.prompt import make_prompt, make_re_evaluation_prompt_body_only, make_re_prompt, make_re_evaluation_prompt, make_prompt_body_only, make_re_prompt_body_only
from aim2.entities_types.entities import CustomExtractedEntities, SimpleExtractedEntities
from aim2.postprocessing.span_adder import add_spans_to_entities
from aim2.data.ontology import load_ontology

warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

async def process_passage_for_ner(semaphore, body, client):
    """Helper function to process a single passage with semaphore and retry logic."""
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                prompt_body = make_prompt_body_only(body)      # body for NER that can be used by openai or groq
                # OPTION 1: OPENAI inference
                result = await gpt_inference_async(
                    client,
                    body=prompt_body,
                    task='ner',
                    API_MODEL=GPT_MODEL_NER,
                    json_object=SimpleExtractedEntities
                )

                # OPTION 2: GROQ inference (async)
                # does not support "json_object" param for Llama 3.3 or older
                # result = await groq_inference_async(
                #     client,
                #     body=prompt_body,
                #     API_MODEL=GROQ_MODEL,
                #     task='ner', 
                #     json_object=SimpleExtractedEntities
                # )

                # Validate the result immediately.
                validated_result = SimpleExtractedEntities.model_validate_json(result)
                return result
            except(pydantic.ValidationError) as e:
                logging.warning(f"Validation failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}. Retrying in {RETRY_DELAY}s...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logging.error(f"Failed to get a valid NER result after {MAX_RETRIES} attempts.")
                    return None # Give up after final attempt
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing a passage: {e}")
                return None # Or handle as appropriate

# TODO: add retry logic when validation error happens
async def process_pair_for_re(semaphore, body, client):
    """Helper function to process a single entity pair with semaphore and retry logic."""
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                prompt_body = make_re_prompt_body_only(body[0], body[1], body[2], body[3])
                # OPTION 1: OPENAI inference
                # result = await gpt_inference_async(
                #     client,
                #     body=prompt_body,
                #     task='re',
                #     API_MODEL=GPT_MODEL_NER,
                #     json_object=SimpleRelation
                # )
                # return result

                # # OPTION 2: GROQ inference (async)
                result = await groq_inference_async(
                    client, 
                    body=prompt_body, 
                    task='re', 
                    API_MODEL=GROQ_MODEL,
                    json_object=SimpleRelation
                )

                # Validate the result immediately.
                validated_result = SimpleRelation.model_validate_json(result)

                return result
            except(pydantic.ValidationError) as e:
                logging.warning(f"Validation failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}. Retrying in {RETRY_DELAY}s...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logging.error(f"Failed to get a valid relation extraction result after {MAX_RETRIES} attempts.")
                    return None # Give up after final attempt
            except Exception as e:
                logging.error(f"An unexpected error occurred during relation extraction for a pair: {e}")
                return None

async def process_for_re_evaluation(semaphore, body, client):
    MAX_RETRIES = 5
    RETRY_DELAY = 5  # seconds
    
    async with semaphore:
        for attempt in range(5):
            try:
                prompt_body = make_re_evaluation_prompt_body_only(body)
                # OPTION 1: OPENAI inference
                result = await gpt_inference_async(
                    client,
                    body=prompt_body,
                    task='re-self-eval',
                    API_MODEL="gpt-5-mini",
                    json_object=SelfEvaluationResult
                )                        
                # # OPTION 2: GROQ inference (async)
                # result = await groq_inference_async(
                #     client,
                #     body=prompt_body, 
                #     task='re-self-eval', 
                #     API_MODEL="openai/gpt-oss-120b",
                #     json_object=SelfEvaluationResult
                # )

                # Validate the result immediately.
                validated_result = SelfEvaluationResult.model_validate_json(result)

                return result 
            except(pydantic.ValidationError) as e:
                logging.warning(f"Validation failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}. Retrying in {RETRY_DELAY}s...")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logging.error(f"Failed to get a valid self-evaluation result after {MAX_RETRIES} attempts.")
                    return None # Give up after final attempt
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing a passage: {e}")
                return None # Or handle as appropriate

async def amain():
    ensure_dirs()
    setup_logging()
    
    logger = logging.getLogger(__name__)

    # load the models to use
    try:
        sapbert_model = load_sapbert()
        # for NER, RE, EVAL
        OPENAI_client = load_openai_client_async()             
        GROQ_client = load_groq_client_async()     
        
        # for local RE  
        model = load_local_model_via_outlinesVLLM()
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

            # log the processing of the file
            logger.info(f"Processing file: {filename}")
            parsing_time = time.time()
            # parse the XML file to get the list of passages w/ offsets and sentences. Set True for sentences and abbreviations
            passages_w_offsets, sentences_w_offsets, abbreviations = parse_xml(input_path, True)
            logger.info(f"Parsing time for {filename}: {time.time() - parsing_time:.2f} seconds")
            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages_w_offsets)} passages and {len(sentences_w_offsets)} sentences from {filename}")

            # process each passage. processing each sentence would be costly.
            # only process if there is no raw output file yet
            if not os.path.exists(raw_ner_output_path):   
                # limit concurrency to 3 requests at a time. Adjust as needed. 1 = sequential
                semaphore = asyncio.Semaphore(3)
                # define a prompt list for batching
                prompts_ner = []
                tasks = []      # for async api calls
                # define a list of results (raw results from the model)
                raw_result_list = [] 
                
                # for sentence_text, sentence_offset in sentences_w_offsets:
                for passage_text, passage_offset in passages_w_offsets:
                    # create a prompt for the passage and add to the list (for local inference)
                    prompt = make_prompt(passage_text)
                    prompts_ner.append(prompt)

                    # OPTION 1: API (async)
                    task = process_passage_for_ner(semaphore, passage_text, OPENAI_client)
                    tasks.append(task)

                # wait for all tasks to complete and get their results
                logger.info(f"Waiting for {len(tasks)} tasks to complete...")
                results = await tqdm.gather(*tasks, desc="Processing passages")

                # # OPTION 2: LOCAL MODEL via outlines+VLLM (batching)
                # comment the API async part as well as the "results await line above" before using local inference
                # structured_ner_output_params = StructuredOutputsParams(
                #     json=SimpleExtractedEntities.model_json_schema()
                # )
                # results = model.batch(
                #     model_input=prompts_ner,
                #     # output_type=SimpleExtractedEntities,
                #     sampling_params=SamplingParams(temperature=1e-67, seed=42, max_tokens=1024, structured_outputs=structured_ner_output_params),
                # )

                for i, result in enumerate(results):
                    if result is None:
                        logger.error(f"Skipping passage {i} due to previous errors.")
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
                    processed_result_list = merge_and_deduplicate(processed_result_list)
                    # 4. Further merge entities based on abbreviation mapping from the XML
                    final_entities = merge_entities_by_abbreviation(processed_result_list, abbreviations)
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
            raw_re_output_path = os.path.join(RAW_RE_OUTPUT_DIR, filename.replace('.xml', '.json'))
            processed_re_output_path = os.path.join(PROCESSED_RE_OUTPUT_DIR, filename.replace('.xml', '.json'))
            final_entities = None
            
            if not os.path.exists(raw_re_output_path):
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
                    # stage 1: Rank top paragraphs for the entity pair
                    top_paragraphs = rank_passages_for_pair_enhanced(
                        compound, other_entity, passages_w_offsets, granularity="paragraph", top_k=2
                    )

                    if not top_paragraphs:
                        continue

                    # Stage 2: Select best sentences from those top paragraphs
                    best_sentences = select_best_sentences_from_paragraphs(
                        compound, other_entity, top_paragraphs, sentences_w_offsets, per_paragraph=2
                    )

                    # 3. Use the text from the best sentences as context for relation extraction
                    # If sentence selection gives good results, use it.
                    # Otherwise, fall back to using the entire text of the single best paragraph.
                    if best_sentences:
                        # 3.1 context window construction with surrounding sentences.
                        # Create a quick lookup from offset to index for all sentences
                        offset_to_index = {offset: i for i, (text, offset) in enumerate(sentences_w_offsets)}
                        
                        # Use a set to automatically handle duplicates and maintain order
                        context_indices = set()
                        
                        for s_text, s_start, score, diag in best_sentences:
                            if s_start in offset_to_index:
                                best_sent_idx = offset_to_index[s_start]
                                # Add the sentence before (if it exists)
                                if best_sent_idx > 0:
                                    context_indices.add(best_sent_idx - 1)
                                # Add the best sentence itself
                                context_indices.add(best_sent_idx)
                                # Add the sentence after (if it exists)
                                if best_sent_idx < len(sentences_w_offsets) - 1:
                                    context_indices.add(best_sent_idx + 1)
                        
                        # Sort indices and build the final context string
                        sorted_indices = sorted(list(context_indices))
                        context_texts = [sentences_w_offsets[i][0] for i in sorted_indices]
                        context_str = "\n".join(context_texts)
                    else:
                        # Fallback to the top-ranked paragraph's full text
                        # logger.warning(f"No single sentence found with both entities for pair ({compound['name']}, {other_entity['name']}). Falling back to top paragraph context.")
                        # Option 1: take all k top paragraphs' text
                        context_texts = [p[0] for p in top_paragraphs]
                        context_str = "\n".join(context_texts)

                        # Option 2: take only the text of the top paragraph
                        # top_paragraph_text = top_paragraphs[0][0]
                        # context_texts = [top_paragraph_text]
                        # context_str = top_paragraph_text

                    # If context is still empty, skip
                    if not context_str.strip():
                        continue

                    prompt_re = make_re_prompt(compound, other_entity, category, context_texts)
                    prompts_re.append(prompt_re)
                    pair_details.append({"compound": compound, "other_entity": other_entity, "category": category, "context": context_str})

                    # # OPTION 1: API (async)
                    # task = process_pair_for_re(re_semaphore, (compound, other_entity, category, context_texts), GROQ_client)
                    # tasks.append(task)
                
                # Execute all API calls concurrently
                logger.info(f"Starting relation extraction for {len(prompts_re)} **VALID** pairs...")
                # re_results = await tqdm.gather(*tasks, desc="Extracting relations")

                # OPTION 2: LOCAL MODEL via outlines+VLLM (batching)
                # comment the API async part as well as the "re_results await line above" before using local inference
                structured_re_output_params = StructuredOutputsParams(
                    json=SimpleRelation.model_json_schema(),
                )
                re_results = model.batch(
                    model_input=prompts_re,
                    # output_type=SimpleRelation,
                    sampling_params=SamplingParams(temperature=1e-67,
                                                   seed=42,
                                                   max_tokens=512, 
                                                   structured_outputs=structured_re_output_params
                                                   ),
                )

                # 4. Process results and build final relation objects
                all_relations = ExtractedRelations()
                all_no_relations = ExtractedRelations()     # DEBUG

                for i, result_json in enumerate(re_results):
                    if result_json is None:
                        logger.error(f"Skipping passage {i} due to previous errors.")
                        continue
                    
                    try:
                        simple_relation = SimpleRelation.model_validate_json(result_json[0])   # if using local model
                        # simple_relation = SimpleRelation.model_validate_json(result_json)
                        if simple_relation.predicate == "No_Relationship":
                            details = pair_details[i]
                            full_relation = Relation(
                                subject_entity=details["compound"],
                                object_entity=details["other_entity"],
                                predicate=simple_relation.predicate,
                                category=details["category"],
                                justification=simple_relation.justification,
                                context=details["context"]
                            )
                            all_no_relations.relations.append(full_relation)
                        else:
                            details = pair_details[i]
                            full_relation = Relation(
                                subject_entity=details["compound"],
                                object_entity=details["other_entity"],
                                predicate=simple_relation.predicate,
                                category=details["category"],
                                justification=simple_relation.justification,
                                context=details["context"]
                            )
                            all_relations.relations.append(full_relation)
                    except Exception as e:
                        logger.error(f"Failed to validate or process RE result: {e}\nResult was: {result_json}")
            
                # 5. Save all found relations to a file
                with open(raw_re_output_path, 'w') as f:
                    f.write(all_relations.model_dump_json(indent=2))
                re_no_output_path = raw_re_output_path.replace('.json', '_no_relationships.json')
                with open(re_no_output_path, 'w') as f:
                    f.write(all_no_relations.model_dump_json(indent=2))

                logger.info(f"Saved {len(all_relations.relations)} relations to {raw_re_output_path}")
            
            end_re_time = time.time()
            logger.info(f"Relation extraction time for {filename}: {end_re_time - start_re_time:.2f} seconds")  

            # Self-evaluation step for RE outputs
            start_re_eval_time = time.time()

            if not os.path.exists(processed_re_output_path):
                logger.info(f"Starting self-evaluation for {filename}...")
                with open(raw_re_output_path, 'r') as f:
                    raw_relations_str = f.read()
        
                # quick self-evaluation of the inputs
                try:
                    raw_relations = ExtractedRelations.model_validate_json(raw_relations_str)
                except Exception as e:
                    logger.error(f"Invalid raw relations object in {raw_re_output_path}: {e}")
                    continue

                semaphore_self_evaluation = asyncio.Semaphore(15)
                tasks = []
                prompts_re_evaluation = []

                relations_to_evaluate = raw_relations.relations
                for rel in relations_to_evaluate:
                    prompt = make_re_evaluation_prompt(rel.model_dump())
                    prompts_re_evaluation.append(prompt)

                    # OPTION 1:API (async)
                    task = process_for_re_evaluation(semaphore_self_evaluation, rel.model_dump(), OPENAI_client)  
                    tasks.append(task)

                # Wait for all tasks to complete and get their results
                logger.info(f"Waiting for {len(tasks)} self-evaluation tasks to complete...")
                self_evaluation_results = await tqdm.gather(*tasks, desc="Self-evaluating relations")
                
                # OPTION 2: LOCAL MODEL via outlines+VLLM (batching)
                # structured_re_evaluation_params = StructuredOutputsParams(json=SelfEvaluationResult.model_json_schema())
                # self_evaluation_results = model.batch(
                #     model_input=prompts_re_evaluation,
                #     sampling_params=SamplingParams(temperature=1e-67, seed=42, max_tokens=32, structured_outputs=structured_re_evaluation_params),
                # )

                evaluated_relations = ExtractedRelations()
                not_evaluated_relations = ExtractedRelations()

                for i, self_eval_result_json in enumerate(self_evaluation_results):
                    try:
                        if self_eval_result_json is None:
                            logger.error(f"Self-evaluation for relation index {i} failed after retries.")
                            continue

                        # self_evaluation = SelfEvaluationResult.model_validate_json(val_result_json[0])   # if using local model
                        self_evaluation = SelfEvaluationResult.model_validate_json(self_eval_result_json)
                        if self_evaluation.decision == "yes":
                            evaluated_relations.relations.append(relations_to_evaluate[i])
                        else:
                            not_evaluated_relations.relations.append(relations_to_evaluate[i])
                    except Exception as e:
                        logger.error(f"Failed to process self-evaluation result: {e}\nResult was: {self_eval_result_json}")
                with open(processed_re_output_path, 'w') as f:
                    f.write(evaluated_relations.model_dump_json(indent=2))
                logger.info(f"Saved {len(evaluated_relations.relations)} evaluated relations to {processed_re_output_path}")
                processed_re_output_path_not_evaluated = processed_re_output_path.replace('.json', '_not_evaluated.json')
                with open(processed_re_output_path_not_evaluated, 'w') as f:
                    f.write(not_evaluated_relations.model_dump_json(indent=2))
                logger.info(f"Saved {len(not_evaluated_relations.relations)} not evaluated relations to {processed_re_output_path_not_evaluated}")

            end_re_eval_time = time.time()
            logger.info(f"Relation evaluation time for {filename}: {end_re_eval_time - start_re_eval_time:.2f} seconds")
            
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(amain())
    