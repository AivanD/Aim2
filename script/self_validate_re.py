import logging
import os
import json
import asyncio
import argparse
import time

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import vllm.reasoning 

from aim2.relation_types.relations import ExtractedRelations, ValidationResult
from aim2.utils.config import RAW_RE_OUTPUT_DIR, PROCESSED_RE_OUTPUT_DIR, ensure_dirs
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_local_model_via_outlinesVLLM, load_openai_model
from aim2.llm.prompt import make_re_evaluation_prompt

async def amain():
    """
    Main asynchronous function to run the relation extraction validation process.
    """
    # file name input
    # user_input = input("Enter the filename of the raw RE output to validate (e.g., 'sample_re_output.json'): ")
    user_input = "PMC8164654.json"
    filename = user_input.strip()

    ensure_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)

    # --- File Paths ---
    raw_re_output_path = os.path.join(RAW_RE_OUTPUT_DIR, filename)
    processed_re_output_path = os.path.join(PROCESSED_RE_OUTPUT_DIR, filename)

    if not os.path.exists(raw_re_output_path):
        logger.error(f"Input file not found: {raw_re_output_path}")
        return

    if os.path.exists(processed_re_output_path):
        logger.warning(f"Processed file already exists: {processed_re_output_path}. Skipping validation.")
        return

    # --- Load Model ---
    try:
        model = load_local_model_via_outlinesVLLM("RedHatAI/gemma-3-27b-it-FP8-dynamic", quantization=None)
        logger.info("Using local model for validation.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- Run Validation ---
    start_time = time.time()
    logger.info(f"Starting self-validation for {filename}...")

    with open(raw_re_output_path, 'r') as f:
        raw_relations_str = f.read()

    try:
        raw_relations = ExtractedRelations.model_validate_json(raw_relations_str)
    except Exception as e:
        logger.error(f"Invalid raw relations object in {raw_re_output_path}: {e}")
        return

    relations_to_validate = raw_relations.relations
    if not relations_to_validate:
        logger.info("No relations to validate.")
        return

    prompts_re_validation = [make_re_evaluation_prompt(rel.model_dump()) for rel in relations_to_validate]

    logger.info(f"Starting validation for {len(prompts_re_validation)} relations...")

    # Using local model with batching
    structured_re_validation_params = StructuredOutputsParams(json=ValidationResult.model_json_schema())
    validation_results = model.batch(
        model_input=prompts_re_validation,
        sampling_params=SamplingParams(temperature=1e-67, seed=42, structured_outputs=structured_re_validation_params),
    )

    validated_relations = ExtractedRelations()
    not_validated_relations = ExtractedRelations()

    for i, val_result_json in enumerate(validation_results):
        try:
            # The result from model.batch is a list of strings, get the first one.
            validation = ValidationResult.model_validate_json(val_result_json[0])
            if validation.decision == "yes":
                validated_relations.relations.append(relations_to_validate[i])
            else:
                not_validated_relations.relations.append(relations_to_validate[i])
        except Exception as e:
            logger.error(f"Failed to process validation result: {e}\nResult was: {val_result_json}")

    # --- Save Results ---
    with open(processed_re_output_path, 'w') as f:
        f.write(validated_relations.model_dump_json(indent=2))
    logger.info(f"Saved {len(validated_relations.relations)} validated relations to {processed_re_output_path}")

    if not_validated_relations.relations:
        processed_re_output_path_not_valid = processed_re_output_path.replace('.json', '_not_validated.json')
        with open(processed_re_output_path_not_valid, 'w') as f:
            f.write(not_validated_relations.model_dump_json(indent=2))
        logger.info(f"Saved {len(not_validated_relations.relations)} not validated relations to {processed_re_output_path_not_valid}")

    end_time = time.time()
    logger.info(f"Total validation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(amain())