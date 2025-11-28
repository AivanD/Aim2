import logging
import os
import json
import asyncio
import argparse
import time
from tqdm.asyncio import tqdm
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import vllm.reasoning 

from aim2.relation_types.relations import ExtractedRelations, SelfEvaluationResult
from aim2.utils.config import RAW_RE_OUTPUT_DIR, PROCESSED_RE_OUTPUT_DIR, ensure_dirs
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_groq_client_async, load_local_model_via_outlinesVLLM, load_openai_client_async, process_for_re_evaluation
from aim2.llm.prompt import make_re_evaluation_prompt

async def amain():
    """
    Main asynchronous function to run the relation extraction self-evaluation process.
    """
    ensure_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)

    # load model
    try:
        # specify local model_name if you want to use a different model than the default one (Llama 3.3 70B)
        model = load_local_model_via_outlinesVLLM()

        # or use API-based model. Dont forget to uncomment related code below "process_for_re_evaluation" and 
        # self_evaluation variable
        
        # OPENAI_client = load_openai_client_async()
        # GROQ_client = load_groq_client_async() 
        logger.info("loading model for self-evaluation.")
    except Exception as e:
        logger.error(f"Error loading model: {e}.")
        return 

    start_re_eval_time = time.time()
    logger.info("Starting self-evaluation for all files in raw_re_outputs...")

    for filename in os.listdir(RAW_RE_OUTPUT_DIR):
        if not filename.endswith('.json') or '_not_evaluated.json' in filename or '_no_relationships.json' in filename:
            continue

        # file path
        raw_re_output_path = os.path.join(RAW_RE_OUTPUT_DIR, filename)
        processed_re_output_path = os.path.join(PROCESSED_RE_OUTPUT_DIR, filename)
        processed_re_output_path_not_evaluated = processed_re_output_path.replace('.json', '_not_evaluated.json')

        # check if input file exists
        if not os.path.exists(raw_re_output_path):
            logger.error(f"Input file not found: {raw_re_output_path}")
            continue
        # skip if the processed file already exists
        if os.path.exists(processed_re_output_path):
            logger.warning(f"Processed file already exists: {processed_re_output_path}. Skipping self-evaluation.")
            continue
        else:
            # proceed with self-evaluation for this file
            logger.info(f"Self-evaluating file: {filename}")
            with open(raw_re_output_path, 'r') as f:
                    raw_relations_str = f.read()

            try: 
                raw_relations = ExtractedRelations.model_validate_json(raw_relations_str)
            except Exception as e:
                logger.error(f"Invalid raw relations object in {raw_re_output_path}: {e}")
                continue
            
            # if using API-based models, use semaphore to limit concurrency
            semaphore_self_evaluation = asyncio.Semaphore(20)
            tasks = []

            # a list of prompts for relation self-evaluation
            prompts_re_evaluation = []

            # get the relations to evaluate and create prompts
            relations_to_evaluate = raw_relations.relations
            for rel in relations_to_evaluate:
                prompt = make_re_evaluation_prompt(rel.model_dump())
                prompts_re_evaluation.append(prompt)

                # OPTION 1:API (async)
                # task = process_for_re_evaluation(semaphore_self_evaluation, rel.model_dump(), OPENAI_client)  
                # tasks.append(task)

            # Wait for all tasks to complete and get their results
            logger.info(f"Waiting for {len(prompts_re_evaluation)} self-evaluation tasks to complete...")
            # self_evaluation_results = await tqdm.gather(*tasks, desc="Self-evaluating relations")
            
            # OPTION 2: LOCAL MODEL via outlines+VLLM (batching)
            structured_re_evaluation_params = StructuredOutputsParams(json=SelfEvaluationResult.model_json_schema())
            self_evaluation_results = model.generate(
                prompts=prompts_re_evaluation,
                sampling_params=SamplingParams(temperature=1e-67, seed=42, max_tokens=32, structured_outputs=structured_re_evaluation_params),
            )

            evaluated_relations = ExtractedRelations()
            not_evaluated_relations = ExtractedRelations()

            for i, self_eval_result_json in enumerate(self_evaluation_results):
                try:
                    if self_eval_result_json is None:
                        logger.error(f"Self-evaluation for relation index {i} failed after retries.")
                        continue

                    self_evaluation = SelfEvaluationResult.model_validate_json(self_eval_result_json.outputs[0].text)  # if using local model
                    # self_evaluation = SelfEvaluationResult.model_validate_json(self_eval_result_json) # if using API-based model
                    if self_evaluation.decision == "yes":
                        evaluated_relations.relations.append(relations_to_evaluate[i])
                    else:
                        not_evaluated_relations.relations.append(relations_to_evaluate[i])
                except Exception as e:
                    logger.error(f"Failed to process self-evaluation result: {e}\nResult was: {self_eval_result_json}")
            
            # save evaluated relations to file
            with open(processed_re_output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluated_relations.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(evaluated_relations.relations)} evaluated relations to {processed_re_output_path}")
            # save not evaluated relations to a separate file
            with open(processed_re_output_path_not_evaluated, 'w', encoding='utf-8') as f:
                json.dump(not_evaluated_relations.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(not_evaluated_relations.relations)} not evaluated relations to {processed_re_output_path_not_evaluated}")

    end_re_eval_time = time.time()
    total_re_eval_time = end_re_eval_time - start_re_eval_time
    logger.info(f"Completed self-evaluation for all files in raw_re_outputs in {total_re_eval_time/60:.2f} minutes.")

if __name__ == "__main__":
    asyncio.run(amain())