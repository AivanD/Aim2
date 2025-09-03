import logging
import os
import logging
import warnings
import json
from vllm import SamplingParams
import time

from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, PO_OBO, PECO_OBO, TO_OBO, GO_OBO
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_openai_model, load_local_model_via_outlines, load_local_model_via_outlinesVLLM
from aim2.llm.prompt import make_prompt
from aim2.entities_types.entities import CustomExtractedEntities
from aim2.postprocessing.span_adder import add_spans_to_entities
from aim2.data.ontology import load_ontology

warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

def main():
    ensure_dirs()
    setup_logging()
    
    logger = logging.getLogger(__name__)

    # load the model to use
    try:
        model = load_openai_model()
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
    except Exception as e:
        logger.error(f"Error loading ontology: {e}")

    logger.info("Starting the XML processing...")
    # process each files in the input folder
    start_time = time.time()
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.xml'):
            # define the input file and output file
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename.replace('.xml', '.json'))

            # define a prompt list for batching
            prompts = []

            # define a list of results
            result_list = []

            # log the processing of the file
            logger.info(f"Processing file: {filename}")

            # parse the XML file to get the list of passages w/ offsets and sentences. Set True for sentences and abbreviations
            passages_w_offsets, sentences_w_offsets, abbreviations = parse_xml(input_path, False)

            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages_w_offsets)} passages and {len(sentences_w_offsets)} sentences from {filename}")

            # process each passage. processing each sentence would be costly.
            for passage_text, passage_offset in passages_w_offsets:
                # create a prompt for the passage
                prompt = make_prompt(passage_text)
                prompts.append(prompt)

                # another option: use sentences instead of passages
                # for sentence_text, sentence_offset in sentences_w_offsets:
                #     prompt = make_prompt(sentence)
                #     prompts.append(prompt)

                # Issues: can't batch inference with a GPT model because the model is not local. Their webpage batching is different as well (has 24hr turnover).
                # Issues: cant static batch with local model unless you can fit the overhead in vram. Sol: use vllm for continuous batching
                # Issues: GPT doesn't like normal Pydantic BaseModel. Use schemic
                openai_schema = CustomExtractedEntities.schemic_schema()

                # # this inference doesnt use batching. CHATGPT API is fast enough
                result = model(
                    model_input=prompt,
                    # output_type=CustomExtractedEntities, # not supported for OPENAI. Just pass the openai_schema through response_format.
                    response_format=openai_schema,
                    max_tokens=1024,  # switch to max_tokens if using gpt. otherwise use <max_new_tokens>
                    temperature=0.1,  # adjust as needed
                )

                # parse the json_string result into a pydantic object
                # TODO: add custom validators in entities.py later to ensure outputs are aligning to what is expected ESPECIALLY FOR LITERALS.
                extracted_entities = CustomExtractedEntities.model_validate_json(result)
                
                # add the spans to the extracted entities
                result_with_spans = add_spans_to_entities(extracted_entities, passage_text, passage_offset)
                result_list.append(result_with_spans)

            # # VLLM batching. Disable the other one
            # results = model.batch(
            #     model_input=prompts,
            #     output_type=CustomExtractedEntities,
            #     sampling_params=SamplingParams(temperature=0.1, max_tokens=1024),
            # )

            # validate each one
            # for result in results:
            #     extracted_entities = CustomExtractedEntities.model_validate_json(result[0])
            #     # add the spans to the extracted entities
            #     result_with_spans = add_spans_to_entities(extracted_entities, sentence_text, sentence_offset)
            #     result_list.append(result_with_spans)

            # TODO: Currently, each passage have their own set of extracted entities so the JSON has x items (x = # of passages)
            # implement a way to normalize, dedupe and merge all extracted entites. Thus this will give us the extracted entities for the whole paper
            # rather than for each passage. 
            # do normalization, then dedupe then merge

            # save the results to the output file
            with open(output_path, 'w') as f:
                json.dump(result_list, f, indent=2)
            logger.info(f"Results saved to {output_path}")
    end_time = time.time()
    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    