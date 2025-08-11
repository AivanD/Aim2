import logging
import os
import logging
import warnings
import json

from aim2.xml.xml_parser import parse_xml
from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, MODELS_DIR
from aim2.utils.logging_cfg import setup_logging
from aim2.llm.models import load_openai_model, load_local_model_via_outlines, load_local_model_via_outlinesVLLM
from aim2.llm.prompt import make_prompt
from aim2.entities_types.entities import CustomExtractedEntities
from aim2.postprocessing.span_adder import add_spans_to_entities

warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

def main():
    ensure_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)

    # load the model to use
    try:
        model = load_openai_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    logger.info("Starting the XML processing...")
    # process each files in the input folder
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

            # parse the XML file to get the list of passages and sentences. Set True for sentences
            passages, sentences = parse_xml(input_path, False)

            # print the number of passages and sentences found
            logger.info(f"Processed {len(passages)} passages and {len(sentences)} sentences from {filename}")

            # process each passage. processing each sentence would be costly. 
            for passage_text, passage_offset in passages:
                # create a prompt for the passage
                prompt = make_prompt(passage_text)
                prompts.append(prompt)

                # another option: use sentences instead of passages
                # for sentence in sentences:
                #     prompt = make_prompt(sentence)
                #     prompts.append(prompt)

                # Issues: can't batch inference with a GPT model because the model is not local. Their webpage batching is different as well (has 24hr turnover).
                # Issues: cant static batch with local model unless you can fit the overhead in vram. Sol: use vllm for continuous batching
                # Issues: GPT doesn't like normal Pydantic BaseModel. Use schemic
                # openai_schema = CustomExtractedEntities.schemic_schema()

                # this inference doesnt use batching. CHATGPT API is fast enough
                result = model(
                    model_input=prompt,
                    output_type=CustomExtractedEntities, # not supported for OPENAI. Just pass the schema through response.
                    # response_format=openai_schema,
                    # max_tokens=512,  # switch to max_tokens if using gpt
                    temperature=0.1,  # adjust as needed
                )

                # parse the json_string result into a pydantic object
                # TODO: add custom validators in entities.py later to ensure outputs are aligning to what is expected ESPECIALLY FOR LITERALS.
                extracted_entities = CustomExtractedEntities.model_validate_json(result)
                
                # add the spans to the extracted entities
                result_with_spans = add_spans_to_entities(extracted_entities, passage_text, passage_offset)
                result_list.append(result_with_spans)

            # TODO: Currently, each passage have their own set of extracted entities so the JSON has x items (x = # of passages)
            # implement a way to dedupe and merge all extracted entites. Thus this will give us the extracted entities for the whole paper
            # rather than for each passage. 

            # save the results to the output file
            with open(output_path, 'w') as f:
                json.dump(result_list, f, indent=2)
            logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
    