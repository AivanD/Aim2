from outlines import Generator, from_transformers, from_openai
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import torch
import openai
from vllm import LLM, SamplingParams

from aim2.utils.config import MODELS_DIR, HF_TOKEN, OPENAI_API_KEY
from aim2.entities_types.entities import CustomExtractedEntities
set_seed(42)

# TODO: Check if this actually works.
def load_local_model_via_outlinesVLLM():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = outlines.from_vllm_offline(LLM(
        model=model_name,
        # quantization="awq_marlin",
        quantization="bitsandbytes",
        download_dir=str(MODELS_DIR),
        # enforce_eager=True,  # Recommended for use with outlines
        seed=42,
        swap_space=2,
        gpu_memory_utilization=0.80,
        # max_model_len=4096,
        # guided_decoding_backend="outlines", # dont use as it gives empty output let it use default xgrammar
        kv_cache_dtype="fp8_e4m3"
    ))

    return model
def load_local_model_via_outlines():
    # Initialize a model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
        )
    model_name = "meta-llama/Llama-3.1-8B-Instruct"     # not only is it weak but it quantized so, it may not perform well on complex tasks
    model = from_transformers(
        AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR),
            device_map="auto",
            local_files_only=True,
            quantization_config=quantization_config,
            token=HF_TOKEN
        ),
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(MODELS_DIR),
            local_files_only=True,
        ),
    )

    return model

def load_openai_model():
    model = from_openai(
        openai.OpenAI(api_key=OPENAI_API_KEY),
        model_name="gpt-4.1"       # Replace with a more powerful model if needed
    )
    return model

def main():
    # Load the model
    model = load_openai_model()

    isdone = False

    while not isdone:
        # prompt the user for article text
        article_text = input("Enter the article text (or type 'exit' to quit): ")
        if article_text.lower() == 'exit':
            isdone = True
            continue

        # 3. Create a clear prompt for the model
        prompt = f"""
        Extract all entities from the following text based on the provided JSON schema. The entity types are: Compound, Stressor, Process, Trait, Stage, Structure, and Organoleptic.
        If no entities of a type are found, return an empty list for that type.

        Text:
        {article_text}
        """
        
        # Generate a response.
        result = model.batch(model_input=prompt,
                             output_type=CustomExtractedEntities,
                             max_new_tokens=200,       # not used in OpenAI models
                             temperature=0,
                             top_p=1)

        # Parse the JSON result into a Pydantic model
        try:
            extracted_entities = CustomExtractedEntities.model_validate_json(result)
            # get the JSON output
            json_output = extracted_entities.model_dump_json(indent=2)
            print(json_output)
        except Exception as e:
            print(f"Error parsing model output: {e}")
            print(f"Model returned: {result}")

if __name__ == "__main__":
    main()