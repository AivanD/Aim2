import sys
from outlines import from_transformers, from_openai, from_vllm_offline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import torch
import openai
import vllm 
from groq import APIStatusError, Groq, AsyncGroq
from openai import RateLimitError
import time
import re
import asyncio
from sentence_transformers import SentenceTransformer
import logging

from aim2.utils.config import MODELS_DIR, HF_TOKEN, OPENAI_API_KEY, GROQ_API_KEY, GROQ_MODEL, GPT_MODEL_NER, GPT_MODEL_RE_EVAL
from aim2.entities_types.entities import CustomExtractedEntities
from aim2.llm.prompt import _static_header, _static_header_re_evaluation, make_prompt, _static_header_re

logger = logging.getLogger(__name__)

def _get_retry_after(response) -> int:
    """Extracts the 'Retry-After' duration from an HTTP response object."""
    retry_after_header = response.headers.get("retry-after")
    if retry_after_header is not None:
        try:
            # The header value is usually in seconds.
            wait_time = int(float(retry_after_header))
            return wait_time + 2  # Add a 2-second buffer to be safe
        except ValueError:
            pass
    return 60 # Default to 60 seconds

def _parse_retry_after(error_message: str) -> int:
    """Parses the retry-after time from a Groq API error message."""
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", error_message)
    if not match:
        return 1200  # Default to 20 minutes if parsing fails

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0
    
    wait_time = hours * 3600 + minutes * 60 + seconds
    return int(wait_time) + 5 # Add a 5-second buffer

# use vllm for concurrency, pageattention, kv_caching, etc.
def load_local_model_via_outlinesVLLM(model_name="kosbu/Llama-3.3-70B-Instruct-AWQ", max_model_len=2500, quantization="awq_marlin"):
    """
    Loads a local (w or w/o quant) LLM model using the Outlines library with vLLM backend.
    This function initializes a model as well as configures memory and performance options 
    suitable for local inference, and returns the loaded model.
    Returns:
        model: The loaded local LLM model instance.
    Notes:
        - Adjust `gpu_memory_utilization` and `max_model_len` for your GPU capacity.
        - Uses swap space to handle memory overflow situations.
        - Recommended to avoid guided decoding backend "outlines" due to empty outputs.
        - Seed is set for reproducibility.
    """
    # model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    # model_name = "kosbu/Llama-3.3-70B-Instruct-AWQ"
    # model_name = "gaunernst/gemma-3-27b-it-int4-awq"
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    try:
        model = from_vllm_offline(vllm.LLM(
            model=model_name,                   
            quantization=quantization,              # for quantized models using awq
            # quantization="bitsandbytes",          # for quantized models using bnb
            download_dir=str(MODELS_DIR),       
            enforce_eager=False,                   # Keep False for better performance
            seed=42,
            swap_space=4,                           # defaults to 4. Uses ram for swapping data if things like kv_cache cant fit in vram. !MIGHT REDUCE PERF
            enable_prefix_caching=True,             # speeds up generation when prompt is long
            gpu_memory_utilization=0.90,            # adjust this for your usecase (default=.9 and .85 is enough for 8gb gpu)
            max_model_len=max_model_len,            # adjust this for your usecase (calc your prompt) (1024 is enough for 8gb gpu)
            max_num_batched_tokens=4096*4,            # this is for batching multiple requests. adjust based on your gpu memory
            # guided_decoding_backend="outlines",   # dont use as it gives empty output let it use default xgrammar
            # kv_cache_dtype="fp8_e4m3"             # uses V0 engine. DO NOT USE! BUGGY!
            # kv_cache_memory_bytes=3 * 1024 * 1024 * 1024  # 3GB for kv cache
        ))
    except ValueError as e:
        raise ValueError(f"Error loading local model via outlines VLLM: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading local model via outlines VLLM: {e}")
    
    return model

def load_local_model_via_outlines():
    """
    Loads a local (w or w/o quant) LLM model using the Outlines library with Transformers backend. 
    Returns:
        model: The loaded local LLM model instance.
    """
    return DeprecationWarning("Loading local model via outlines Transformers is deprecated. Please use 'load_local_model_via_outlinesVLLM' instead.")
    # Initialize a model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
        )
    model_name = "meta-llama/Llama-3.1-8B-Instruct"     # not only is it weak but it quantized so, it may not perform well on complex tasks
    try:
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
    except Exception as e:
        raise RuntimeError(f"Error loading local model via outlines VLLM: {e}")

    return model

def load_openai_client_sync():
    """
    Initializes and returns an OpenAI client using the provided API key.
    Returns:
        An instance of the OpenAI client configured with the provided API key.
    """
    return DeprecationWarning("Synchronous OpenAI client is deprecated. Please use the asynchronous version 'load_openai_client_async' instead.")
    try: 
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing OpenAI client: {e}")

    return client

def load_openai_client_async():
    """
    Initializes and returns an OpenAI client using the provided API key.
    Returns:
        An instance of the OpenAI client configured with the provided API key.
    """
    try: 
        client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=0, # we handle retries ourselves
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing OpenAI client: {e}")

    return client

def load_groq_client_sync():
    """
    Initializes and returns a Groq client using the provided API key.
    Returns:
        An instance of the Groq client configured with the provided API key.
    """
    return DeprecationWarning("Synchronous Groq client is deprecated. Please use the asynchronous version 'load_groq_client_async' instead.")
    try: 
        client = Groq(
            api_key=GROQ_API_KEY
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing Groq client: {e}")

    return client

def load_groq_client_async():
    """
    Initializes and returns an Async Groq client using the provided API key.
    Returns:
        An instance of the Async Groq client configured with the provided API key.
    """
    try: 
        client = AsyncGroq(
            api_key=GROQ_API_KEY,
            max_retries=0,  # we handle retries ourselves
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing Async Groq client: {e}")

    return client

async def gpt_inference_async(client, body, task=None, API_MODEL=GPT_MODEL_NER, json_object=None):
    if API_MODEL == "gpt-4.1":
        temperature = 1e-67
        reasoning = None
        max_output_tokens = 2048
        if task == "ner":
            system_content = _static_header()
        elif task == "re":
            system_content = _static_header_re()
        else:
            raise Exception("GPT_MODEL_NER model can only be used for 'ner' or 're' tasks.")
    elif API_MODEL == "gpt-5-mini":
        temperature = None
        reasoning = {"effort": "low"}        # add "summary": auto if you want reasoning summaries to appear in the response. Organization needs to be verified via the website first
        max_output_tokens = 200
        if task == "re-self-eval":
            system_content = _static_header_re_evaluation()
        else:
            raise Exception("GPT_MODEL_RE_EVAL model can only be used for 're-self-eval' task.")
    else:
        raise Exception(f"Unsupported API_MODEL: {API_MODEL}")
    
    schema = None
    if json_object:
        schema = json_object.model_json_schema()
        # The API requires a 'required' field listing all properties if 'properties' is present.
        if "properties" in schema and "required" not in schema:
            schema["required"] = list(schema["properties"].keys())
        # The API requires 'additionalProperties' to be explicitly set to false.
        schema["additionalProperties"] = False

    params = {
        "model": API_MODEL,
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_content
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": body
                    }
                ]
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": json_object.__name__ if json_object else "output_schema",
                "schema": schema
            },
        },      
    }
    
    if temperature:
        params["temperature"] = temperature
    if reasoning:
        params["reasoning"] = reasoning
    if max_output_tokens:
        params["max_output_tokens"] = max_output_tokens

    MAX_RETRIES = 10
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.responses.create(**params)
            await asyncio.sleep(1)  # brief pause before returning the response
            # if no exception, break the loop and return the response
            break
        except RateLimitError as e: 
            if attempt < MAX_RETRIES - 1:
                wait_time = _get_retry_after(e.response)
                logger.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after a delay of {wait_time}s.")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Rate limit exceeded on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                raise e
        except openai.APIStatusError as e:
            if e.status_code == 400: # Bad request, potentially a schema issue
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API Error (400): {e}. Retrying... ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(5) # wait a bit before retrying
                    continue
                else:
                    logger.error(f"API Error (400) on last attempt. Error: {e}. Exiting.")
                    raise e
            elif e.status_code == 500 or e.status_code == 503: # Internal server error or service unavailable
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API Error ({e.status_code}): Service unavailable. Retrying after 20s... ({attempt + 1}/{MAX_RETRIES})")
                    await asyncio.sleep(20)
                    continue
                else:
                    logger.error(f"API Error ({e.status_code}) on last attempt. Error: {e}. Exiting.")
                    raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}. Exiting.")
            raise e
    return response.output_text

async def groq_inference_async(client, body, task=None, API_MODEL=GROQ_MODEL, json_object=None):
    MAX_RETRIES = 10

    if API_MODEL == "llama-3.3-70b-versatile":
        temperature = 1e-67
        max_completion_tokens = 2048
        seed = 42
        reasoning_effort_value = None
        response_format = {"type": "json_object"}
        if task == "ner":
            system_content = _static_header()
        elif task == "re":
            system_content = _static_header_re()
        else:
            raise Exception("GROQ_MODEL model can only be used for 'ner' or 're' tasks.")
    elif API_MODEL == "openai/gpt-oss-20b" or API_MODEL == "openai/gpt-oss-120b":
        temperature = 1e-67
        max_completion_tokens = 4096
        seed = 42
        reasoning_effort_value = "medium"
        if json_object:
            schema = json_object.model_json_schema()
            if "properties" in schema and "required" not in schema:
                schema["required"] = list(schema["properties"].keys())
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_object.__name__ if json_object else "output_schema",
                    "schema": schema
                }
            }
        else:
            response_format = {"type": "json_object"}
        if task == "re-self-eval":
            system_content = _static_header_re_evaluation()
        else:
            raise Exception("GROQ_MODEL_RE_EVAL model can only be used for 're-self-eval' task.")
    else:
        raise Exception(f"Unsupported API_MODEL: {API_MODEL}")
      
    param = {
        "model": API_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": body
            }
        ],
        "stream": False,
        "stop": None,
        "seed": seed
    }

    if temperature:
        param["temperature"] = temperature
    if max_completion_tokens:
        param["max_completion_tokens"] = max_completion_tokens
    if response_format:
        param["response_format"] = response_format
    if task == "re-self-eval":
        param["reasoning_effort"] = reasoning_effort_value

    for attempt in range(MAX_RETRIES):
        try: 
            response = await client.chat.completions.create(**param)
            await asyncio.sleep(1)  # brief pause before returning the response
            # if no exception, break the loop and return the response
            break
        except APIStatusError as e:
            if e.status_code == 400 and "json_validate_failed" in str(e.message):
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Retrying API call. Attempt {attempt + 1} of {MAX_RETRIES} due to JSON validation error.")
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"JSON validation error on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                    raise e
            elif e.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = _get_retry_after(e.response)
                    logger.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after a delay of {wait_time}s.")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                    raise e
            elif e.status_code == 503:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Service unavailable. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after 1200s.")
                    await asyncio.sleep(1200)
                    continue
                else:
                    logger.error(f"Service unavailable on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                    raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}. Exiting.")
            raise e
    
    return response.choices[0].message.content

# TODO: update the synchronous version to match the async one
# TODO: create a sync openai_inference function 
def groq_inference(client, body, task=None):
    return DeprecationWarning("Synchronous Groq inference is deprecated. Please use the asynchronous version 'groq_inference_async' instead.")
    MAX_RETRIES = 10
    if task == "ner":
        system_content = _static_header()
    elif task == "re":
        system_content = _static_header_re()
    else:
        # Default to NER system prompt if task is not specified
        system_content = _static_header() 
    
    for attempt in range(MAX_RETRIES):
        try: 
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": body
                }
                ],
                temperature=1e-67,
                max_completion_tokens=2048,
                stream=False,
                response_format={"type": "json_object"},
                stop=None,
                seed=42
            )
            time.sleep(0.3)  # brief pause before returning the response
            # if no exception, break the loop and return the response
            break
        except Exception as e:
            # API errors
            if isinstance(e, APIStatusError):
                if e.status_code == 400 and "json_validate_failed" in str(e.message):
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Retrying API call. Attempt {attempt + 1} of {MAX_RETRIES} due to JSON validation error.")
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"JSON validation error on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        raise e
                elif e.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = _get_retry_after(e.response)
                        logger.warning(f"Rate limit exceeded. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after a delay of {wait_time}s.")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        raise e
                if e.status_code == 503:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Service unavailable. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after 1200s.")
                        time.sleep(1200)
                        continue
                    else:
                        logger.error(f"Service unavailable on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        raise e
            # some other error
            else:
                logger.error(f"An unexpected error occurred: {e}. Exiting.")
                raise e
    
    return response.choices[0].message.content

def load_sapbert():
    """
    Loads and returns a SAPBERT model instance using the specified model name.
    Returns:
        An instance of the SAPBERT model configured with the provided model name.
    """
    try: 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(
            model_name_or_path="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", 
            device=device,
            cache_folder=str(MODELS_DIR),
            # local_files_only=True
        )
    except Exception as e:
        raise RuntimeError(f"Error loading SAPBERT model via outlines: {e}")

    return model