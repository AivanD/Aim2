import sys
from outlines import from_transformers, from_openai, from_vllm_offline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import torch
import openai
import vllm 
from groq import APIStatusError, Groq, AsyncGroq
import time
from pydantic import ValidationError
import re
import asyncio
from sentence_transformers import SentenceTransformer

from aim2.utils.config import MODELS_DIR, HF_TOKEN, OPENAI_API_KEY, GROQ_API_KEY, GROQ_MODEL
from aim2.entities_types.entities import CustomExtractedEntities
from aim2.llm.prompt import _static_header, make_prompt, _static_header_re

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
def load_local_model_via_outlinesVLLM():
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
    model_name = "kosbu/Llama-3.3-70B-Instruct-AWQ"
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    try:
        model = from_vllm_offline(vllm.LLM(
            model=model_name,                   
            quantization="awq_marlin",              # for quantized models using awq
            # quantization="bitsandbytes",          # for quantized models using bnb
            download_dir=str(MODELS_DIR),       
            # enforce_eager=True,                   # Recommended for use with guided_decoding_backend=outlines. Also, this skips CUDA GRAPH
            seed=42,
            swap_space=2,                           # defaults to 4. Uses ram for swapping data if things like kv_cache cant fit in vram. !MIGHT REDUCE PERF
            gpu_memory_utilization=0.92,            # adjust this for your usecase (default=.9 and .85 is enough for 8gb gpu)
            max_model_len=2048,                     # adjust this for your usecase (calc your prompt) (1024 is enough for 8gb gpu)
            # guided_decoding_backend="outlines",   # dont use as it gives empty output let it use default xgrammar
            # kv_cache_dtype="fp8_e4m3"             # uses V0 engine. DO NOT USE! BUGGY!
            kv_cache_memory_bytes=2 * 1024 * 1024 * 1024  # 2GB for kv cache
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

def load_openai_model():
    """
    Loads and returns an OpenAI language model instance using the specified API key and model name.
    Returns:
        An instance of the OpenAI language model configured with the provided API key and model name.
    """
    try: 
        model = from_openai(
            openai.AsyncOpenAI(api_key=OPENAI_API_KEY),
            model_name="gpt-4.1"       # Replace with a more powerful model if needed
        )
    except Exception as e:
        raise RuntimeError(f"Error loading local model via outlines VLLM: {e}")

    return model

client = Groq(
    api_key=GROQ_API_KEY)

async_client = AsyncGroq(
    api_key=GROQ_API_KEY
)

async def groq_inference_async(body, task=None):
    MAX_RETRIES = 10
    if task == "ner":
        system_content = _static_header()
    elif task == "re":
        system_content = _static_header_re()
    else:
        system_content = _static_header_re()        
    
    for attempt in range(MAX_RETRIES):
        try: 
            response = await async_client.chat.completions.create(
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
            await asyncio.sleep(0.3)  # brief pause before returning the response
            # if no exception, break the loop and return the response
            break
        except Exception as e:
            # API errors
            if isinstance(e, APIStatusError):
                if e.status_code == 400 and "json_validate_failed" in str(e.message):
                    if attempt < MAX_RETRIES - 1:
                        print(f"Retrying API call. Attempt {attempt + 1} of {MAX_RETRIES} due to JSON validation error.")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print(f"JSON validation error on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
                elif e.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = _parse_retry_after(str(e.message))
                        print(f"Rate limit exceeded. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after a delay of {wait_time}s.")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
                if e.status_code == 503:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Service unavailable. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after 1200s.")
                        await asyncio.sleep(1200)
                        continue
                    else:
                        print(f"Service unavailable on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
            # some other error
            else:
                print(f"An unexpected error occurred: {e}. Exiting.")
                sys.exit(1)
    
    return response.choices[0].message.content


def groq_inference(body, task=None):
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
                        print(f"Retrying API call. Attempt {attempt + 1} of {MAX_RETRIES} due to JSON validation error.")
                        time.sleep(1)
                        continue
                    else:
                        print(f"JSON validation error on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
                elif e.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = _parse_retry_after(str(e.message))
                        print(f"Rate limit exceeded. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after a delay of {wait_time}s.")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
                if e.status_code == 503:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Service unavailable. Attempt {attempt + 1} of {MAX_RETRIES}. Retrying after 1200s.")
                        time.sleep(1200)
                        continue
                    else:
                        print(f"Service unavailable on last attempt ({attempt + 1}/{MAX_RETRIES}). Error: {e}. Exiting.")
                        sys.exit(1)
            # some other error
            else:
                print(f"An unexpected error occurred: {e}. Exiting.")
                sys.exit(1)
    
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