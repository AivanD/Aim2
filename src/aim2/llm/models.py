from outlines import from_transformers, from_openai, from_vllm_offline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import torch
import openai
import vllm 

from aim2.utils.config import MODELS_DIR, HF_TOKEN, OPENAI_API_KEY
from aim2.entities_types.entities import CustomExtractedEntities

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
    model_name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
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
            gpu_memory_utilization=0.85,            # adjust this for your usecase (default=.9 and .85 is enough for 8gb gpu)
            max_model_len=1048,                     # adjust this for your usecase (calc your prompt) (1024 is enough for 8gb gpu)
            # guided_decoding_backend="outlines",   # dont use as it gives empty output let it use default xgrammar
            # kv_cache_dtype="fp8_e4m3"             # uses V0 engine. V1 is faster but resort to V0 if V1 doesnt work
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
            openai.OpenAI(api_key=OPENAI_API_KEY),
            model_name="gpt-4.1"       # Replace with a more powerful model if needed
        )
    except Exception as e:
        raise RuntimeError(f"Error loading local model via outlines VLLM: {e}")

    return model
