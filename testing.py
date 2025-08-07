import outlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from aim2.utils.config import ensure_dirs, INPUT_DIR, OUTPUT_DIR, MODELS_DIR, ensure_dirs
from pydantic import BaseModel
from typing import List
from outlines import Generator
ensure_dirs()

class Character(BaseModel):
    name: str
    age: int
    skills: List[str]

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
        )

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", 
                                         cache_dir=str(MODELS_DIR), 
                                         local_files_only=True, 
                                         quantization_config=quantization_config,
                                         device_map="cuda"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                   cache_dir=str(MODELS_DIR),
                                   local_files_only=True
                                   )
)
# generator = Generator(model, Character)
# result = generator("Create a character.")
result = model("Create a character.", Character)
print(result) # '{"name": "Evelyn", "age": 34, "skills": ["archery", "stealth", "alchemy"]}'
print(Character.model_validate_json(result)) # name=Evelyn, age=34, skills=['archery', 'stealth', 'alchemy']