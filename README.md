# AIM 2

# Installation instructions
```bash
uv venv --python 3.11 --seed -n .cspirit # create virtual environment using uv (or use your own method)
pip install -e . # important

pip install vllm==0.11.0 
pip install scispacy==0.6.2	# ignore the warning about opencv-python-headless
pip install python-dotenv
pip install sentence_transformers
pip install schemic
pip install outlines==1.2.8
pip install groq
# for models used for sentence splitting (can be use for other purposes). 
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz # for non-transformer model 
```

# API tokens
use the `.env.example` to create a `.env` file to hold your API keys.
# How to run (4 relevant steps)
1. Place your input files in `input/` folder. The input files should be in BioC XML format.
2. Run the pipeline using the following command:
```bash
cd scripts/
python cache_ontology_embeddings.py # only need to run this once to cache the ontology embeddings. 
python run.py
```
3. The output will be in `output/` folder. The output folder will have `ner` and `re` subfolders. 
- the `ner` folder will have subfolders (used as checkpoints):
	- `raw` - raw outputs from the LLM
	- `evaluation` - results that is used for evaluation (JSON format)
	- `annotated` -  xlsx files with entities annotated in the text.
	- `processed` - json files with entities and their offsets. This will be used as input to the relation extraction step.
- the `re` folder will have subfolders (used as checkpoints):
	- `raw` - raw outputs from the LLM
	- `annotated` -  xlsx files with relations annotated in the text.
	- `processed` - json files with relations for pairs.

# other notes
`src/` folder contains the implementation of the pipeline. `script/` folder contains the scripts to run the pipeline. There are five main scrips:
- cache_ontology_embeddings.py - script to cache the ontology embeddings. Only need to run this once.
- evaluate_ner.py - script to evaluate the NER results.
- evaluate_relations.py	 - script to evaluate the relation extraction results.
- run_ner.py - script to run the NER pipeline.
- self_evaluate_re.py - script to run the relation extraction pipeline (run.py has this built in but if you want to run your own and separate, use this).




