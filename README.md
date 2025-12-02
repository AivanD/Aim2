# AIM 2

# Installation instructions
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh	# installs uv

# navigate to project folder
uv venv --python 3.11 --seed -n .cspirit # create virtual environment using uv (or use your own method)
source .cspirit/bin/activate 
pip install -r requirements_ln2.txt
```

# API tokens
use the `.env.example` to create a `.env` file to hold your API keys.
# How to run (4 relevant steps)
1. Place your input files in `input/` folder. The input files should be in BioC XML format.
2. Run the `cache_ontology_embeddings.py` script using the following command. This would create a .pkl file in the `data/` folder that would be used for entity linking after entity extraction but before relation extraction.
```bash
python scripts/cache_ontology_embeddings.py # only need to run this once to cache the ontology embeddings. 
```
3. Run the `pubchem_ncbi_db.py` script to setup the local PubChem and NCBI taxonomy databases. This is needed for normalizing compounds and species during post-processing of entity extraction just before relation extraction. This command will install all the necessary dependencies by download the required data like `CID-InChI-Key.gz`, `CID-SMILES.gz`, `CID-Synonym-filtered.gz` from PubChem and `taxdump.tar.gz` from NCBI taxonomy. This only needs to be run once and it will create a database (.sqlite) file in `data/` folder. Download time varies and the .sqlite creation takes ~5-10 minutes.  
```bash
python scripts/pubchem_ncbi_db.py  # only need to run this once to setup the local dbs.
```
### NOTE: NP classifier step (SMILES --> NP CLASS) is currently disabled since it relies on API but can be run separately on a separate script that I have to write still.
4. Run the `run.py` script to run the entire pipeline (NER + RE + Self-Eval) using the following command. This would process all the files in the `input/` folder and generate outputs in the `output/` folder.
```bash
python scripts/run.py
```

The output will be in `output/` folder. The output folder will have `ner` and `re` subfolders. 
- the `ner` folder will have subfolders (used as checkpoints):
	- `raw` - raw outputs from the LLM. Contains entites for each paragraph (paragraph level).
	- `evaluation` - results that is used for evaluation (JSON format). Contains entities with offsets/spans.
	- `annotated` -  xlsx files with entities annotated in the text. This is used for evaluating the files in `evaluation` folder. It is very strict in that it requires exact match of the entity text and offsets, not just the entity text.
	- `processed` - json files with entities, offsets and other metadata like NP class, Classyfire, etc for a given article (article level). This will be used as input to the relation extraction step.
- the `re` folder will have subfolders (used as checkpoints):
	- `raw` - raw outputs from the LLM. Contains relations for each pair of entities that have co-occurrence.
	- `annotated` -  xlsx files with relations annotated in the text for each pair of entities found for an article. It is very strict in that it requires exact match of triples (entity1, relation, entity2) even if relation can be more than one.
	- `processed` - json files with relations for pairs after self-evaluation step. This will be used for evaluation against the annotated files.

# other notes
`src/` folder contains the implementation of the pipeline. `script/` folder contains the scripts to run the pipeline. There are five main scrips:
- cache_ontology_embeddings.py - script to cache the ontology embeddings. Only need to run this once.
- evaluate_ner.py - script to evaluate the NER results.
- evaluate_relations.py	 - script to evaluate the relation extraction results.
- run_ner.py - script to run the NER pipeline.
- self_evaluate_re.py - script to run the relation extraction pipeline (run.py has this built in but if you want to run your own and separate, use this).




