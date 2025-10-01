import os
import sys
import torch
import pickle
import logging
import html
from tqdm import tqdm
import re
import csv

from aim2.utils.config import PO_OBO, GO_OBO, TO_OBO, PECO_OBO, CHEMONT_OBO, PATHWAYS_PMN, DATA_DIR
from aim2.data.ontology import load_ontology
from aim2.llm.models import load_sapbert
from aim2.utils.logging_cfg import setup_logging

def create_plantcyc_embedding_cache(plantcyc_file, sapbert_model, output_path):
    """
    Loads PlantCyc pathways from the TXT file, creates embeddings, and saves them to a file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PlantCyc pathways: {plantcyc_file}")

    embedding_cache = {}
    terms_to_embed = []
    term_metadata = []

    try:
        with open(plantcyc_file, 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            reader = csv.reader(f, delimiter='\t')
            for row in tqdm(reader, desc="Collecting PlantCyc pathways"):
                if not row or len(row) < 2:
                    continue
                
                pathway_name = row[0]
                pathway_id = row[1]

                # Clean the name (remove HTML tags, etc.)
                cleaned_name = html.unescape(pathway_name).strip()
                cleaned_name = re.sub(r'<[^>]+>', '', cleaned_name) # Remove any HTML tags like <i>
                cleaned_name = re.sub(r'\s*\([^)]*\)$', '', cleaned_name).strip() # Remove trailing parentheses content

                if cleaned_name and pathway_id:
                    terms_to_embed.append(cleaned_name)
                    term_metadata.append({
                        'id': pathway_id,
                        'canonical_name': pathway_name, # Store original name
                        'namespace': 'plantcyc_pathway' # Assign a namespace
                    })
    except FileNotFoundError:
        logger.error(f"PlantCyc pathways file not found at {plantcyc_file}")
        return

    if not terms_to_embed:
        logger.warning(f"No terms found to embed in {plantcyc_file}")
        return

    logger.info(f"Generating embeddings for {len(terms_to_embed)} PlantCyc pathways...")
    embeddings = sapbert_model.encode(terms_to_embed, batch_size=128, show_progress_bar=True)

    embedding_cache['embeddings'] = torch.tensor(embeddings)
    embedding_cache['metadata'] = term_metadata

    with open(output_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    logger.info(f"PlantCyc pathway embedding cache saved to {output_path}")


def create_ontology_embedding_cache(ontology_file, sapbert_model, output_path):
    """
    Loads an ontology, creates embeddings for all terms and synonyms, and saves them to a file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing ontology: {ontology_file}")

    _, ontology_graph = load_ontology(ontology_file)
    if not ontology_graph:
        logger.error(f"Could not load graph from {ontology_file}")
        return
    
    embedding_cache = {}
    terms_to_embed = []
    term_metadata = []

    # Collect all terms and synonyms
    for term_id, data in tqdm(ontology_graph.nodes(data=True), desc="Collecting terms"):
        if 'name' in data:
            # Use a set to store unique names for this term
            names_for_term = {data['name'].strip()}

            # Process and clean synonym strings
            for syn_string in data.get('synonym', []):
                # The actual synonym is usually within the first pair of quotes
                if '"' in syn_string:
                    try:
                        syn_text = syn_string.split('"')[1]
                        # Unescape HTML entities (e.g., '&#243;' -> 'ó')
                        cleaned_name = html.unescape(syn_text).strip()
                        if cleaned_name:
                            cleaned_name = re.sub(r'\s*\([^)]*\)$', '', cleaned_name).strip()
                            names_for_term.add(cleaned_name)
                    except IndexError:
                        # Log if the synonym format is unexpected, but continue
                        logger.debug(f"Could not parse synonym string: {syn_string}")

            # Add the unique, cleaned names to the lists for embedding
            for name in names_for_term:
                if name:
                    terms_to_embed.append(name)
                    term_metadata.append({
                        'id': term_id, 
                        'canonical_name': data['name'],
                        'namespace': data.get('namespace') # Add namespace to metadata
                    })

    if not terms_to_embed:
        logger.warning(f"No terms found to embed in {ontology_file}")
        return

    # Generate embeddings in batches
    logger.info(f"Generating embeddings for {len(terms_to_embed)} terms including synonyms...")
    embeddings = sapbert_model.encode(terms_to_embed, batch_size=128, show_progress_bar=True)

    # Store embeddings with their metadata
    embedding_cache['embeddings'] = torch.tensor(embeddings)
    embedding_cache['metadata'] = term_metadata

    # Save the cache
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    logger.info(f"Embedding cache saved to {output_path}")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        sapbert = load_sapbert()
        logger.info("SapBERT model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SapBERT model: {e}")
        sys.exit(1)

    # Define ontologies and their output cache files
    ontologies_to_process = {
        "po": (PO_OBO, DATA_DIR / "po_embeddings.pkl"),
        "go": (GO_OBO, DATA_DIR / "go_embeddings.pkl"),
        "to": (TO_OBO, DATA_DIR / "to_embeddings.pkl"),
        "peco": (PECO_OBO, DATA_DIR / "peco_embeddings.pkl"),
        "chemont": (CHEMONT_OBO, DATA_DIR / "chemont_embeddings.pkl"),
    }

    for name, (obo_file, cache_file) in ontologies_to_process.items():
        if not os.path.exists(cache_file):
            create_ontology_embedding_cache(obo_file, sapbert, cache_file)
        else:
            logger.info(f"Cache file for '{name}' already exists at {cache_file}. Skipping.")

    # Process the PlantCyc pathways file
    plantcyc_cache_file = DATA_DIR / "plantcyc_pathways_embeddings.pkl"
    if not os.path.exists(plantcyc_cache_file):
        create_plantcyc_embedding_cache(PATHWAYS_PMN, sapbert, plantcyc_cache_file)
    else:
        logger.info(f"Cache file for 'plantcyc_pathways' already exists at {plantcyc_cache_file}. Skipping.")

if __name__ == "__main__":
    main()
    