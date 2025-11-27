import logging
import pickle
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Dict, Any
import time

from aim2.utils.config import DATA_DIR

logger = logging.getLogger(__name__)

class SapbertNormalizer:
    """
    Normalizes entities against cached ontology embeddings using SapBERT.
    """
    def __init__(self, sapbert_model: SentenceTransformer, thresholds: Dict[str, float] = None):
        self.model = sapbert_model
        self.device = self.model.device  # Get the device from the model
        self.default_threshold = 0.9
        self.thresholds = thresholds if thresholds is not None else {}
        self.ontology_caches = self._load_caches()
        self.entity_to_ontology_map = {
            "compounds": "chemont",
            "pathways": "plantcyc_pathways",
            "anatomical_structures": "po",
            "molecular_traits": "go",
            "plant_traits": "to",
            "experimental_conditions": "peco",
        }
        # Map entity types to a specific namespace within an ontology
        self.entity_to_namespace_map = {
            "molecular_traits": "molecular_function",
            # "plant_traits": "plant_anatomy"   # TODO: find out whether we want to restrict it to just plant_anatomy or plant_anatomy and plant_structure_development_stage (default)
        }

    def _load_caches(self) -> Dict[str, Dict]:
        """Loads all pre-computed ontology embedding caches."""
        caches = {}
        ontology_files = {
            "po":       DATA_DIR / "po_embeddings.pkl",
            "go":       DATA_DIR / "go_embeddings.pkl",
            "to":       DATA_DIR / "to_embeddings.pkl",
            "peco":     DATA_DIR / "peco_embeddings.pkl",
            "chemont":  DATA_DIR / "chemont_embeddings.pkl",
            "plantcyc_pathways": DATA_DIR / "plantcyc_pathways_embeddings.pkl",

        }
        for name, path in ontology_files.items():
            try:
                with open(path, 'rb') as f:
                    cache_data = pickle.load(f)
                    # Move the cached embeddings tensor to the model's device (GPU)
                    cache_data['embeddings'] = cache_data['embeddings'].to(self.device)
                    caches[name] = cache_data
                logger.info(f"Successfully loaded '{name}' ontology embedding cache from {path}.")
            except FileNotFoundError:
                logger.warning(f"Cache file for '{name}' not found at {path}. Run cache_ontology_embeddings.py.")
                caches[name] = None
        return caches

    def _find_best_match(self, entity_name: str, ontology_key: str, threshold: float, namespace: str = None) -> Dict[str, Any]:
        """Finds the best match for a single entity name in the specified ontology cache."""
        cache = self.ontology_caches.get(ontology_key)
        if not cache or not entity_name:
            return None

        entity_embedding = self.model.encode(entity_name, convert_to_tensor=True, show_progress_bar=False)
        
        # namespace filtering (just in case you embed more than one namespace in the cache)
        candidate_indices = list(range(len(cache['metadata'])))
        if namespace:
            candidate_indices = [
                i for i, meta in enumerate(cache['metadata'])
                if meta.get('namespace') == namespace
            ]

        if not candidate_indices:
            return None # No candidates match the required namespace

        candidate_embeddings = cache['embeddings'][candidate_indices]

        similarities = cos_sim(entity_embedding, candidate_embeddings)[0]

        best_match_idx = torch.argmax(similarities).item()
        best_score = similarities[best_match_idx].item()

        if best_score > threshold:
            metadata = cache['metadata'][best_match_idx]
            return {
                "ontology_id": metadata['id'],
                "normalized_name": metadata['canonical_name'],
                "score": best_score,
            }
        return None

    def normalize_entities(self, processed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Iterates through results and normalizes relevant entity types using specific thresholds."""
        ontology_norm_start_time = time.time()
        logger.info("Starting ontology normalization for entities...")
        for passage_entities in processed_results:
            for entity_type, ontology_key in self.entity_to_ontology_map.items():
                if entity_type in passage_entities:
                    # Get the specific threshold for this entity type, or use the default
                    threshold = self.thresholds.get(entity_type, self.default_threshold)
                    # Check if a specific namespace is required for this entity type
                    required_namespace = self.entity_to_namespace_map.get(entity_type)  
                    for entity in passage_entities[entity_type]:
                        match = self._find_best_match(entity['name'], ontology_key, threshold, namespace=required_namespace)
                        if match:
                            entity.update(match)
        logger.info(f"Ontology normalization complete in {time.time() - ontology_norm_start_time:.2f} seconds.")
        return processed_results