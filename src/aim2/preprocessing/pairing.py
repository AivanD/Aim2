import logging
import math
from typing import Dict, List, Any, Tuple

from aim2.entities_types.entities import Compound

logger = logging.getLogger(__name__)

def find_entity_pairs(final_entities: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[Dict, Dict, str]]:
    """
    Identifies pairs of (compound, other_entity) for relation extraction.
    The target compounds are those that do not have an 'ontology_id'.
    """
    pairs = []
    
    # 1. Identify target compounds (molecular compounds, not classes)
    target_compounds = [
        entity for entity in final_entities.get("compounds", [])
        if not entity.get("ontology_id") and entity.get("CID")
    ]
    if not target_compounds:
        logger.info("No target molecular compounds found for relation extraction.")
        return []

    # 2. Pair them with all other entities
    for compound in target_compounds:
        for category, entities in final_entities.items():
            if category == "compounds":
                continue  # Don't pair compounds with themselves
            for entity in entities:
                pairs.append((compound, entity, category))
    
    logger.info(f"Generated {len(pairs)} entity pairs for relation extraction.")
    return pairs

def rank_passages_for_pair(
    entity_A: Dict, 
    entity_B: Dict, 
    passages_w_offsets: List[Tuple[str, int]]
) -> List[Tuple[str, float]]:
    """
    Finds and ranks passages containing both entities based on the balanced, damped frequency score.
    """
    candidate_passages = []
    spans_A = entity_A.get("spans", [])
    spans_B = entity_B.get("spans", [])

    for passage_text, passage_offset in passages_w_offsets:
        passage_end = passage_offset + len(passage_text)
        
        # Count mentions of each entity in the current passage
        nA = sum(1 for start, end in spans_A if passage_offset <= start and end <= passage_end)
        nB = sum(1 for start, end in spans_B if passage_offset <= start and end <= passage_end)

        if nA > 0 and nB > 0:
            # Calculate the balanced, damped frequency score
            freq_balanced = math.log(1 + nA) * math.log(1 + nB)
            balance = min(nA, nB) / max(nA, nB)
            score = freq_balanced * balance
            candidate_passages.append((passage_text, score))

    # Sort passages by score in descending order
    candidate_passages.sort(key=lambda x: x[1], reverse=True)
    return candidate_passages