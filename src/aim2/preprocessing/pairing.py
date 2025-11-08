import logging
import math, re
from typing import Dict, List, Tuple, Iterable, Any

Unit = Tuple[str, int]  # (text, abs_start_offset)

logger = logging.getLogger(__name__)

def find_entity_pairs(final_entities: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[Dict, Dict, str]]:
    """
    Identifies pairs of (compound, other_entity) for relation extraction.
    The target compounds are those that do not have an 'ontology_id'.
    """
    pairs = []
    
    # 1. Identify target compounds (molecular compounds, not classes)
    # target_compounds = [
    #     entity for entity in final_entities.get("compounds", [])
    #     if not entity.get("ontology_id") and entity.get("CID")
    # ]

    # 1. Identify target compounds including all (molecular compounds, classes, unknowns)
    target_compounds = final_entities.get("compounds", [])

    if not target_compounds:
        logger.info("No target molecular compounds found for relation extraction.")
        return []

    # 2. Pair them with all other entities
    for compound in target_compounds:
        for category, entities in final_entities.items():
            if category == "compounds":
                continue  # skip pairing with other compounds for now
                # for other_compound in entities:
                #     if compound is other_compound:
                #         continue  # skip self-pair
                #     pairs.append((compound, other_compound, "compound"))
            for entity in entities:
                pairs.append((compound, entity, category))
    
    logger.info(f"Generated {len(pairs)} entity pairs for relation extraction.")
    return pairs

def _count_mentions(spans: List[Tuple[int,int]], start: int, end: int) -> int:
    c = 0
    for s, e in spans:
        if not (e <= start or s >= end):  # any overlap counts
            c += 1
    return c

def _mention_positions(spans: List[Tuple[int,int]], start: int, end: int) -> List[int]:
    # absolute centers for proximity
    return [ (s+e)//2 for (s,e) in spans if not (e <= start or s >= end) ]

# simple sentence splitter used only inside a paragraph unit
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')

def _split_sentences(unit_text: str, unit_start: int) -> List[Tuple[int,int]]:
    """Return sentence spans as (abs_start, abs_end) within the unit."""
    spans, i = [], 0
    parts = _SENT_SPLIT.split(unit_text) or [unit_text]
    for k, chunk in enumerate(parts):
        s = unit_start + i
        e = s + len(chunk)
        spans.append((s, e))
        # +1 is a best-effort for the removed split whitespace (fine for ranking)
        i += len(chunk) + (1 if k < len(parts)-1 else 0)
    return spans

def _sent_index_of_pos(pos: int, sent_spans: List[Tuple[int,int]]) -> int:
    for idx, (s, e) in enumerate(sent_spans):
        if s <= pos < e:
            return idx
    return max(0, len(sent_spans)-1)

def _proximity_bonus(posA: List[int], posB: List[int], sent_spans: List[Tuple[int,int]]) -> float:
    """Sum over A of 1/(1 + min sentence-distance to any B)."""
    if not posA or not posB:
        return 0.0
    bonus = 0.0
    for pa in posA:
        sa = _sent_index_of_pos(pa, sent_spans)
        d = min(abs(sa - _sent_index_of_pos(pb, sent_spans)) for pb in posB)
        bonus += 1.0 / (1.0 + d)  # same sentence = +1.0; one apart = +0.5; etc.
    return bonus

def _idf_over_units(units: Iterable[Unit], spans: List[Tuple[int,int]]) -> float:
    units = list(units)
    N = len(units)
    df = 0
    for text, u_start in units:
        u_end = u_start + len(text)
        if _count_mentions(spans, u_start, u_end) > 0:
            df += 1
    # smoothed IDF at the chosen granularity
    return math.log(1.0 + (N / (df if df > 0 else 1.0)))

# ---------- main API ----------

def rank_passages_for_pair_enhanced(
    entity_A: Dict,                      # subject: compound
    entity_B: Dict,                      # object: pathway/species/…
    units_w_offsets: List[Unit],         # paragraphs OR sentences
    granularity: str = "paragraph",      # "paragraph" | "sentence"
    top_k: int = 10,
    same_sentence_boost: float = 1.2
) -> List[Tuple[str, float, Dict]]:
    """
    Pair-aware ranking usable for paragraphs or sentences:
      - harmonic co-count (F1-like) with log damping
      - intra-unit proximity (sentence distance; for sentences this collapses to same-sentence check)
      - IDF per entity across units
    Returns: [(text, score, diagnostics), ...]
    """
    spans_A = entity_A.get("spans", [])
    spans_B = entity_B.get("spans", [])

    # Inverse Document Frequency at the current granularity
    # (IDF- down-weights mentions that are common across many units (paragraphs/sentences), so they don’t dominate the ranking just because they appear everywhere.) 
    idfA = _idf_over_units(units_w_offsets, spans_A)
    idfB = _idf_over_units(units_w_offsets, spans_B)

    ranked = []
    for text, u_start in units_w_offsets:
        u_end = u_start + len(text)
        nA = _count_mentions(spans_A, u_start, u_end)
        nB = _count_mentions(spans_B, u_start, u_end)
        if nA == 0 or nB == 0:
            continue

        # harmonic co-count with log damping
        harm = (2.0 * nA * nB) / (nA + nB + 1e-6)
        co   = math.log(1.0 + harm)

        # sentence spans inside the unit (for sentences: just one)
        sent_spans = [(u_start, u_end)] if granularity == "sentence" else _split_sentences(text, u_start)

        posA = _mention_positions(spans_A, u_start, u_end)
        posB = _mention_positions(spans_B, u_start, u_end)
        prox = _proximity_bonus(posA, posB, sent_spans)

        # same-sentence boost if any overlap in sentence indices
        same = False
        if posA and posB:
            a_idx = {_sent_index_of_pos(p, sent_spans) for p in posA}
            b_idx = {_sent_index_of_pos(p, sent_spans) for p in posB}
            same = len(a_idx & b_idx) > 0
        boost = same_sentence_boost if same else 1.0

        score = co * (1.0 + prox) * idfA * idfB * boost

        ranked.append((
            text,
            score,
            {"nA": nA, "nB": nB, "co": co, "prox": prox,
             "idfA": idfA, "idfB": idfB, "same_sentence": same,
             "unit_start": u_start, "unit_end": u_end}
        ))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ---------- helper: pick the best sentences only from chosen paragraphs ----------

def select_best_sentences_from_paragraphs(
    entity_A: Dict,
    entity_B: Dict,
    top_paragraphs: List[Tuple[str, float, Dict]],  # output of rank_passages_for_pair_enhanced on paragraphs
    sentences_w_offsets: List[Unit],
    per_paragraph: int = 2
) -> List[Tuple[str, float, Dict]]:
    """
    For each *selected paragraph*, pick the top 1–N sentences (by the same enhanced score),
    but only among sentences that fall inside that paragraph's span.
    """
    spans_A = entity_A.get("spans", [])
    spans_B = entity_B.get("spans", [])

    # Precompute sentence-level IDF once (over all sentences)
    idfA_sent = _idf_over_units(sentences_w_offsets, spans_A)
    idfB_sent = _idf_over_units(sentences_w_offsets, spans_B)

    selected_sentences: List[Tuple[str, float, Dict]] = []
    unique_sentences = set() # To avoid adding the same sentence from different paragraphs

    for par_text, _, diag in top_paragraphs:
        par_start = diag.get("unit_start")
        par_end = diag.get("unit_end")

        if par_start is None or par_end is None:
            continue # Should not happen with the change above

        par_candidates = []
        for s_text, s_start in sentences_w_offsets:
            s_end = s_start + len(s_text)
            # Use precise offset check instead of string containment
            if s_start >= par_start and s_end <= par_end:
                # Check if we've already processed this sentence
                if (s_text, s_start) in unique_sentences:
                    continue

                nA = _count_mentions(spans_A, s_start, s_end)
                nB = _count_mentions(spans_B, s_start, s_end)
                if nA == 0 or nB == 0:
                    continue
                # sentence-level score (no proximity across sentences; prox collapses to same-sentence case)
                harm = (2.0 * nA * nB) / (nA + nB + 1e-6)
                co   = math.log(1.0 + harm)
                # same-sentence case => prox=1 per A mention; keep it simple: prox=1 if both present
                prox = 1.0
                # Apply same-sentence boost, as both entities are in this sentence
                score = co * (1.0 + prox) * idfA_sent * idfB_sent * 1.2
                par_candidates.append((s_text, s_start, score, {"nA": nA, "nB": nB, "co": co, "prox": prox,
                                                                "idfA": idfA_sent, "idfB": idfB_sent, 
                                                                "unit_start": s_start, "unit_end": s_end}))
                unique_sentences.add((s_text, s_start))

        par_candidates.sort(key=lambda x: x[2], reverse=True)       # sort by score (x=2)
        selected_sentences.extend(par_candidates[:per_paragraph])

    # Final sort of all selected sentences to get the absolute best ones
    selected_sentences.sort(key=lambda x: x[2], reverse=True)       # sort by score (x=2)
    return selected_sentences
