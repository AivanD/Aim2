from textwrap import dedent
from typing import List, Dict, Any

from aim2.entities_types.entities import Compound

def _static_header():
    """
    Creates the static header part of the prompt.
    Returns:
        str: The static header string.
    """
    prompt = dedent("""
        You are an expert in plant biology and scientific literature analysis. Extract entities from the provided scientific text.
        Adhere strictly to the JSON schema enforced by the tool.

        Entity types to extract:
        - Compounds: Compounds and/or metabolites found in plants, including specialized plant compounds and phytohormones (e.g., salicylic acid, jasmonic acid).
        - Pathways: Metabolic pathways involving the transformation of metabolites (e.g., thioredoxin pathway, TCA cycle, methiin metabolism).
        - Genes: Plant gene names (e.g., HISN6A, AT2G46505, AT3G19450).
        - Anatomical Structures: Physical anatomical structures in plants, including organs, tissues, cells, cell parts and anatomical spaces (e.g., anther wall, root tip, plant cuticle, lenticel).
        - Species: Plant species names only in binomial nomenclature format (genus + specific epithet), either in full form (e.g., Arabidopsis thaliana, Beta vulgaris, Nicotiana benthamiana) or abbreviated form (e.g., A. thaliana, B. vulgaris, N. benthamiana). Do NOT extract higher taxonomic categories such as families (ending in -aceae), orders (ending in -ales), classes, phyla, or other taxonomic ranks above species level.
        - Experimental Conditions: Treatments, growing conditions, and/or study types used in plant biology experiments (e.g., salt exposure, drought environment exposure, cold temperature exposure, IR light exposure).
        - Molecular Traits: Molecular-level traits or molecular functions in plants (e.g., oxidoreductase activity, glycosyltransferase activity, GTP binding).
        - Plant Traits: Observable phenotypic traits of a plant, distinguishable features, characteristics or qualities of a developing or maturing plant (e.g., plant height, leaf shattering, flowering time trait, drought tolerance).
        - Human Traits: Traits of plants that are relevant to humans (e.g., nutritional quality, allergenicity, flavor, toxicity).

        Output format (JSON only):
        {
            "compounds": [ { "name": "..." } ],
            "pathways": [ { "name": "..." } ],
            "genes": [ { "name": "..." } ],
            "anatomical_structures": [ { "name": "..." } ],
            "species": [ { "name": "..." } ],
            "experimental_conditions": [ { "name": "..." } ],
            "molecular_traits": [ { "name": "..." } ],
            "plant_traits": [ { "name": "..." } ],
            "human_traits": [ { "name": "..." } ]
        }

        Rules:
        - Return ONLY a valid JSON object. Do not wrap it in quotes. No markdown or explanations.
        - Do not infer or guess entities; extract only verbatim mentions from the text. For example, if the text says "A. thaliana", extract "A. thaliana", not "Arabidopsis thaliana".
        - If an entity name is followed by an abbreviation or alternative name in parentheses, extract both separately.
        - Cross-layer overlaps: If a span embeds a valid mention of another entity type, annotate both—the full span for its type and the minimal nested substring for the other type.
        - Hyphenated/derived modifiers: extract only the underlying compound names, not the full hyphenated phrase.
        - Compounds should be extracted as atomic units only. If multiple compounds appear in one phrase, extract them separately.
        - Each list may be empty if none are found.
                    
        Now, analyze the following text:
    """)
    return prompt

def make_prompt(article_text: str) -> str:
    """
    Creates a prompt for the LLM to extract entities from a given text.
    Returns:
        str: The complete prompt string.
    """
    prompt = _static_header()
    prompt += f"{article_text}\n"
    return prompt

RELATION_MAP = {
    "pathways": "made_via, biosynthesized_via, degraded_via, No_Relationship",
    "genes": "made_by, biosynthesized_by, degraded_by, associated_with, No_Relationship",
    "anatomical_structures": "made_in, accumulates_in, found_in, present_in, No_Relationship",
    "species": "made_in, accumulates_in, found_in, present_in, No_Relationship",
    "experimental_conditions": "made_in, accumulates_in, found_in, present_in, No_Relationship",
    "molecular_traits": "affects, modulates, influences, involved_in, associated_with, No_Relationship",
    "plant_traits": "affects, modulates, influences, involved_in, associated_with, No_Relationship",
    "human_traits": "affects, modulates, influences, No_Relationship",
}

def _static_header_re(compound: Dict[str, Any], other_entity: Dict[str, Any], category: str) -> str:
    """
    Creates a prompt for the LLM to extract a relationship between two entities.
    """
    compound_name = compound.get('name')
    other_entity_name = other_entity.get('name')
    allowed_relations = RELATION_MAP.get(category, "No_Relationship")

    prompt = dedent(f"""
        You are an expert biologist analyzing scientific text. Your task is to identify the relationship between a specific compound and another entity based *only* on the provided text context.
        
        **Entities:**
        - Subject (Compound): "{compound_name}"
        - Object ({category}): "{other_entity_name}"
        
        **Allowed Relationships for this pair:**
        [{allowed_relations}]
        
        **Instructions:**
        1. Carefully read the "Text Context" below. 
        2. Determine the most accurate relationship from the "Allowed Relationships list that describes the connection between the Subject and the Object. 
        3. If no relationship is explicitly stated or strongly implied in the text, you must choose "No_Relationship".
        4. Provide a brief, direct quote from the text that serves as the justification for your chosen relationship. If no direct quote is possible, write "No justification found".
        5. Output ONLY a valid JSON object with the keys "predicate" and "justification". 

        Output Format (JSON only):
        {{
            "predicate": "...", 
            "justification": "..."
        }}

        **Text Context:**
    """)
    return prompt

def make_re_prompt(compound: Dict[str, Any], other_entity: Dict[str, Any], category: str, context_passages: List[str]) -> str:
    """
    Creates a prompt for the LLM to extract a relationship between two entities.
    Returns:
        str: The complete prompt string.
    """
    prompt = _static_header_re(compound, other_entity, category)
    for context_passage in context_passages:
        prompt += context_passage + "\n"

    return prompt