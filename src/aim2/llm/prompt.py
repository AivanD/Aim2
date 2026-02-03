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

def make_prompt_body_only(article_text: str) -> str:
    """
    Creates the body of a prompt for the LLM to extract entities from a given text.
    Returns:
        str: The complete prompt body string.
    """
    prompt = f"{article_text}\n"
    return prompt

def make_prompt(article_text: str) -> str:
    """
    Creates a prompt for the LLM to extract entities from a given text.
    Returns:
        str: The complete prompt string.
    """
    prompt = _static_header()
    prompt += make_prompt_body_only(article_text)
    return prompt

RELATION_GUIDELINES = {
    "compounds": {
        "associated_with": "A general, non-causal link between compounds is reported.",
        "present_in": "The compound is detected or found in the presence of the other compound.",
        "correlates_with": "Levels of the two compounds co-vary (direction not specified).",
        "positively_regulated_by": "A compound’s level increases in response to the presence/perturbation of another compound.",
        "negatively_regulated_by": "A compound’s level decreases in response to the presence/perturbation of another compound.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "pathways": {
        "made_via": "The compound is synthesized, created or produced via this pathway.",
        "degraded_via": "The compound is broken down or consumed through this pathway.",
        "involved_in": "The compound is a substrate/intermediate/product within this pathway.",
        "associated_with": "A non-specific link between the compound and the pathway.",
        "correlates_with": "The compound level correlates with pathway activity/abundance (direction not specified).",
        "positively_correlates_with": "Higher compound levels tend to co-occur with increased pathway activity/abundance.",
        "negatively_correlates_with": "Higher compound levels tend to co-occur with decreased pathway activity/abundance.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "genes": {
        "made_by": "The gene/protein (e.g., as an enzyme) directly produces the compound.",
        "degraded_by": "The gene/protein directly degrades, converts, or consumes the compound.",
        "correlates_with": "Compound levels correlate with gene expression/activity (direction not specified).",
        "positively_correlates_with": "Higher gene expression/activity tends to co-occur with higher compound levels.",
        "negatively_correlates_with": "Higher gene expression/activity tends to co-occur with lower compound levels.",
        "positively_regulated_by": "Genetic manipulations (e.g., OE/activation) cause clear increases in compound levels.",
        "negatively_regulated_by": "Genetic manipulations (e.g., KO/repression) cause clear decreases in compound levels.",
        "secreted_by": "Gene/protein mediates secretion/export of the compound.",
        "transported_by": "Gene/protein transports the compound across membranes/compartments.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "anatomical_structures": {
        "made_in": "The compound is synthesized in this anatomical structure.",
        "accumulates_in": "The compound builds up in this anatomical structure.",
        "present_in": "The compound is detected in this anatomical structure.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "species": {
        "made_in": "The compound is synthesized by this species.",
        "accumulates_in": "The compound builds up or increases in this species.",
        "present_in": "The compound is given to or detected in this species.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "experimental_conditions": {
        "made_under": " The compound is synthesized/produced under the named experimental condition.",
        "accumulates_under": "The compound accumulates (increases) under the experimental condition.",
        "present_under": " The compound is detected under the named experimental condition.",
        "associated_with": "A general link between the compound and the experimental condition is reported.",
        "involved_in": "The compound’s role/participation is reported in the context of this experimental condition.",
        "positively_regulated_by": "Compound level increases under this experimental condition.",
        "negatively_regulated_by": "Compound level decreases under this experimental condition.",
        "correlates_with": "Compound level correlates with intensity/duration of the experimental condition (direction not specified).",
        "positively_correlates_with": "Higher compound levels tend to co-occur with stronger/more prolonged experimental condition.",
        "negatively_correlates_with": "Higher compound levels tend to co-occur with weaker/shorter experimental condition.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "molecular_traits": {
        "affects": "The compound alters the trait, activity or function.",
        "modulates": "The compound regulates the trait, activity or function.",
        "involved_in": "The compound participates in processes underlying the trait, activity or function.",
        "associated_with": "A general, non-causal link with the trait, activity or function is reported.",
        "correlates_with": "Compound levels correlate with the trait, activity or function (direction not specified).",
        "positively_correlates_with": "Higher compound levels tend to co-occur with higher trait, activity or function values.",
        "negatively_correlates_with": "Higher compound levels tend to co-occur with lower trait, activity or function values.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    },

    "plant_traits": {
        "affects": "The compound alters the plant trait.",
        "modulates": "The compound regulates the plant trait.",
        "involved_in": "The compound participates in processes underlying the plant trait.",
        "associated_with": "A general, non-causal link with the plant trait is reported.",
        "correlates_with": "Compound levels correlate with the plant trait (direction not specified).",
        "positively_correlates_with": "Higher compound levels tend to co-occur with higher plant trait values.",
        "negatively_correlates_with": "Higher compound levels tend to co-occur with lower plant trait values.",
        "No_Relationship": "No direct relationship is stated or strongly implied."
    }
}

def _static_header_re() -> str:
    """
    Creates a prompt for the LLM to extract a relationship between two entities.
    """

    prompt = dedent(f"""
        You are an expert annotator analyzing scientific text. Decide the relationship between the "Subject" and the "Object" based ONLY on the provided Text Context.

        Instructions:
        1. Carefully read the "Text Context" below.
        2. Describe the relationship using a concise phrase.
        3. If no relationship is explicitly stated or strongly implied in the text, you must choose "No_Relationship" for the predicate.
        4. Provide an exact in-text justification for your chosen relationship. Otherwise, write "No justification found".
        5. Output ONLY a valid JSON object.
                    
        Output Format (JSON only):
        {{"predicate": "...", "justification": "..."}}
    """)
    return prompt

def make_re_prompt_body_only(compound: Dict[str, Any], other_entity: Dict[str, Any], category: str, context_passages: List[str]) -> str:
    """
    Creates the body of a prompt for the LLM to extract a relationship between two entities.
    Returns:
        str: The complete prompt body string.
    """

    compound_name = compound.get('name')
    other_entity_name = other_entity.get('name')
    
    # Format subject line with alternative names if available
    subject_line = f'- Subject (Compound): "{compound_name}"'
    compound_alt_names = compound.get('alt_names')
    if compound_alt_names:
        alt_names_str = ', '.join([f'"{name}"' for name in compound_alt_names])
        subject_line += f' (also known as: {alt_names_str})'

    # Format object line with alternative names if available
    object_line = f'- Object ({category}): "{other_entity_name}"'
    other_entity_alt_names = other_entity.get('alt_names')
    if other_entity_alt_names:
        alt_names_str = ', '.join([f'"{name}"' for name in other_entity_alt_names])
        object_line += f' (also known as: {alt_names_str})'
    
    # # Get the specific guidelines for the given category
    # if category == "compound":
    #     guidelines = RELATION_GUIDELINES.get("compounds", {})
    # else:
    #     guidelines = RELATION_GUIDELINES.get(category, {})
    
    # # Format the guidelines directly as a string with consistent indentation
    # guideline_lines = [f"        - \"{rel}\": {desc}" for rel, desc in guidelines.items()]
    # guideline_block = f"Allowed Relationships:\n" + "\n".join(guideline_lines)

    # prompt with predicate guidelines removed
    prompt = dedent(f"""           
        Entities:
        {subject_line}
        {object_line}

        Text Context:
    """)

    # # prompt with predicate guidelines included
    # prompt = dedent(f"""           
    #     Entities:
    #     {subject_line}
    #     {object_line}

    #     {guideline_block}

    #     Text Context:
    # """)

    for context_passage in context_passages:
        prompt += context_passage + "\n"

    return prompt

def make_re_prompt(compound: Dict[str, Any], other_entity: Dict[str, Any], category: str, context_passages: List[str]) -> str:
    """
    Creates a prompt for the LLM to extract a relationship between two entities.
    Returns:
        str: The complete prompt string.
    """

    prompt = _static_header_re()
    prompt += make_re_prompt_body_only(compound, other_entity, category, context_passages)
    return prompt


def _static_header_re_evaluation() -> str:
    """
    Creates a static header part of the prompt for relationship validation.
    Returns:
        str: The static header string.
    """
    prompt = dedent("""
        You are an expert annotator verifying a relationship extracted from a scientific text.
        Based ONLY on the provided "Text Context", determine if the following statement is true.

        Is the statement supported by the text? Answer "yes" or "no".

        Output Format (JSON only):
        {"decision": "..."}
    """)
    return prompt

def make_re_evaluation_prompt_body_only(relation: Dict[str, Any]) -> str:
    """
    Creates the body of a prompt for the LLM to validate a relationship.
    """
    subject_entity = relation['subject_entity']
    object_entity = relation['object_entity']
    predicate = relation['predicate']
    category = relation.get('category', 'unknown') # Assumes category is added to the relation object

    # Get the definition for the specific predicate
    predicate_definition = RELATION_GUIDELINES.get(category, {}).get(predicate, "No definition available.")

    # Format subject line with alternative names
    subject_line = f'- Subject (Compound): "{subject_entity["name"]}"'
    if subject_entity.get('alt_names'):
        alt_names_str = ', '.join([f'"{name}"' for name in subject_entity['alt_names']])
        subject_line += f' (also known as: {alt_names_str})'

    # Format object line with alternative names
    object_line = f'- Object ({category}): "{object_entity["name"]}"'
    if object_entity.get('alt_names'):
        alt_names_str = ', '.join([f'"{name}"' for name in object_entity['alt_names']])
        object_line += f' (also known as: {alt_names_str})'

    body = dedent(f"""
        Statement: 
        {subject_line}
        {object_line}
        - Relationship: "{predicate}" (Definition: {predicate_definition})

        Is the statement above supported by the following text?

        Text Context:
        {relation['context']}
    """)
    return body

def make_re_evaluation_prompt(relation: Dict[str, Any]) -> str:
    """
    Creates a prompt for the LLM to validate an extracted relationship.
    """
    prompt = _static_header_re_evaluation()
    prompt += make_re_evaluation_prompt_body_only(relation)
    return prompt