from textwrap import dedent

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
        - Species: Plant species names, including common and scientific names (e.g., Arabidopsis thaliana, Oryza sativa, Zea mays).
        - Experimental Conditions: Treatments, growing conditions, and/or study types used in plant biology experiments. (e.g., salt exposure, drought environment exposure, cold temperature exposure, IR light exposure).
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