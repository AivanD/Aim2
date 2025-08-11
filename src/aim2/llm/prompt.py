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
        - Metabolites: Metabolites found in plants, including specialized plant compounds and phytohormones (e.g., salicylic acid, jasmonic acid, luteolin).
        - Pathways: Metabolic pathways involving the transformation of metabolites (e.g., glycolysis, TCA cycle, photosynthetic electron transport).
        - Genes: Plant gene names (e.g., MAP kinase 6, phytochrome B, FLC).
        - Anatomical Structures: Anatomical structures in plants (e.g., anther wall, root tip, plant cuticle, lenticel).
        - Species: Plant species names, including common and scientific names (e.g., Arabidopsis thaliana, Oryza sativa, Zea mays).
        - Experimental Conditions: Experimental conditions or treatments applied in plant studies (e.g., salt stress, drought, cold exposure, heavy metal exposure).
        - Molecular Traits: Molecular-level traits or measurements in plants (e.g., protein abundance, enzyme activity, gene expression level).
        - Plant Traits: Organism-level traits or qualities of a plant (e.g., plant height, leaf shattering, flowering time, drought resistance).
        - Human Traits: Traits of plants that are relevant to humans (e.g., nutritional quality, allergenicity, flavor, toxicity).

        Output format (JSON only):
        {
          "metabolites": [ { "name": "..." } ],
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
        - The examples below are for illustration only. Do not extract entities from the examples.
        - Each list may be empty if none are found.
                            
        Examples:
        Text: "This study does not involve the use of any metabolites."
        JSON: {"metabolites":[],"pathways":[],"genes":[],"anatomical_structures":[],"species":[],"experimental_conditions":[],"molecular_traits":[],"plant_traits":[],"human_traits":[]}

        Text: "In Arabidopsis thaliana, drought conditions led to increased protein abundance of the FLC gene in the root tip, affecting flowering time."
        JSON: {"metabolites":[],"pathways":[],"genes":[{"name":"FLC"}],"anatomical_structures":[{"name":"root tip"}],"species":[{"name":"Arabidopsis thaliana"}],"experimental_conditions":[{"name":"drought"}],"molecular_traits":[{"name":"protein abundance"}],"plant_traits":[{"name":"flowering time trait"}],"human_traits":[]}

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