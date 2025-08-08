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
        - Metabolites: metabolites (e.g., β-sitosterol, abscisic acid).
        - Pathways: metabolic pathways (e.g., glycolysis, TCA cycle).
        - Species: species names (including common names like "mouse" and scientific names like "Arabidopsis thaliana").

        Output format (JSON only):
        {
          "metabolites": [ { "name": "..." } ],
          "pathways":    [ { "name": "..." } ],
          "species":     [ { "name": "..." } ]
        }

        Rules:
        - Return ONLY a valid JSON object. Do not wrap it in quotes. No markdown or explanations.
        - Do not infer or guess entities; extract only verbatim mentions from the text. For example, if the text says "mouse", extract "mouse", not "Mus musculus".
        - Each list may be empty if none are found.
                            
        Examples:
        Text: "This study does not involve the use of any metabolites."
        JSON: {"metabolites":[],"pathways":[],"species":[]}

        Text: "The study on glycolysis in mice showed interesting results."
        JSON: {"metabolites":[],"pathways":[{"name":"glycolysis"}],"species":[{"name":"mice"}]}

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
