def _static_header():
    """
    Creates the static header part of the prompt.
    Returns:
        str: The static header string.
    """
    prompt = f"""
        Extract all entities from the following text based on the provided JSON schema. The entity types are: Compound, Stressor, Process, Trait, Stage, Structure, and Organoleptic.
        If no entities of a type are found, return an empty list for that type.

        Text:
    """
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
