def embed_ontology_terms(terms_dict, model):
    """
    Given a dictionary of ontology terms (ID -> term data) and an embedding model,
    Parameters:
        - terms_dict: A dictionary where keys are term IDs and values are dictionaries containing
            'name', 'namespace', and optionally 'relationship'.
        - model: The embedding model to use for encoding the term names.
    Returns:
        - a list of term IDs,
        - a list of term names (in the same order),
        - a tensor of embeddings for the term names,
        - a list of term namespaces,
        - a list of term relationships (if available, otherwise default to 'unknown')
    """
    term_ids = []
    term_names = []
    term_namespaces = []
    term_relationships = []

    for term_id, data in terms_dict.items():
        if 'name' in data:
            term_ids.append(term_id)
            term_names.append(data['name'])
            term_namespaces.append(data.get('namespace', 'unknown'))
            # Some term entries might not have a "relationship" field; default to 'unknown'
            term_relationships.append(data.get('relationship', 'unknown'))

    term_embeddings = model.encode(term_names, convert_to_tensor=True)
    return term_ids, term_names, term_embeddings, term_namespaces, term_relationships