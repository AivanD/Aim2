import obonet

def load_plant_ontology(obo_file):
    """
    Loads the plant ontology from an OBO file.
    """
    po_graph = None
    terms_dict = {}
    
    try:
        po_graph = obonet.read_obo(obo_file)
        for go_id, data in po_graph.nodes(data=True):
            # Only include nodes that have a 'name' field
            if 'name' in data:
                terms_dict[go_id] = data
        print(f"Successfully loaded GO ontology from {obo_file}")
    except FileNotFoundError:
        print(f"Warning: GO ontology file {obo_file} not found. GO_Name will be NULL.")
    except Exception as e:
        print(f"Warning: Error loading GO ontology from {obo_file}: {e}. GO_Name will be NULL.")

    return terms_dict, po_graph