import obonet
import logging

from aim2.utils.logging_cfg import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_ontology(obo_file):
    """
    Loads an ontology from an OBO file.
    """
    graph = None
    terms_dict = {}

    try:
        graph = obonet.read_obo(obo_file)

        # Extract the default namespace from the graph metadata
        default_namespace = graph.graph.get('default-namespace', 'unknown_namespace')

        for term_id, data in graph.nodes(data=True):
            # skips obselete terms or terms not meant for annotation
            if data.get('is_obsolete'):
                continue
            if 'gocheck_do_not_annotate' in data.get('subset', []):
                continue
            
            if 'name' in data:
                # check if the data has a namespace, if not, use the default namespace in "default-namespace:"
                if 'namespace' not in data:
                    data['namespace'] = default_namespace[0]
                terms_dict[term_id] = data
        logger.info(f"Successfully loaded ontology from {obo_file}")
    except FileNotFoundError:
        logger.warning(f"Ontology file {obo_file} not found.")
    except Exception as e:
        logger.warning(f"Error loading ontology from {obo_file}: {e}.")

    return terms_dict, graph
