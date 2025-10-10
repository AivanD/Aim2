from pydantic import ConfigDict, Field
from schemic import SchemicModel
from typing import List, Optional

# -------------------------
# --- ENTITY DEFINITIONS --
# examples are not seen by the model. It is used for documentation purposes.
# -------------------------
# TODO: for each entity class, check if built-in validation is necessary. (you wrote this in run.py as well)
# TODO: investigate why the output is now giving you fields that are supposed to be hidden.
class Natural_Product_Class(SchemicModel):
    Np_class: Optional[List[str]] = Field(default=None, description="Natural product class(es) for the compound.")
    Np_superclass: Optional[List[str]] = Field(default=None, description="Natural product superclass(es) for the compound.")

class ClassyFire(SchemicModel):
    Kingdom: Optional[str] = Field(default=None, description="Classyfire Kingdom for the compound.")
    Superclass: Optional[str] = Field(default=None, description="Classyfire Superclass for the compound.")
    Class: Optional[str] = Field(default=None, description="Classyfire Class for the compound.")
    Subclass: Optional[str] = Field(default=None, description="Classyfire Subclass for the compound.")

class Compound(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    CID: Optional[int] = Field(default=None, description="PubChem Compound ID (CID) for the compound.")
    name: str = Field(
        description="Compounds and/or metabolites found in plants, including specialized plant compounds, phytohormones, etc.",
        examples=["β-sitosterol", "abscisic acid", "gibberellin"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the compound is mentioned in the passage."
    )
    SMILES: Optional[str] = Field(default=None, description="SMILES string for the compound.")
    InChIKey: Optional[str] = Field(default=None, description="InChIKey for the compound.")
    Natural_product_class: Optional[Natural_Product_Class] = Field(default=None, description="Natural product class and superclass for the compound.")
    Classyfire: Optional[ClassyFire] = Field(default=None, description="Classyfire classification for the compound.")
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class Pathway(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolic pathways involving the transformation of metabolites.",
        examples=["glycolysis", "TCA cycle", "photosynthetic electron transport"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the pathway is mentioned in the passage."
    )
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class Genes(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant gene names.",
        examples=["MAP kinase 6", "phytochrome B"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the gene is mentioned in the passage."
    )
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class AnatomicalStructure(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Anatomical structures in plants.",
        examples=["plant embryo proper", 'lenticel', "root cortex"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the anatomical structure is mentioned in the passage."
    )
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class Species(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant species names.",
        examples=["Arabidopsis thaliana", "Oryza sativa", "Zea mays"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the species is mentioned in the passage."
    )
    taxonomy_id: Optional[int] = Field(default=None, description="NCBI Taxonomy ID for the species.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class ExperimentalCondition(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Experimental conditions in plant studies.",
        examples=['nickel exposure', 'oxygen sensitivity', 'leaf shattering']
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the experimental condition is mentioned in the passage."
    )
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

# class NP_Class(SchemicModel):
#     model_config = ConfigDict(extra="forbid")
#     name: str = Field(
#         description="Natural product classes in plants.",
#         examples=["steroidal glycoalkaloid", "glycoalkaloid", "flavonoids"]
#     )

class MolecularTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Molecular traits in plants.",
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the molecular trait is mentioned in the passage."
    )
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class PlantTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant traits in plants.",
        examples=["chromium sensitivity", "mimic response", "stem strength"]
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the plant trait is mentioned in the passage."
    )
    ontology_id: Optional[str] = Field(default=None, description="Ontology ID for the entity.")
    normalized_name: Optional[str] = Field(default=None, description="The canonical name from the ontology.")
    score: Optional[float] = Field(default=None, description="The similarity score from the normalization model.")
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

class HumanTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Human traits in plants.",
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the human trait is mentioned in the passage."
    )
    alt_names: Optional[List[str]] = Field(default=None, description="Alternative names or surface forms found in the text.")

# Create a main model to hold lists of all extracted entities
# TODO: consider using some NLP for some of the NER tasks. For example
# Genes and Species can be done using AIONER/GNORM2.
# Metabolites can be done using MetaboListem /TABoLiSTM. 
# if applying NLPs, remove some of the attibutes/classes in CustomExtractedEntities as well as in postprocessing/span_adder.py.
class CustomExtractedEntities(SchemicModel):
    """All entities extracted from the text."""
    model_config = ConfigDict(extra="forbid")
    compounds: List[Compound] = Field(default_factory=list, description="List of compound mentions.")
    pathways: List[Pathway] = Field(default_factory=list, description="List of pathway mentions.")
    genes: List[Genes] = Field(default_factory=list, description="List of gene mentions.")
    anatomical_structures: List[AnatomicalStructure] = Field(default_factory=list, description="List of anatomical structure mentions.")
    species: List[Species] = Field(default_factory=list, description="List of species mentions.")
    experimental_conditions: List[ExperimentalCondition] = Field(default_factory=list, description="List of experimental condition mentions.")
    # natural_product_classes: List[NP_Class] = Field(default_factory=list, description="List of natural product class mentions.")
    molecular_traits: List[MolecularTraits] = Field(default_factory=list, description="List of molecular trait mentions.")
    plant_traits: List[PlantTraits] = Field(default_factory=list, description="List of plant trait mentions.")
    human_traits: List[HumanTraits] = Field(default_factory=list, description="List of human trait mentions.")

# --- Simplified Models for LLM Extraction ---

class SimpleCompound(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Compounds and/or metabolites found in plants, including specialized plant compounds, phytohormones, etc.",
        examples=["β-sitosterol", "abscisic acid", "gibberellin"]
    )    

class SimplePathway(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolic pathways involving the transformation of metabolites.",
        examples=["glycolysis", "TCA cycle", "photosynthetic electron transport"]
    )

class SimpleGenes(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant gene names.",
        examples=["MAP kinase 6", "phytochrome B"]
    )

class SimpleAnatomicalStructure(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Anatomical structures in plants.",
        examples=["plant embryo proper", 'lenticel', "root cortex"]
    )

class SimpleSpecies(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant species names.",
        examples=["Arabidopsis thaliana", "Oryza sativa", "Zea mays"]
    )

class SimpleExperimentalCondition(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Experimental conditions in plant studies.",
        examples=['nickel exposure', 'oxygen sensitivity', 'leaf shattering']
    )

class SimpleMolecularTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Molecular traits in plants.",
    )

class SimplePlantTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant traits in plants.",
        examples=["chromium sensitivity", "mimic response", "stem strength"]
    )

class SimpleHumanTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Human traits in plants.",
    )

class SimpleExtractedEntities(SchemicModel):
    """A simplified version of entities for initial LLM extraction, containing only names."""
    model_config = ConfigDict(extra="forbid")
    compounds: List[SimpleCompound] = Field(default_factory=list, description="List of compound mentions.")
    pathways: List[SimplePathway] = Field(default_factory=list, description="List of pathway mentions.")
    genes: List[SimpleGenes] = Field(default_factory=list, description="List of gene mentions.")
    anatomical_structures: List[SimpleAnatomicalStructure] = Field(default_factory=list, description="List of anatomical structure mentions.")
    species: List[SimpleSpecies] = Field(default_factory=list, description="List of species mentions.")
    experimental_conditions: List[SimpleExperimentalCondition] = Field(default_factory=list, description="List of experimental condition mentions.")
    # natural_product_classes: List[NP_Class] = Field(default_factory=list, description="List of natural product class mentions.")
    molecular_traits: List[SimpleMolecularTraits] = Field(default_factory=list, description="List of molecular trait mentions.")
    plant_traits: List[SimplePlantTraits] = Field(default_factory=list, description="List of plant trait mentions.")
    human_traits: List[SimpleHumanTraits] = Field(default_factory=list, description="List of human trait mentions.")
