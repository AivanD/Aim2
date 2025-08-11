from pydantic import ConfigDict, Field
from schemic import SchemicModel
from typing import List, Optional

# -------------------------
# --- ENTITY DEFINITIONS --
# examples are not seen by the model. It is used for documentation purposes.
# -------------------------
# TODO: add the rest of the entities. 
# TODO: for each entity class, check if built-in validation is necessary. (you wrote this in run.py as well)
class Metabolite(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolites found in plants, including specialized plant compounds, phytohormones, etc.",
        examples=["β-sitosterol", "abscisic acid", "gibberellin"]
    )

class Pathway(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolic pathways involving the transformation of metabolites.",
        examples=["glycolysis", "TCA cycle", "photosynthetic electron transport"]
    )

class Genes(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant gene names.",
        examples=["MAP kinase 6", "phytochrome B"]
    )

class AnatomicalStructure(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Anatomical structures in plants.",
        examples=["plant embryo proper", 'lenticel', "root cortex"]
    )

class Species(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant species names.",
        examples=["Arabidopsis thaliana", "Oryza sativa", "Zea mays"]
    )

class ExperimentalCondition(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Experimental conditions in plant studies.",
        examples=['nickel exposure', 'oxygen sensitivity', 'leaf shattering']
    )

class MolecularTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Molecular traits in plants.",
    )

class PlantTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Plant traits in plants.",
        examples=["chromium sensitivity", "mimic response", "stem strength"]
    )

class HumanTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Human traits in plants.",
    )

# Create a main model to hold lists of all extracted entities
# TODO: consider using some NLP for some of the NER tasks. For example
# Genes and Species can be done using AIONER/GNORM2.
# Metabolites can be done using MetaboListem /TABoLiSTM. 
# if applying NLPs, remove some of the attibutes/classes in CustomExtractedEntities as well as in postprocessing/span_adder.py.
class CustomExtractedEntities(SchemicModel):
    """All entities extracted from the text."""
    model_config = ConfigDict(extra="forbid")
    metabolites: List[Metabolite] = Field(default_factory=list, description="List of metabolite mentions.")
    pathways: List[Pathway] = Field(default_factory=list, description="List of pathway mentions.")
    genes: List[Genes] = Field(default_factory=list, description="List of gene mentions.")
    anatomical_structures: List[AnatomicalStructure] = Field(default_factory=list, description="List of anatomical structure mentions.")
    species: List[Species] = Field(default_factory=list, description="List of species mentions.")
    experimental_conditions: List[ExperimentalCondition] = Field(default_factory=list, description="List of experimental condition mentions.")
    molecular_traits: List[MolecularTraits] = Field(default_factory=list, description="List of molecular trait mentions.")
    plant_traits: List[PlantTraits] = Field(default_factory=list, description="List of plant trait mentions.")
    human_traits: List[HumanTraits] = Field(default_factory=list, description="List of human trait mentions.")