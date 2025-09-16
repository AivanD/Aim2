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
    Level_5: Optional[str] = Field(default=None, description="Classyfire Level 5 for the compound.")

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
    Natural_product_class: Optional[Natural_Product_Class] = Field(default=None, description="Natural product class and superclass for the compound.")
    Classyfire: Optional[ClassyFire] = Field(default=None, description="Classyfire classification for the compound.")

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

class HumanTraits(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Human traits in plants.",
    )
    spans: Optional[List[List[int]]] = Field(
        default=None,
        description="Text spans where the human trait is mentioned in the passage."
    )

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