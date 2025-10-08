from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field
from aim2.entities_types.entities import Compound, Pathway, Genes, AnatomicalStructure, Species, ExperimentalCondition, MolecularTraits, PlantTraits, HumanTraits

# Define the allowed relationship types for each category
PATHWAY_RELATIONS = Literal["made_via", "biosynthesized_via", "degraded_via", "No_Relationship"]
GENE_RELATIONS = Literal["made_by", "biosynthesized_by", "degraded_by", "associated_with", "No_Relationship"]
ANATOMICAL_STRUCTURE_RELATIONS = Literal["made_in", "accumulates_in", "found_in", "present_in", "No_Relationship"]
SPECIES_RELATIONS = Literal["made_in", "accumulates_in", "found_in", "present_in", "No_Relationship"]
EXPERIMENTAL_CONDITION_RELATIONS = Literal["made_in", "accumulates_in", "found_in", "present_in", "No_Relationship"]
MOLECULAR_TRAIT_RELATIONS = Literal["affects", "modulates", "influences", "involved_in", "associated_with", "No_Relationship"]
PLANT_TRAIT_RELATIONS = Literal["affects", "modulates", "influences", "involved_in", "associated_with", "No_Relationship"]
HUMAN_TRAIT_RELATIONS = Literal["affects", "modulates", "influences", "No_Relationship"]

# A Union of all possible object types
ObjectEntityType = Union[Pathway, Genes, AnatomicalStructure, Species, ExperimentalCondition, MolecularTraits, PlantTraits, HumanTraits]

# A Union of all possible predicate types
PredicateType = Union[
    PATHWAY_RELATIONS,
    GENE_RELATIONS,
    ANATOMICAL_STRUCTURE_RELATIONS,
    SPECIES_RELATIONS,
    EXPERIMENTAL_CONDITION_RELATIONS,
    MOLECULAR_TRAIT_RELATIONS,
    PLANT_TRAIT_RELATIONS,
    HUMAN_TRAIT_RELATIONS
]

class Relation(BaseModel):
    """
    Represents a single relationship between a molecular compound (subject)
    and another entity (object).
    """
    subject: Compound = Field(description="The molecular compound that is the subject of the relationship.")
    predicate: PredicateType = Field(description="The relationship between the subject and the object.")
    object: ObjectEntityType = Field(description="The entity that is the object of the relationship.")
    context: Optional[str] = Field(default=None, description="The text passage(s) used by the LLM to determine the relationship.")
    score: Optional[float] = Field(default=None, description="The confidence score for the extracted relationship.")

class ExtractedRelations(BaseModel):
    """
    A container for all the relationships extracted from a document.
    """
    relations: List[Relation] = Field(default_factory=list, description="A list of all relationships found in the document.")