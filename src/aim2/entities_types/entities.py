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

class Species(SchemicModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Species names.",
        examples=["Arabidopsis thaliana", "Oryza sativa", "Zea mays", "Homo sapiens", "Mus musculus"]
    )

# Create a main model to hold lists of all extracted entities
class CustomExtractedEntities(SchemicModel):
    """All entities extracted from the text."""
    model_config = ConfigDict(extra="forbid")
    metabolites: List[Metabolite] = Field(default_factory=list, description="List of metabolite mentions.")
    pathways: List[Pathway] = Field(default_factory=list, description="List of pathway mentions.")
    species: List[Species] = Field(default_factory=list, description="List of species mentions.")