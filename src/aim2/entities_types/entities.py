from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional

# -------------------------
# --- ENTITY DEFINITIONS --
# -------------------------

class Metabolite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolites found in plants, including specialized plant compounds, phytohormones, etc.",
        examples=["β-sitosterol", "abscisic acid", "gibberellin"]
    )
    span: Optional[tuple[int, int]] = Field(None, description="Start and end character offsets of the entity.")

class Pathway(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Metabolic pathways involving the transformation of metabolites.",
        examples=["glycolysis", "TCA cycle", "photosynthetic electron transport"]
    )
    span: Optional[tuple[int, int]] = Field(None, description="Start and end character offsets of the entity.")

class Species(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        description="Species names.",
        examples=["Arabidopsis thaliana", "Oryza sativa", "Zea mays", "Homo sapiens", "Mus musculus"]
    )
    span: Optional[tuple[int, int]] = Field(None, description="Start and end character offsets of the entity.")

# Create a main model to hold lists of all extracted entities
class CustomExtractedEntities(BaseModel):
    """All entities extracted from the text."""
    model_config = ConfigDict(extra="forbid")
    metabolites: List[Metabolite] = Field(default_factory=list, description="List of metabolite mentions.")
    pathways: List[Pathway] = Field(default_factory=list, description="List of pathway mentions.")
    species: List[Species] = Field(default_factory=list, description="List of species mentions.")