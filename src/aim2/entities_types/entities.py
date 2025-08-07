from pydantic import BaseModel, Field
from typing import List

class Compound(BaseModel):
    name: str = Field(description="A specific compound mentioned in the text.")

class Chemicals(BaseModel):
    name: str = Field(description="A specific chemical mentioned in the text.")

class Metabolite(BaseModel):
    name: str = Field(description="A specific metabolite mentioned in the text.")

class BiologicalProcess(BaseModel):
    name: str = Field(description="A biological process.")

class Trait(BaseModel):
    name: str = Field(description="An observable characteristic or trait.")

class Tissue(BaseModel):
    name: str = Field(description="A specific tissue type mentioned in the text.")

class Species(BaseModel):
    name: str = Field(description="A specific species mentioned in the text.")

# 2. Create a main model to hold lists of all extracted entities
class CustomExtractedEntities(BaseModel):
    """All entities extracted from the text."""
    compound: List[Compound]
    chemicals: List[Chemicals]
    metabolites: List[Metabolite]
    # biological_processes: List[BiologicalProcess]
    traits: List[Trait]
    tissues: List[Tissue]
    species: List[Species]
