"""
Pydantic models for Valheim blueprint data.

These models define the structure for prefabs (building pieces), individual placed
pieces, and complete blueprints. Used throughout the pipeline for validation.
"""

from pydantic import BaseModel, Field
from typing import Literal


class Vector3(BaseModel):
    """A 3D coordinate. Y is up in Valheim's coordinate system."""
    x: float
    y: float
    z: float


class Prefab(BaseModel):
    """
    A Valheim building prefab from the game's database.
    
    Contains the piece's internal name, display name, dimensions, and snap points.
    Snap points are local offsets from the piece center where it can connect to others.
    """
    name: str
    englishName: str
    description: str
    biome: str
    width: float
    height: float
    depth: float
    snapPoints: list[Vector3] | None = None


class Piece(BaseModel):
    """
    A single placed piece in a blueprint.
    
    Position is the center of the piece in world coordinates.
    rotY is rotation around the vertical axis in degrees (0, 90, 180, 270).
    """
    prefab: str
    x: float
    y: float
    z: float
    rotY: Literal[0, 90, 180, 270] = 0


class Blueprint(BaseModel):
    """
    A complete Valheim blueprint containing multiple placed pieces.
    
    This is the final output format that can be imported into Valheim mods
    like PlanBuild or BuildShare.
    """
    name: str = "Generated Blueprint"
    creator: str = "BlueprintGenerator"
    description: str = ""
    category: str = "Misc"
    pieces: list[Piece] = Field(default_factory=list)
