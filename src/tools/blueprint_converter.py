"""
Convert JSON blueprint to Valheim .blueprint file format.

The .blueprint format is a text file used by mods like PlanBuild and InfinityHammer.
It contains piece data with quaternion rotations and optional ZDO data.
"""

import math
from pathlib import Path

from src.models import Blueprint, Piece


def degrees_to_quaternion(rot_y: float) -> tuple[float, float, float, float]:
    """
    Convert Y-axis rotation in degrees to quaternion (x, y, z, w).
    
    Valheim uses Unity's quaternion format. For Y-axis rotation:
    - qx = 0
    - qy = sin(angle/2)
    - qz = 0
    - qw = cos(angle/2)
    """
    rad = math.radians(rot_y) / 2
    qx = 0.0
    qy = round(math.sin(rad), 3)
    qz = 0.0
    qw = round(math.cos(rad), 3)
    return (qx, qy, qz, qw)


def format_piece_line(piece: Piece) -> str:
    """
    Format a single piece as a .blueprint line.
    
    Format: prefab;;x;y;z;qx;qy;qz;qw;;scaleX;scaleY;scaleZ;zdoData
    
    We use scale 1;1;1 and empty ZDO data for generated pieces.
    """
    qx, qy, qz, qw = degrees_to_quaternion(piece.rotY)
    
    # Round positions to 3 decimal places for cleaner output.
    x = round(piece.x, 3)
    y = round(piece.y, 3)
    z = round(piece.z, 3)
    
    # Format: prefab;;x;y;z;qx;qy;qz;qw;;scaleX;scaleY;scaleZ;
    # The trailing semicolon indicates no ZDO data (empty).
    return f"{piece.prefab};;{x};{y};{z};{qx};{qy};{qz};{qw};;1;1;1;"


def convert_to_blueprint_format(blueprint: Blueprint) -> str:
    """
    Convert a Blueprint object to .blueprint file format string.
    
    The format has:
    - Header with metadata
    - #SnapPoints section (empty for generated blueprints)
    - #Pieces section with one piece per line
    """
    lines = []
    
    # Header section.
    lines.append(f"#Name:{blueprint.name}")
    lines.append(f"#Creator:{blueprint.creator}")
    lines.append(f"#Description:{blueprint.description}")
    lines.append(f"#Category:{blueprint.category}")
    lines.append("#Center:")
    lines.append("#Coordinates:0,0,0")
    lines.append("#Rotation:0,0,0")
    
    # Snap points section (empty for generated blueprints).
    lines.append("#SnapPoints")
    
    # Pieces section.
    lines.append("#Pieces")
    for piece in blueprint.pieces:
        lines.append(format_piece_line(piece))
    
    return "\n".join(lines)


def save_blueprint_file(blueprint: Blueprint, output_path: Path) -> None:
    """Save a Blueprint as a .blueprint file."""
    content = convert_to_blueprint_format(blueprint)
    output_path.write_text(content)
