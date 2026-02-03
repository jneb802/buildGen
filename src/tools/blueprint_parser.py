"""
Blueprint Parser - Parse Valheim .blueprint files into structured data.

Blueprint format (PlanBuild/InfinityHammer):
#Name:BuildingName
#Creator:CreatorName
#Description:"optional description"
#Category:Blueprints
#SnapPoints
x;y;z
...
#Pieces
prefab;station;x;y;z;quatX;quatY;quatZ;quatW;data;scaleX;scaleY;scaleZ
"""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPiece:
    """A single piece from a blueprint."""
    prefab: str
    x: float
    y: float
    z: float
    rotY: float  # Euler Y rotation in degrees (0, 90, 180, 270 typically)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class ParsedBlueprint:
    """A parsed blueprint with metadata and pieces."""
    name: str
    creator: str
    description: str
    pieces: list[ParsedPiece]


def quaternion_to_euler_y(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Convert quaternion to Y-axis euler rotation in degrees.
    
    For Valheim buildings, we primarily care about rotation around Y (up) axis.
    Uses: rotY = 2 * atan2(qy, qw) * 180 / pi
    
    This is a simplified extraction assuming rotation is primarily around Y.
    """
    # Handle edge cases for numerical stability
    if abs(qw) < 1e-6:
        # qw near zero means ~180 degree rotation
        if qy >= 0:
            return 180.0
        else:
            return -180.0
    
    # Standard Y-axis extraction from quaternion
    rot_y_rad = 2.0 * math.atan2(qy, qw)
    rot_y_deg = math.degrees(rot_y_rad)
    
    # Normalize to 0-360 range
    rot_y_deg = rot_y_deg % 360
    if rot_y_deg < 0:
        rot_y_deg += 360
    
    return rot_y_deg


def parse_piece_line(line: str) -> ParsedPiece | None:
    """
    Parse a single piece line from a blueprint file.
    
    Format: prefab;station;x;y;z;quatX;quatY;quatZ;quatW;data;scaleX;scaleY;scaleZ
    
    Returns None if line is invalid.
    """
    parts = line.split(";")
    if len(parts) < 10:
        return None
    
    try:
        prefab = parts[0]
        # parts[1] is crafting station, skip
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        qx = float(parts[5])
        qy = float(parts[6])
        qz = float(parts[7])
        qw = float(parts[8])
        # parts[9] is data string, skip
        
        # Scale is optional (parts 10, 11, 12)
        if len(parts) >= 13:
            scale = (float(parts[10]), float(parts[11]), float(parts[12]))
        else:
            scale = (1.0, 1.0, 1.0)
        
        rot_y = quaternion_to_euler_y(qx, qy, qz, qw)
        
        return ParsedPiece(
            prefab=prefab,
            x=x,
            y=y,
            z=z,
            rotY=rot_y,
            scale=scale
        )
    except (ValueError, IndexError):
        return None


def parse_blueprint(path: str | Path) -> ParsedBlueprint:
    """
    Parse a .blueprint file into structured data.
    
    Args:
        path: Path to the .blueprint file
        
    Returns:
        ParsedBlueprint with name, creator, and list of pieces
    """
    path = Path(path)
    
    name = "Unknown"
    creator = "Unknown"
    description = ""
    pieces: list[ParsedPiece] = []
    
    in_pieces_section = False
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse header lines
            if line.startswith("#Name:"):
                name = line[6:]
            elif line.startswith("#Creator:"):
                creator = line[9:]
            elif line.startswith("#Description:"):
                description = line[13:].strip('"')
            elif line == "#Pieces":
                in_pieces_section = True
            elif line.startswith("#"):
                # Other header like #SnapPoints, #Category
                in_pieces_section = False
            elif in_pieces_section:
                # Parse piece line
                piece = parse_piece_line(line)
                if piece:
                    pieces.append(piece)
    
    return ParsedBlueprint(
        name=name,
        creator=creator,
        description=description,
        pieces=pieces
    )
