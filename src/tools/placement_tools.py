"""
Procedural placement tools for the build agent.

These tools generate piece arrays deterministically, shifting coordinate
calculation from the LLM to reliable code. The LLM decides what to build;
these functions handle the math.

Primitive Actions (inspired by APT paper for Minecraft):
- place_piece: The fundamental primitive - places a single piece with snap correction

Composite Actions (built on primitives):
- generate_floor_grid: Tile floor pieces over an area
- generate_wall_line: Place walls along a line with fillers/corners
- generate_roof_slope: Place sloped roof pieces in a row
"""

import json
import math
from dataclasses import dataclass
from typing import Literal

from src.tools.prefab_lookup import get_prefab_details


# ============================================================================
# Snap Point Math (matches Valheim's Player.FindClosestSnapPoints algorithm)
# ============================================================================

# Valheim uses 0.5m snap tolerance in Player.FindClosestSnapPoints
SNAP_TOLERANCE = 0.5


@dataclass
class Vec3:
    """Simple vector for snap point calculations."""
    x: float
    y: float
    z: float
    
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def distance(self, other: "Vec3") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def _rotate_y(point: Vec3, degrees: float) -> Vec3:
    """Rotate a point around Y axis. Matches Unity's quaternion Y rotation."""
    rad = math.radians(degrees)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return Vec3(
        point.x * cos_r - point.z * sin_r,
        point.y,
        point.x * sin_r + point.z * cos_r
    )


def _get_world_snap_points(prefab: str, x: float, y: float, z: float, rotY: float) -> list[Vec3]:
    """
    Get world-space snap points for a piece at a given position/rotation.
    
    Mirrors Valheim's Piece.GetSnapPoints which returns child transforms tagged "snappoint",
    transformed to world space by the piece's position and rotation.
    """
    details = get_prefab_details(prefab)
    if not details or not details.get("snapPoints"):
        return []
    
    piece_pos = Vec3(x, y, z)
    world_points = []
    
    for sp in details["snapPoints"]:
        local = Vec3(sp["x"], sp["y"], sp["z"])
        rotated = _rotate_y(local, rotY)
        world = piece_pos + rotated
        world_points.append(world)
    
    return world_points


def _find_snap_correction(
    prefab: str,
    x: float,
    y: float, 
    z: float,
    rotY: float,
    placed_pieces: list[dict]
) -> tuple[float, float, float, bool, float]:
    """
    Find snap correction for a new piece against already-placed pieces.
    
    Mirrors Valheim's Player.FindClosestSnapPoints algorithm:
    1. Get snap points from the new piece
    2. Get snap points from all placed pieces within range
    3. Find the closest pair within SNAP_TOLERANCE
    4. Return offset to align them
    
    Returns:
        (corrected_x, corrected_y, corrected_z, was_snapped, snap_distance)
    """
    new_snaps = _get_world_snap_points(prefab, x, y, z, rotY)
    if not new_snaps:
        return x, y, z, False, 0.0
    
    best_new = None
    best_placed = None
    best_dist = float("inf")
    
    for placed in placed_pieces:
        placed_snaps = _get_world_snap_points(
            placed["prefab"],
            placed["x"],
            placed["y"],
            placed["z"],
            placed["rotY"]
        )
        for ns in new_snaps:
            for ps in placed_snaps:
                dist = ns.distance(ps)
                if dist < best_dist:
                    best_dist = dist
                    best_new = ns
                    best_placed = ps
    
    # Only snap if within tolerance (Valheim uses 0.5m)
    if best_dist <= SNAP_TOLERANCE and best_new and best_placed:
        # Calculate offset: move piece so its snap point lands on placed snap point
        offset = best_placed - best_new
        return (
            round(x + offset.x, 3),
            round(y + offset.y, 3),
            round(z + offset.z, 3),
            True,
            round(best_dist, 3)
        )
    
    return x, y, z, False, 0.0


# ============================================================================
# Primitive Action: place_piece
# ============================================================================

def place_piece(
    prefab: str,
    x: float,
    y: float,
    z: float,
    rotY: Literal[0, 90, 180, 270],
    placed_pieces: list[dict] | None = None,
    snap: bool = True
) -> dict:
    """
    Place a single piece at (x, y, z) with rotation rotY.
    
    This is the fundamental primitive action.
    All composite tools (generate_floor_grid, etc.) ultimately produce piece dicts
    in the same format this returns.
    
    Args:
        prefab: Exact prefab name (e.g., "stone_floor_2x2")
        x, y, z: World position (piece center)
        rotY: Y-axis rotation in degrees (0, 90, 180, or 270)
        placed_pieces: List of already-placed pieces for snap correction
        snap: Whether to apply snap point correction (default True)
    
    Returns:
        Dict with keys: prefab, x, y, z, rotY, snapped, snap_distance
        
    Snap Behavior (mirrors Valheim's Player.FindClosestSnapPoints):
        - Finds the closest snap point pair between this piece and placed_pieces
        - If distance < 0.5m, adjusts position so snap points align exactly
        - Returns snapped=True and snap_distance if correction was applied
    """
    details = get_prefab_details(prefab)
    if not details:
        return {
            "error": f"Unknown prefab: {prefab}",
            "prefab": prefab,
            "x": x,
            "y": y,
            "z": z,
            "rotY": rotY
        }
    
    final_x, final_y, final_z = x, y, z
    snapped = False
    snap_distance = 0.0
    
    if snap and placed_pieces:
        final_x, final_y, final_z, snapped, snap_distance = _find_snap_correction(
            prefab, x, y, z, rotY, placed_pieces
        )
    
    return {
        "prefab": prefab,
        "x": round(final_x, 3),
        "y": round(final_y, 3),
        "z": round(final_z, 3),
        "rotY": rotY,
        "snapped": snapped,
        "snap_distance": snap_distance
    }


# ============================================================================
# Composite Placement Generators
# ============================================================================

def generate_floor_grid(
    prefab: str,
    width: float,
    depth: float,
    y: float,
    origin_x: float = 0.0,
    origin_z: float = 0.0
) -> list[dict]:
    """
    Generate a grid of floor tiles to cover the specified area.
    
    Args:
        prefab: Floor prefab name (e.g., "stone_floor_2x2")
        width: Total width of floor area (X axis)
        depth: Total depth of floor area (Z axis)
        y: Y position for all floor pieces
        origin_x: X offset for the floor grid origin
        origin_z: Z offset for the floor grid origin
    
    Returns:
        List of piece dicts with prefab, x, y, z, rotY keys.
    """
    details = get_prefab_details(prefab)
    if not details:
        return [{"error": f"Unknown prefab: {prefab}"}]
    
    piece_w = details["width"]
    piece_d = details["depth"]
    
    pieces = []
    cols = max(1, int(round(width / piece_w)))
    rows = max(1, int(round(depth / piece_d)))
    
    for i in range(cols):
        for j in range(rows):
            x = origin_x + piece_w / 2 + i * piece_w
            z = origin_z + piece_d / 2 + j * piece_d
            pieces.append({
                "prefab": prefab,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "rotY": 0
            })
    
    return pieces


def generate_wall_line(
    prefab: str,
    start_x: float,
    start_z: float,
    end_x: float,
    end_z: float,
    y: float,
    rotY: Literal[0, 90, 180, 270],
    filler_prefab: str | None = None,
    corner_prefab: str | None = None,
    corner_y: float | None = None,
    include_start_corner: bool = True,
    include_end_corner: bool = True
) -> list[dict]:
    """
    Generate wall segments along a straight line, optionally with filler pieces and corner posts.
    
    Args:
        prefab: Wall prefab name (primary/larger pieces)
        start_x, start_z: Starting point of the wall line
        end_x, end_z: Ending point of the wall line
        y: Y position (center of wall pieces)
        rotY: Rotation (0=facing +Z, 90=facing +X, 180=facing -Z, 270=facing -X)
        filler_prefab: Optional smaller prefab to fill remaining gaps (e.g., "stone_wall_1x1")
        corner_prefab: Optional pole/pillar prefab for corners (e.g., "wood_pole2")
        corner_y: Y position for corner posts (defaults to wall y if not specified)
        include_start_corner: Place corner at start point (default True)
        include_end_corner: Place corner at end point (default True)
    
    Returns:
        List of piece dicts (walls + optional fillers + optional corners).
    """
    details = get_prefab_details(prefab)
    if not details:
        return [{"error": f"Unknown prefab: {prefab}"}]
    
    piece_w = details["width"]
    
    # Calculate line length and direction.
    dx = end_x - start_x
    dz = end_z - start_z
    length = math.sqrt(dx * dx + dz * dz)
    
    if length < 0.01:
        return [{"error": "Wall line too short (start and end points are the same)"}]
    
    # Normalize direction.
    dir_x = dx / length
    dir_z = dz / length
    
    pieces = []
    
    # Add start corner if requested.
    if corner_prefab and include_start_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            pieces.append({
                "prefab": corner_prefab,
                "x": round(start_x, 3),
                "y": round(corner_y if corner_y is not None else y, 3),
                "z": round(start_z, 3),
                "rotY": 0
            })
    
    # Calculate how many main pieces fit completely.
    main_count = int(length / piece_w)  # floor, not round
    covered = 0.0
    
    # Place main wall pieces.
    for i in range(main_count):
        center_offset = covered + piece_w / 2
        x = start_x + dir_x * center_offset
        z = start_z + dir_z * center_offset
        pieces.append({
            "prefab": prefab,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "rotY": rotY
        })
        covered += piece_w
    
    # Fill remaining gap with filler pieces.
    remaining = length - covered
    if filler_prefab and remaining > 0.1:  # Only fill if gap is significant
        filler_details = get_prefab_details(filler_prefab)
        if filler_details:
            filler_w = filler_details["width"]
            filler_count = max(1, int(round(remaining / filler_w)))
            
            for i in range(filler_count):
                center_offset = covered + (i + 0.5) * (remaining / filler_count)
                x = start_x + dir_x * center_offset
                z = start_z + dir_z * center_offset
                pieces.append({
                    "prefab": filler_prefab,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "rotY": rotY
                })
    elif remaining > 0.1 and main_count == 0:
        # No main pieces fit, place at least one main piece centered
        x = start_x + dir_x * (length / 2)
        z = start_z + dir_z * (length / 2)
        pieces.append({
            "prefab": prefab,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "rotY": rotY
        })
    
    # Add end corner if requested.
    if corner_prefab and include_end_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            pieces.append({
                "prefab": corner_prefab,
                "x": round(end_x, 3),
                "y": round(corner_y if corner_y is not None else y, 3),
                "z": round(end_z, 3),
                "rotY": 0
            })
    
    return pieces


def generate_roof_slope(
    prefab: str,
    start_x: float,
    start_z: float,
    y: float,
    count: int,
    direction: Literal["north", "south", "east", "west"],
    rotY: Literal[0, 90, 180, 270]
) -> list[dict]:
    """
    Generate a row of sloped roof pieces.
    
    Args:
        prefab: Roof prefab name (e.g., "wood_roof_45")
        start_x, start_z: Starting position for first piece
        y: Y position for first piece
        count: Number of roof pieces to place along the row
        direction: Which way the row extends ("north"=+Z, "south"=-Z, "east"=+X, "west"=-X)
        rotY: Rotation of roof pieces (determines slope direction)
    
    Returns:
        List of piece dicts.
    """
    details = get_prefab_details(prefab)
    if not details:
        return [{"error": f"Unknown prefab: {prefab}"}]
    
    piece_w = details["width"]
    
    # Direction vectors for row placement.
    dir_map = {
        "north": (0, 1),
        "south": (0, -1),
        "east": (1, 0),
        "west": (-1, 0),
    }
    dx, dz = dir_map.get(direction, (0, 1))
    
    pieces = []
    for i in range(count):
        x = start_x + i * piece_w * dx
        z = start_z + i * piece_w * dz
        pieces.append({
            "prefab": prefab,
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "rotY": rotY
        })
    
    return pieces


# ============================================================================
# Claude Tool Definitions
# ============================================================================

PLACEMENT_TOOLS = [
    {
        "name": "place_piece",
        "description": """Place a single piece at a specific position. This is the fundamental primitive action.

Automatically applies snap point correction when placed_pieces is provided:
- Finds closest snap point pair between new piece and existing pieces
- If within 0.5m (Valheim's snap tolerance), adjusts position to align exactly
- Returns snapped=true and snap_distance if correction was applied

Use this for:
- Individual decorations, furniture, doors
- Pieces that don't fit regular patterns
- Fine-tuning or corrections after composite tools""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Exact prefab name (e.g., 'stone_floor_2x2', 'wood_door')"
                },
                "x": {
                    "type": "number",
                    "description": "X position (piece center)"
                },
                "y": {
                    "type": "number",
                    "description": "Y position (piece center)"
                },
                "z": {
                    "type": "number",
                    "description": "Z position (piece center)"
                },
                "rotY": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                    "description": "Y-axis rotation in degrees"
                },
                "placed_pieces": {
                    "type": "array",
                    "description": "Already-placed pieces for snap correction. Each item needs: prefab, x, y, z, rotY",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prefab": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "rotY": {"type": "number"}
                        },
                        "required": ["prefab", "x", "y", "z", "rotY"]
                    }
                },
                "snap": {
                    "type": "boolean",
                    "description": "Whether to apply snap correction (default true)"
                }
            },
            "required": ["prefab", "x", "y", "z", "rotY"]
        }
    },
    {
        "name": "generate_floor_grid",
        "description": "Generate a grid of floor tiles covering a rectangular area. Returns piece array with correct positions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Floor prefab name"
                },
                "width": {
                    "type": "number",
                    "description": "Total floor width in meters (X axis)"
                },
                "depth": {
                    "type": "number",
                    "description": "Total floor depth in meters (Z axis)"
                },
                "y": {
                    "type": "number",
                    "description": "Y position for all floor pieces"
                },
                "origin_x": {
                    "type": "number",
                    "description": "X offset for floor origin (default 0)"
                },
                "origin_z": {
                    "type": "number",
                    "description": "Z offset for floor origin (default 0)"
                }
            },
            "required": ["prefab", "width", "depth", "y"]
        }
    },
    {
        "name": "generate_wall_line",
        "description": "Generate wall segments along a straight line, optionally with corner posts at start/end.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Wall prefab name (e.g., 'stone_wall_2x1', 'woodwall')"
                },
                "start_x": {
                    "type": "number",
                    "description": "Starting X position"
                },
                "start_z": {
                    "type": "number",
                    "description": "Starting Z position"
                },
                "end_x": {
                    "type": "number",
                    "description": "Ending X position"
                },
                "end_z": {
                    "type": "number",
                    "description": "Ending Z position"
                },
                "y": {
                    "type": "number",
                    "description": "Y position for wall centers"
                },
                "rotY": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                    "description": "Wall rotation (0=facing +Z, 90=facing +X, 180=facing -Z, 270=facing -X)"
                },
                "filler_prefab": {
                    "type": "string",
                    "description": "Optional smaller wall prefab to fill remaining gaps (e.g., 'stone_wall_1x1' when using 'stone_wall_4x2')"
                },
                "corner_prefab": {
                    "type": "string",
                    "description": "Optional pole/pillar prefab for corners (e.g., 'wood_pole2')"
                },
                "corner_y": {
                    "type": "number",
                    "description": "Y position for corner posts (defaults to wall y)"
                },
                "include_start_corner": {
                    "type": "boolean",
                    "description": "Place corner at start point (default true)"
                },
                "include_end_corner": {
                    "type": "boolean",
                    "description": "Place corner at end point (default true)"
                }
            },
            "required": ["prefab", "start_x", "start_z", "end_x", "end_z", "y", "rotY"]
        }
    },
    {
        "name": "generate_roof_slope",
        "description": "Generate a row of sloped roof pieces extending in a direction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Roof prefab name (e.g., 'wood_roof_45', 'darkwood_roof')"
                },
                "start_x": {
                    "type": "number",
                    "description": "Starting X position"
                },
                "start_z": {
                    "type": "number",
                    "description": "Starting Z position"
                },
                "y": {
                    "type": "number",
                    "description": "Y position for roof pieces"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of roof pieces in the row"
                },
                "direction": {
                    "type": "string",
                    "enum": ["north", "south", "east", "west"],
                    "description": "Direction the row extends (north=+Z, south=-Z, east=+X, west=-X)"
                },
                "rotY": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                    "description": "Rotation of roof pieces (determines slope direction)"
                }
            },
            "required": ["prefab", "start_x", "start_z", "y", "count", "direction", "rotY"]
        }
    }
]


def execute_placement_tool(name: str, args: dict) -> str:
    """
    Execute a placement tool by name and return JSON result.
    
    Called by the build agent when Claude uses a placement tool.
    """
    if name == "place_piece":
        result = place_piece(
            prefab=args["prefab"],
            x=args["x"],
            y=args["y"],
            z=args["z"],
            rotY=args["rotY"],
            placed_pieces=args.get("placed_pieces"),
            snap=args.get("snap", True)
        )
    elif name == "generate_floor_grid":
        result = generate_floor_grid(
            prefab=args["prefab"],
            width=args["width"],
            depth=args["depth"],
            y=args["y"],
            origin_x=args.get("origin_x", 0.0),
            origin_z=args.get("origin_z", 0.0)
        )
    elif name == "generate_wall_line":
        result = generate_wall_line(
            prefab=args["prefab"],
            start_x=args["start_x"],
            start_z=args["start_z"],
            end_x=args["end_x"],
            end_z=args["end_z"],
            y=args["y"],
            rotY=args["rotY"],
            filler_prefab=args.get("filler_prefab"),
            corner_prefab=args.get("corner_prefab"),
            corner_y=args.get("corner_y"),
            include_start_corner=args.get("include_start_corner", True),
            include_end_corner=args.get("include_end_corner", True)
        )
    elif name == "generate_roof_slope":
        result = generate_roof_slope(
            prefab=args["prefab"],
            start_x=args["start_x"],
            start_z=args["start_z"],
            y=args["y"],
            count=args["count"],
            direction=args["direction"],
            rotY=args["rotY"]
        )
    else:
        result = {"error": f"Unknown placement tool: {name}"}
    
    return json.dumps(result, indent=2)
