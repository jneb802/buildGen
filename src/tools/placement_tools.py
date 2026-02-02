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
from functools import lru_cache
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
# Cached Snap Point Lookups
# ============================================================================

@lru_cache(maxsize=256)
def _get_local_snap_points(prefab: str) -> tuple[tuple[float, float, float], ...]:
    """
    Get LOCAL snap points for a prefab (cached).
    
    Returns a tuple of (x, y, z) tuples for hashability/caching.
    These are the raw snap point offsets from the prefab center.
    """
    details = get_prefab_details(prefab)
    if not details or not details.get("snapPoints"):
        return ()
    return tuple((sp["x"], sp["y"], sp["z"]) for sp in details["snapPoints"])


def _get_world_snap_points_cached(prefab: str, x: float, y: float, z: float, rotY: float) -> list[Vec3]:
    """
    Get world-space snap points using cached local points.
    
    More efficient than _get_world_snap_points for repeated lookups.
    """
    local_points = _get_local_snap_points(prefab)
    if not local_points:
        return []
    
    piece_pos = Vec3(x, y, z)
    world_points = []
    
    for lx, ly, lz in local_points:
        local = Vec3(lx, ly, lz)
        rotated = _rotate_y(local, rotY)
        world = piece_pos + rotated
        world_points.append(world)
    
    return world_points


# ============================================================================
# O(1) Single-Piece Snap Helper
# ============================================================================

def _snap_to_piece(
    prefab: str,
    x: float,
    y: float,
    z: float,
    rotY: float,
    target: dict
) -> tuple[float, float, float, bool]:
    """
    Snap a new piece to a single target piece. O(1) complexity.
    
    Used for chain-snapping in composite tools where each piece
    only needs to snap to the previous piece.
    
    Args:
        prefab: Prefab name of the new piece
        x, y, z: Initial position of the new piece
        rotY: Rotation of the new piece
        target: Single piece dict to snap to (must have prefab, x, y, z, rotY)
    
    Returns:
        (corrected_x, corrected_y, corrected_z, was_snapped)
    """
    new_snaps = _get_world_snap_points_cached(prefab, x, y, z, rotY)
    if not new_snaps:
        return x, y, z, False
    
    target_snaps = _get_world_snap_points_cached(
        target["prefab"],
        target["x"],
        target["y"],
        target["z"],
        target["rotY"]
    )
    if not target_snaps:
        return x, y, z, False
    
    # Find closest snap point pair
    best_new = None
    best_target = None
    best_dist = float("inf")
    
    for ns in new_snaps:
        for ts in target_snaps:
            dist = ns.distance(ts)
            if dist < best_dist:
                best_dist = dist
                best_new = ns
                best_target = ts
    
    # Snap if within tolerance
    if best_dist <= SNAP_TOLERANCE and best_new and best_target:
        offset = best_target - best_new
        return (
            round(x + offset.x, 3),
            round(y + offset.y, 3),
            round(z + offset.z, 3),
            True
        )
    
    return x, y, z, False


def _snap_to_anchor_pieces(
    prefab: str,
    x: float,
    y: float,
    z: float,
    rotY: float,
    anchors: list[dict]
) -> tuple[float, float, float, bool]:
    """
    Snap a piece to the closest snap point among multiple anchor pieces.
    
    Used for snapping the first piece in a composite tool to existing
    structure (e.g., first wall to floor edge).
    
    Args:
        prefab: Prefab name of the new piece
        x, y, z: Initial position
        rotY: Rotation
        anchors: List of piece dicts to potentially snap to
    
    Returns:
        (corrected_x, corrected_y, corrected_z, was_snapped)
    """
    if not anchors:
        return x, y, z, False
    
    new_snaps = _get_world_snap_points_cached(prefab, x, y, z, rotY)
    if not new_snaps:
        return x, y, z, False
    
    best_new = None
    best_anchor = None
    best_dist = float("inf")
    
    for anchor in anchors:
        anchor_snaps = _get_world_snap_points_cached(
            anchor["prefab"],
            anchor["x"],
            anchor["y"],
            anchor["z"],
            anchor["rotY"]
        )
        for ns in new_snaps:
            for as_ in anchor_snaps:
                dist = ns.distance(as_)
                if dist < best_dist:
                    best_dist = dist
                    best_new = ns
                    best_anchor = as_
    
    if best_dist <= SNAP_TOLERANCE and best_new and best_anchor:
        offset = best_anchor - best_new
        return (
            round(x + offset.x, 3),
            round(y + offset.y, 3),
            round(z + offset.z, 3),
            True
        )
    
    return x, y, z, False


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
    snap: bool = False
) -> dict:
    """
    Place a single piece at (x, y, z) with rotation rotY.
    
    Use this for individual pieces that don't fit composite tools:
    - Doors, arches, stairs
    - Decorations and furniture
    - One-off pieces needing precise placement
    
    For walls, floors, and roofs, prefer the composite tools (generate_wall_line,
    generate_floor_grid, generate_roof_slope) which handle snapping internally.
    
    Args:
        prefab: Exact prefab name (e.g., "stone_floor_2x2")
        x, y, z: World position (piece center)
        rotY: Y-axis rotation in degrees (0, 90, 180, or 270)
        placed_pieces: List of pieces to snap to (only used if snap=True)
        snap: Whether to apply snap correction (default False - use for doors/decorations)
    
    Returns:
        Dict with keys: prefab, x, y, z, rotY, snapped, snap_distance
        
    Snap Behavior (when snap=True and placed_pieces provided):
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
    include_end_corner: bool = True,
    anchor_pieces: list[dict] | None = None
) -> list[dict]:
    """
    Generate wall segments along a straight line, optionally with filler pieces and corner posts.
    
    Snapping behavior:
    - First wall piece snaps to anchor_pieces if provided (e.g., floor edges)
    - Subsequent pieces chain-snap to the previous piece (O(1) per piece)
    - This ensures all pieces connect properly in Valheim
    
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
        anchor_pieces: Optional list of pieces to snap first wall to (e.g., floor pieces)
    
    Returns:
        List of piece dicts (walls + optional fillers + optional corners), all snapped.
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
    last_piece = None  # Track last piece for chain snapping
    
    # Add start corner if requested.
    if corner_prefab and include_start_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            corner_x = start_x
            corner_z = start_z
            corner_y_pos = corner_y if corner_y is not None else y
            
            # Snap corner to anchors if provided
            if anchor_pieces:
                corner_x, corner_y_pos, corner_z, _ = _snap_to_anchor_pieces(
                    corner_prefab, corner_x, corner_y_pos, corner_z, 0, anchor_pieces
                )
            
            corner_piece = {
                "prefab": corner_prefab,
                "x": round(corner_x, 3),
                "y": round(corner_y_pos, 3),
                "z": round(corner_z, 3),
                "rotY": 0
            }
            pieces.append(corner_piece)
            last_piece = corner_piece
    
    # Calculate how many main pieces fit completely.
    main_count = int(length / piece_w)  # floor, not round
    covered = 0.0
    
    # Place main wall pieces with chain snapping.
    for i in range(main_count):
        center_offset = covered + piece_w / 2
        wall_x = start_x + dir_x * center_offset
        wall_z = start_z + dir_z * center_offset
        wall_y = y
        
        # Snap: first piece to anchors, subsequent pieces to previous
        if i == 0 and last_piece is None and anchor_pieces:
            wall_x, wall_y, wall_z, _ = _snap_to_anchor_pieces(
                prefab, wall_x, wall_y, wall_z, rotY, anchor_pieces
            )
        elif last_piece:
            wall_x, wall_y, wall_z, _ = _snap_to_piece(
                prefab, wall_x, wall_y, wall_z, rotY, last_piece
            )
        
        wall_piece = {
            "prefab": prefab,
            "x": round(wall_x, 3),
            "y": round(wall_y, 3),
            "z": round(wall_z, 3),
            "rotY": rotY
        }
        pieces.append(wall_piece)
        last_piece = wall_piece
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
                filler_x = start_x + dir_x * center_offset
                filler_z = start_z + dir_z * center_offset
                filler_y = y
                
                # Chain snap filler to previous piece
                if last_piece:
                    filler_x, filler_y, filler_z, _ = _snap_to_piece(
                        filler_prefab, filler_x, filler_y, filler_z, rotY, last_piece
                    )
                
                filler_piece = {
                    "prefab": filler_prefab,
                    "x": round(filler_x, 3),
                    "y": round(filler_y, 3),
                    "z": round(filler_z, 3),
                    "rotY": rotY
                }
                pieces.append(filler_piece)
                last_piece = filler_piece
    elif remaining > 0.1 and main_count == 0:
        # No main pieces fit, place at least one main piece centered
        wall_x = start_x + dir_x * (length / 2)
        wall_z = start_z + dir_z * (length / 2)
        wall_y = y
        
        if anchor_pieces:
            wall_x, wall_y, wall_z, _ = _snap_to_anchor_pieces(
                prefab, wall_x, wall_y, wall_z, rotY, anchor_pieces
            )
        
        wall_piece = {
            "prefab": prefab,
            "x": round(wall_x, 3),
            "y": round(wall_y, 3),
            "z": round(wall_z, 3),
            "rotY": rotY
        }
        pieces.append(wall_piece)
        last_piece = wall_piece
    
    # Add end corner if requested.
    if corner_prefab and include_end_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            corner_x = end_x
            corner_z = end_z
            corner_y_pos = corner_y if corner_y is not None else y
            
            # Chain snap corner to last wall piece
            if last_piece:
                corner_x, corner_y_pos, corner_z, _ = _snap_to_piece(
                    corner_prefab, corner_x, corner_y_pos, corner_z, 0, last_piece
                )
            
            corner_piece = {
                "prefab": corner_prefab,
                "x": round(corner_x, 3),
                "y": round(corner_y_pos, 3),
                "z": round(corner_z, 3),
                "rotY": 0
            }
            pieces.append(corner_piece)
    
    return pieces


def generate_roof_slope(
    prefab: str,
    start_x: float,
    start_z: float,
    y: float,
    count: int,
    direction: Literal["north", "south", "east", "west"],
    rotY: Literal[0, 90, 180, 270],
    anchor_pieces: list[dict] | None = None
) -> list[dict]:
    """
    Generate a row of sloped roof pieces.
    
    Snapping behavior:
    - First roof piece snaps to anchor_pieces if provided (e.g., wall tops)
    - Subsequent pieces chain-snap to the previous piece (O(1) per piece)
    - This ensures all pieces connect properly in Valheim
    
    Args:
        prefab: Roof prefab name (e.g., "wood_roof_45")
        start_x, start_z: Starting position for first piece
        y: Y position for first piece
        count: Number of roof pieces to place along the row
        direction: Which way the row extends ("north"=+Z, "south"=-Z, "east"=+X, "west"=-X)
        rotY: Rotation of roof pieces (determines slope direction)
        anchor_pieces: Optional list of pieces to snap first roof piece to (e.g., walls)
    
    Returns:
        List of piece dicts, all snapped.
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
    last_piece = None
    
    for i in range(count):
        roof_x = start_x + i * piece_w * dx
        roof_z = start_z + i * piece_w * dz
        roof_y = y
        
        # Snap: first piece to anchors, subsequent pieces to previous
        if i == 0 and anchor_pieces:
            roof_x, roof_y, roof_z, _ = _snap_to_anchor_pieces(
                prefab, roof_x, roof_y, roof_z, rotY, anchor_pieces
            )
        elif last_piece:
            roof_x, roof_y, roof_z, _ = _snap_to_piece(
                prefab, roof_x, roof_y, roof_z, rotY, last_piece
            )
        
        roof_piece = {
            "prefab": prefab,
            "x": round(roof_x, 3),
            "y": round(roof_y, 3),
            "z": round(roof_z, 3),
            "rotY": rotY
        }
        pieces.append(roof_piece)
        last_piece = roof_piece
    
    return pieces


# ============================================================================
# Claude Tool Definitions
# ============================================================================

PLACEMENT_TOOLS = [
    {
        "name": "place_piece",
        "description": """Place a single piece at a specific position.

Use for pieces that don't fit composite tools:
- Doors, arches, stairs
- Decorations and furniture
- One-off pieces needing precise placement

For walls/floors/roofs, prefer composite tools (generate_wall_line, generate_floor_grid,
generate_roof_slope) which handle snapping internally and are more efficient.

Snap correction (optional, default off):
- Set snap=true and provide placed_pieces to snap to existing structure
- Finds closest snap point pair within 0.5m tolerance""",
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
                    "description": "Pieces to snap to (only used if snap=true). Each item needs: prefab, x, y, z, rotY",
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
                    "description": "Whether to apply snap correction (default false)"
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
        "description": """Generate wall segments along a straight line with automatic snapping.

Snapping behavior (handled internally):
- First wall snaps to anchor_pieces if provided (e.g., floor edges)
- Subsequent walls chain-snap to the previous wall
- All pieces are returned already snapped - no manual snap correction needed

Use anchor_pieces to connect walls to existing structure (floors, other walls).""",
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
                },
                "anchor_pieces": {
                    "type": "array",
                    "description": "Pieces to snap the first wall to (e.g., floor pieces). Each item needs: prefab, x, y, z, rotY",
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
                }
            },
            "required": ["prefab", "start_x", "start_z", "end_x", "end_z", "y", "rotY"]
        }
    },
    {
        "name": "generate_roof_slope",
        "description": """Generate a row of sloped roof pieces with automatic snapping.

Snapping behavior (handled internally):
- First roof piece snaps to anchor_pieces if provided (e.g., wall tops)
- Subsequent pieces chain-snap to the previous piece
- All pieces are returned already snapped - no manual snap correction needed

Use anchor_pieces to connect roof to walls.""",
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
                },
                "anchor_pieces": {
                    "type": "array",
                    "description": "Pieces to snap the first roof piece to (e.g., wall pieces). Each item needs: prefab, x, y, z, rotY",
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
            snap=args.get("snap", False)
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
            include_end_corner=args.get("include_end_corner", True),
            anchor_pieces=args.get("anchor_pieces")
        )
    elif name == "generate_roof_slope":
        result = generate_roof_slope(
            prefab=args["prefab"],
            start_x=args["start_x"],
            start_z=args["start_z"],
            y=args["y"],
            count=args["count"],
            direction=args["direction"],
            rotY=args["rotY"],
            anchor_pieces=args.get("anchor_pieces")
        )
    else:
        result = {"error": f"Unknown placement tool: {name}"}
    
    return json.dumps(result, indent=2)
