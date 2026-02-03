"""
Procedural placement tools for the build agent.

These tools generate piece arrays deterministically, shifting coordinate
calculation from the LLM to reliable code. The LLM decides what to build;
these functions handle the math.

Primitive Actions (inspired by APT paper for Minecraft):
- place_piece: The fundamental primitive - places a single piece with snap correction

Composite Actions (built on primitives):
- generate_floor_grid: Tile floor pieces over an area
- generate_floor_walls: Generate all 4 walls for a floor with openings support
- generate_wall: Place a single wall segment with proper height (internal use)
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
    
    For walls, floors, and roofs, prefer the composite tools (generate_wall,
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
    
    # Use snap spacing for floor tiling, not bounding box
    snap_w = _get_snap_spacing(details, "x")
    snap_d = _get_snap_spacing(details, "z")
    
    pieces = []
    cols = max(1, int(round(width / snap_w)))
    rows = max(1, int(round(depth / snap_d)))
    
    for i in range(cols):
        for j in range(rows):
            x = origin_x + snap_w / 2 + i * snap_w
            z = origin_z + snap_d / 2 + j * snap_d
            pieces.append({
                "prefab": prefab,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "rotY": 0
            })
    
    return pieces


def _get_snap_spacing(prefab_details: dict, axis: str = "y") -> float:
    """
    Get snap spacing for a prefab from its snap points along the specified axis.
    
    Valheim pieces connect by aligning snap points. This returns the distance
    between min and max snap points on the given axis.
    
    Args:
        prefab_details: The prefab details dict
        axis: "x", "y", or "z" 
    
    Falls back to bounding box dimension if no snap points exist.
    """
    dimension_map = {"x": "width", "y": "height", "z": "depth"}
    fallback = prefab_details.get(dimension_map.get(axis, "height"), 1.0)
    
    snap_points = prefab_details.get("snapPoints")
    if not snap_points:
        return fallback
    
    values = [sp[axis] for sp in snap_points]
    if not values:
        return fallback
    
    spacing = max(values) - min(values)
    
    # If spacing is 0 or very small, fall back to bounding box
    return spacing if spacing > 0.1 else fallback


def _get_snap_height(prefab_details: dict) -> float:
    """
    Get vertical snap spacing for a prefab from its snap points.
    
    Valheim walls stack by connecting top snap points to bottom snap points.
    This returns the vertical distance between them (e.g., 2.0m for stone walls).
    
    Falls back to bounding box height if no snap points exist.
    """
    return _get_snap_spacing(prefab_details, "y")


def _get_bottom_snap_offset(prefab_details: dict) -> float:
    """
    Get the Y offset from piece center to its bottom snap point.
    
    This is used to position walls so their bottom snap point sits on the floor.
    For example, stone_wall_4x2 has bottom snaps at y=-1.0, so offset is -1.0.
    """
    snap_points = prefab_details.get("snapPoints")
    if not snap_points:
        return -prefab_details["height"] / 2  # fallback: assume center-origin
    
    y_values = [sp["y"] for sp in snap_points]
    return min(y_values) if y_values else -prefab_details["height"] / 2


def generate_wall(
    prefab: str,
    start_x: float,
    start_z: float,
    end_x: float,
    end_z: float,
    base_y: float,
    height: float,
    rotY: Literal[0, 90, 180, 270],
    filler_prefab: str | None = None,
    corner_prefab: str | None = None,
    corner_y: float | None = None,
    include_start_corner: bool = True,
    include_end_corner: bool = True,
    anchor_pieces: list[dict] | None = None
) -> list[dict]:
    """
    Generate a complete wall with proper height by stacking rows of wall pieces.
    
    This function places wall pieces both horizontally (along the line) and vertically
    (stacking rows to reach the target height). Stacking uses SNAP POINTS, not bounding
    box dimensions, to match Valheim's in-game behavior.
    
    For example, a 6m tall wall using stone_wall_4x2 (snap height 2m) generates 3 rows.
    
    Snapping behavior:
    - First wall piece snaps to anchor_pieces if provided (e.g., floor edges)
    - Subsequent pieces chain-snap to the previous piece (O(1) per piece)
    - Rows stack vertically using snap point spacing
    
    Args:
        prefab: Wall prefab name (primary/larger pieces)
        start_x, start_z: Starting point of the wall line
        end_x, end_z: Ending point of the wall line
        base_y: Y position of the floor surface the wall sits on
        height: Target wall height in meters (will stack rows to achieve this)
        rotY: Rotation (0=facing +Z, 90=facing +X, 180=facing -Z, 270=facing -X)
        filler_prefab: Optional smaller prefab to fill remaining horizontal gaps
        corner_prefab: Optional pole/pillar prefab for corners (e.g., "wood_pole2")
        corner_y: Y position for corner posts (defaults to base_y if not specified)
        include_start_corner: Place corner at start point (default True)
        include_end_corner: Place corner at end point (default True)
        anchor_pieces: Optional list of pieces to snap first wall to (e.g., floor pieces)
    
    Returns:
        List of piece dicts (walls + optional fillers + optional corners), all snapped.
    """
    details = get_prefab_details(prefab)
    if not details:
        return [{"error": f"Unknown prefab: {prefab}"}]
    
    # Use snap point spacing for horizontal tiling (X-axis for walls)
    snap_w = _get_snap_spacing(details, "x")
    # Use snap point spacing for vertical stacking, not bounding box height
    snap_h = _get_snap_height(details)
    bottom_snap_offset = _get_bottom_snap_offset(details)
    
    # Calculate line length and direction.
    dx = end_x - start_x
    dz = end_z - start_z
    length = math.sqrt(dx * dx + dz * dz)
    
    if length < 0.01:
        return [{"error": "Wall too short (start and end points are the same)"}]
    
    # Normalize direction.
    dir_x = dx / length
    dir_z = dz / length
    
    # Calculate how many vertical rows we need to reach target height.
    # Use snap height (from snap points), not bounding box height.
    num_rows = max(1, int(math.ceil(height / snap_h)))
    
    pieces = []
    
    # Add corners if requested (full height).
    if corner_prefab and include_start_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            corner_snap_h = _get_snap_height(corner_details)
            corner_bottom_offset = _get_bottom_snap_offset(corner_details)
            # Stack corner poles to match wall height using snap spacing.
            corner_rows = max(1, int(math.ceil(height / corner_snap_h)))
            for row in range(corner_rows):
                corner_x = start_x
                corner_z = start_z
                # Position: floor + offset to put bottom snap at floor + row offset
                corner_y_pos = base_y - corner_bottom_offset + row * corner_snap_h
                if corner_y is not None and row == 0:
                    corner_y_pos = corner_y
                
                # Snap first corner piece to anchors if provided
                if row == 0 and anchor_pieces:
                    corner_x, corner_y_pos, corner_z, _ = _snap_to_anchor_pieces(
                        corner_prefab, corner_x, corner_y_pos, corner_z, 0, anchor_pieces
                    )
                elif row > 0 and pieces:
                    # Snap to the corner piece below
                    corner_x, corner_y_pos, corner_z, _ = _snap_to_piece(
                        corner_prefab, corner_x, corner_y_pos, corner_z, 0, pieces[-1]
                    )
                
                corner_piece = {
                    "prefab": corner_prefab,
                    "x": round(corner_x, 3),
                    "y": round(corner_y_pos, 3),
                    "z": round(corner_z, 3),
                    "rotY": 0
                }
                pieces.append(corner_piece)
    
    # Calculate how many main pieces fit horizontally using snap spacing.
    main_count = int(length / snap_w)  # floor, not round
    
    # Get filler snap height if provided.
    filler_snap_h = snap_h  # Default to same snap height as main prefab
    if filler_prefab:
        filler_details = get_prefab_details(filler_prefab)
        if filler_details:
            filler_snap_h = _get_snap_height(filler_details)
    
    # Place wall pieces row by row (bottom to top).
    for row in range(num_rows):
        # Position wall so its bottom snap point sits on the floor (or stacked snap point)
        # row_y = floor + offset_to_center + row * snap_spacing
        row_y = base_y - bottom_snap_offset + row * snap_h
        covered = 0.0
        last_piece_in_row = None
        
        # For first row, we may snap to anchor_pieces.
        # For subsequent rows, we snap to the row below.
        row_anchor = anchor_pieces if row == 0 else None
        
        # Place main wall pieces along this row.
        for i in range(main_count):
            center_offset = covered + snap_w / 2
            wall_x = start_x + dir_x * center_offset
            wall_z = start_z + dir_z * center_offset
            wall_y = row_y
            
            # Snap logic
            if i == 0 and row == 0 and anchor_pieces:
                # First piece of first row: snap to floor/anchors
                wall_x, wall_y, wall_z, _ = _snap_to_anchor_pieces(
                    prefab, wall_x, wall_y, wall_z, rotY, anchor_pieces
                )
            elif i == 0 and row > 0:
                # First piece of subsequent row: snap to piece below
                # Find the corresponding piece in the previous row
                prev_row_start = len(pieces) - main_count if main_count > 0 else len(pieces) - 1
                if prev_row_start >= 0 and prev_row_start < len(pieces):
                    wall_x, wall_y, wall_z, _ = _snap_to_piece(
                        prefab, wall_x, wall_y, wall_z, rotY, pieces[prev_row_start]
                    )
            elif last_piece_in_row:
                # Subsequent pieces in row: snap to previous piece
                wall_x, wall_y, wall_z, _ = _snap_to_piece(
                    prefab, wall_x, wall_y, wall_z, rotY, last_piece_in_row
                )
            
            wall_piece = {
                "prefab": prefab,
                "x": round(wall_x, 3),
                "y": round(wall_y, 3),
                "z": round(wall_z, 3),
                "rotY": rotY
            }
            pieces.append(wall_piece)
            last_piece_in_row = wall_piece
            covered += snap_w
        
        # Fill remaining horizontal gap with filler pieces.
        remaining = length - covered
        if filler_prefab and remaining > 0.1:
            filler_details = get_prefab_details(filler_prefab)
            if filler_details:
                filler_snap_w = _get_snap_spacing(filler_details, "x")
                filler_count = max(1, int(round(remaining / filler_snap_w)))
                
                for i in range(filler_count):
                    center_offset = covered + (i + 0.5) * (remaining / filler_count)
                    filler_x = start_x + dir_x * center_offset
                    filler_z = start_z + dir_z * center_offset
                    filler_y = row_y
                    
                    if last_piece_in_row:
                        filler_x, filler_y, filler_z, _ = _snap_to_piece(
                            filler_prefab, filler_x, filler_y, filler_z, rotY, last_piece_in_row
                        )
                    
                    filler_piece = {
                        "prefab": filler_prefab,
                        "x": round(filler_x, 3),
                        "y": round(filler_y, 3),
                        "z": round(filler_z, 3),
                        "rotY": rotY
                    }
                    pieces.append(filler_piece)
                    last_piece_in_row = filler_piece
        elif remaining > 0.1 and main_count == 0:
            # No main pieces fit, place at least one main piece centered
            wall_x = start_x + dir_x * (length / 2)
            wall_z = start_z + dir_z * (length / 2)
            wall_y = row_y
            
            if row == 0 and anchor_pieces:
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
            last_piece_in_row = wall_piece
    
    # Add end corner if requested (full height).
    if corner_prefab and include_end_corner:
        corner_details = get_prefab_details(corner_prefab)
        if corner_details:
            corner_snap_h = _get_snap_height(corner_details)
            corner_bottom_offset = _get_bottom_snap_offset(corner_details)
            corner_rows = max(1, int(math.ceil(height / corner_snap_h)))
            for row in range(corner_rows):
                corner_x = end_x
                corner_z = end_z
                corner_y_pos = base_y - corner_bottom_offset + row * corner_snap_h
                if corner_y is not None and row == 0:
                    corner_y_pos = corner_y
                
                # Snap to last wall piece or previous corner
                if pieces:
                    corner_x, corner_y_pos, corner_z, _ = _snap_to_piece(
                        corner_prefab, corner_x, corner_y_pos, corner_z, 0, pieces[-1]
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


def generate_floor_walls(
    prefab: str,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    base_y: float,
    height: float,
    filler_prefab: str | None = None,
    openings: list[dict] | None = None,
    anchor_pieces: list[dict] | None = None
) -> list[dict]:
    """
    Generate all four walls for a rectangular floor in a single call.
    
    This is the primary wall generation tool. It creates north, east, south, and west
    walls with proper height by stacking rows of wall pieces. Handles door/window
    openings automatically.
    
    IMPORTANT: Use height=6 or more for typical interior walls.
    
    Args:
        prefab: Wall prefab name (e.g., "stone_wall_4x2")
        x_min, x_max: X bounds of the floor (walls placed at edges)
        z_min, z_max: Z bounds of the floor (walls placed at edges)
        base_y: Y position of the floor surface the walls sit on
        height: Target wall height in meters (typically 6 for interior walls)
        filler_prefab: Optional smaller prefab to fill horizontal gaps
        openings: Optional list of openings (doors, arches, windows). Each dict:
            - wall: "north", "east", "south", or "west"
            - position: coordinate along wall (x for north/south, z for east/west)
            - prefab: opening piece prefab name (e.g., "stone_arch", "wood_door")
            - width: width of opening in meters (default: prefab width)
        anchor_pieces: Optional list of pieces to snap walls to (e.g., floor pieces)
    
    Returns:
        List of piece dicts for all four walls plus any opening pieces.
    
    Example:
        generate_floor_walls(
            prefab="stone_wall_4x2",
            x_min=-5, x_max=5,
            z_min=-5, z_max=5,
            base_y=0.5,
            height=6,
            filler_prefab="stone_wall_2x1",
            openings=[{"wall": "south", "position": 0, "prefab": "stone_arch"}]
        )
    """
    pieces = []
    openings = openings or []
    
    # Helper to get openings for a specific wall
    def get_wall_openings(wall_name: str) -> list[dict]:
        return [o for o in openings if o.get("wall") == wall_name]
    
    # Helper to generate a wall segment (handles splitting for openings)
    def generate_wall_segment(
        start_x: float, start_z: float,
        end_x: float, end_z: float,
        rotY: int,
        wall_openings: list[dict],
        is_x_axis: bool  # True for north/south walls (vary in X), False for east/west (vary in Z)
    ) -> list[dict]:
        segment_pieces = []
        
        if not wall_openings:
            # No openings - generate full wall
            wall_pieces = generate_wall(
                prefab=prefab,
                start_x=start_x, start_z=start_z,
                end_x=end_x, end_z=end_z,
                base_y=base_y, height=height,
                rotY=rotY,
                filler_prefab=filler_prefab,
                include_start_corner=False,
                include_end_corner=False,
                anchor_pieces=anchor_pieces if not pieces else None
            )
            segment_pieces.extend(wall_pieces)
        else:
            # Sort openings by position
            sorted_openings = sorted(wall_openings, key=lambda o: o.get("position", 0))
            
            # Current position along the wall
            if is_x_axis:
                current_pos = start_x
                wall_end = end_x
                fixed_coord = start_z  # Z is fixed for north/south walls
            else:
                current_pos = start_z
                wall_end = end_z
                fixed_coord = start_x  # X is fixed for east/west walls
            
            for opening in sorted_openings:
                open_pos = opening.get("position", 0)
                open_prefab = opening.get("prefab", "stone_arch")
                
                # Get opening dimensions using snap points
                open_details = get_prefab_details(open_prefab)
                open_width = opening.get("width", open_details["width"] if open_details else 2.0)
                # Use snap point offset for Y positioning, not bounding box height
                open_bottom_offset = _get_bottom_snap_offset(open_details) if open_details else -1.0
                
                open_start = open_pos - open_width / 2
                open_end = open_pos + open_width / 2
                
                # Generate wall segment before opening (if there's space)
                if is_x_axis:
                    if current_pos < open_start - 0.1:
                        wall_pieces = generate_wall(
                            prefab=prefab,
                            start_x=current_pos, start_z=fixed_coord,
                            end_x=open_start, end_z=fixed_coord,
                            base_y=base_y, height=height,
                            rotY=rotY,
                            filler_prefab=filler_prefab,
                            include_start_corner=False,
                            include_end_corner=False,
                            anchor_pieces=anchor_pieces if not pieces and not segment_pieces else None
                        )
                        segment_pieces.extend(wall_pieces)
                else:
                    if current_pos < open_start - 0.1:
                        wall_pieces = generate_wall(
                            prefab=prefab,
                            start_x=fixed_coord, start_z=current_pos,
                            end_x=fixed_coord, end_z=open_start,
                            base_y=base_y, height=height,
                            rotY=rotY,
                            filler_prefab=filler_prefab,
                            include_start_corner=False,
                            include_end_corner=False,
                            anchor_pieces=anchor_pieces if not pieces and not segment_pieces else None
                        )
                        segment_pieces.extend(wall_pieces)
                
                # Place the opening piece - position so bottom snap sits on floor
                open_y = base_y - open_bottom_offset
                if is_x_axis:
                    open_piece = {
                        "prefab": open_prefab,
                        "x": round(open_pos, 3),
                        "y": round(open_y, 3),
                        "z": round(fixed_coord, 3),
                        "rotY": rotY
                    }
                else:
                    open_piece = {
                        "prefab": open_prefab,
                        "x": round(fixed_coord, 3),
                        "y": round(open_y, 3),
                        "z": round(open_pos, 3),
                        "rotY": rotY
                    }
                segment_pieces.append(open_piece)
                
                current_pos = open_end
            
            # Generate wall segment after last opening (if there's space)
            if is_x_axis:
                if current_pos < wall_end - 0.1:
                    wall_pieces = generate_wall(
                        prefab=prefab,
                        start_x=current_pos, start_z=fixed_coord,
                        end_x=wall_end, end_z=fixed_coord,
                        base_y=base_y, height=height,
                        rotY=rotY,
                        filler_prefab=filler_prefab,
                        include_start_corner=False,
                        include_end_corner=False,
                        anchor_pieces=None
                    )
                    segment_pieces.extend(wall_pieces)
            else:
                if current_pos < wall_end - 0.1:
                    wall_pieces = generate_wall(
                        prefab=prefab,
                        start_x=fixed_coord, start_z=current_pos,
                        end_x=fixed_coord, end_z=wall_end,
                        base_y=base_y, height=height,
                        rotY=rotY,
                        filler_prefab=filler_prefab,
                        include_start_corner=False,
                        include_end_corner=False,
                        anchor_pieces=None
                    )
                    segment_pieces.extend(wall_pieces)
        
        return segment_pieces
    
    # Generate all four walls
    # North wall: z=z_max, x goes from x_min to x_max, faces +Z (rotY=0)
    north_pieces = generate_wall_segment(
        start_x=x_min, start_z=z_max,
        end_x=x_max, end_z=z_max,
        rotY=0,
        wall_openings=get_wall_openings("north"),
        is_x_axis=True
    )
    pieces.extend(north_pieces)
    
    # East wall: x=x_max, z goes from z_max to z_min, faces +X (rotY=90)
    east_pieces = generate_wall_segment(
        start_x=x_max, start_z=z_max,
        end_x=x_max, end_z=z_min,
        rotY=90,
        wall_openings=get_wall_openings("east"),
        is_x_axis=False
    )
    pieces.extend(east_pieces)
    
    # South wall: z=z_min, x goes from x_max to x_min, faces -Z (rotY=180)
    south_pieces = generate_wall_segment(
        start_x=x_max, start_z=z_min,
        end_x=x_min, end_z=z_min,
        rotY=180,
        wall_openings=get_wall_openings("south"),
        is_x_axis=True
    )
    pieces.extend(south_pieces)
    
    # West wall: x=x_min, z goes from z_min to z_max, faces -X (rotY=270)
    west_pieces = generate_wall_segment(
        start_x=x_min, start_z=z_min,
        end_x=x_min, end_z=z_max,
        rotY=270,
        wall_openings=get_wall_openings("west"),
        is_x_axis=False
    )
    pieces.extend(west_pieces)
    
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
    
    # Use snap spacing for roof tiling - direction determines which axis
    # north/south extend along Z, east/west extend along X
    if direction in ("north", "south"):
        snap_spacing = _get_snap_spacing(details, "z")
    else:
        snap_spacing = _get_snap_spacing(details, "x")
    
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
        roof_x = start_x + i * snap_spacing * dx
        roof_z = start_z + i * snap_spacing * dz
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

For walls/floors/roofs, prefer composite tools (generate_wall, generate_floor_grid,
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
        "name": "generate_floor_walls",
        "description": """Generate all four walls for a rectangular floor in a single call.

This is the PRIMARY wall generation tool. Creates north, east, south, and west walls
with proper height by stacking rows. Handles door/window openings automatically.

IMPORTANT: Use height=6 or more for typical interior walls.

Example - basic walls:
  generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                       base_y=0.5, height=6, filler_prefab="stone_wall_2x1")

Example - with door opening:
  generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                       base_y=0.5, height=6, filler_prefab="stone_wall_2x1",
                       openings=[{"wall": "south", "position": 0, "prefab": "stone_arch"}])

For a 3-floor tower, call this once per floor (3 total calls vs 12+ with individual walls).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Wall prefab name (e.g., 'stone_wall_4x2', 'woodwall')"
                },
                "x_min": {
                    "type": "number",
                    "description": "Minimum X bound (west edge)"
                },
                "x_max": {
                    "type": "number",
                    "description": "Maximum X bound (east edge)"
                },
                "z_min": {
                    "type": "number",
                    "description": "Minimum Z bound (south edge)"
                },
                "z_max": {
                    "type": "number",
                    "description": "Maximum Z bound (north edge)"
                },
                "base_y": {
                    "type": "number",
                    "description": "Y position of the floor surface the walls sit on"
                },
                "height": {
                    "type": "number",
                    "description": "Target wall height in meters (use 6 or more for typical interior walls)"
                },
                "filler_prefab": {
                    "type": "string",
                    "description": "Optional smaller wall prefab to fill horizontal gaps"
                },
                "openings": {
                    "type": "array",
                    "description": "Optional door/window openings. Each needs: wall ('north'/'east'/'south'/'west'), position (coordinate), prefab",
                    "items": {
                        "type": "object",
                        "properties": {
                            "wall": {
                                "type": "string",
                                "enum": ["north", "east", "south", "west"],
                                "description": "Which wall the opening is on"
                            },
                            "position": {
                                "type": "number",
                                "description": "Position along wall (X for north/south, Z for east/west)"
                            },
                            "prefab": {
                                "type": "string",
                                "description": "Opening prefab (e.g., 'stone_arch', 'wood_door')"
                            },
                            "width": {
                                "type": "number",
                                "description": "Width of opening (defaults to prefab width)"
                            }
                        },
                        "required": ["wall", "position", "prefab"]
                    }
                },
                "anchor_pieces": {
                    "type": "array",
                    "description": "Pieces to snap walls to (e.g., floor pieces)",
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
            "required": ["prefab", "x_min", "x_max", "z_min", "z_max", "base_y", "height"]
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
    elif name == "generate_floor_walls":
        result = generate_floor_walls(
            prefab=args["prefab"],
            x_min=args["x_min"],
            x_max=args["x_max"],
            z_min=args["z_min"],
            z_max=args["z_max"],
            base_y=args["base_y"],
            height=args["height"],
            filler_prefab=args.get("filler_prefab"),
            openings=args.get("openings"),
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
