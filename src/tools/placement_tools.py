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
- generate_roof: Generate a complete gabled roof with both slopes and ridge
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

def _get_anchor_y_offset(prefab: str, rotY: float, anchor: str) -> float:
    """
    Calculate Y offset to position a piece by its anchor point instead of center.
    
    Uses snap points to determine the piece's vertical extent:
    - "bottom": offset so the lowest snap point sits at the specified Y
    - "top": offset so the highest snap point sits at the specified Y
    - "center": no offset (default behavior)
    
    Returns the offset to add to Y position.
    """
    if anchor == "center":
        return 0.0
    
    details = get_prefab_details(prefab)
    if not details or not details.get("snapPoints"):
        return 0.0
    
    # Get snap points in local space and find Y extents
    snap_ys = []
    for sp in details["snapPoints"]:
        # Rotate the snap point (Y doesn't change with Y-axis rotation)
        snap_ys.append(sp["y"])
    
    if not snap_ys:
        return 0.0
    
    if anchor == "bottom":
        # To place bottom snap at Y, we need to raise the center
        # If lowest snap is at -1.0 relative to center, we add 1.0 to Y
        lowest_snap_y = min(snap_ys)
        return -lowest_snap_y
    elif anchor == "top":
        # To place top snap at Y, we need to lower the center
        # If highest snap is at +1.0 relative to center, we subtract 1.0 from Y
        highest_snap_y = max(snap_ys)
        return -highest_snap_y
    
    return 0.0


def place_piece(
    prefab: str,
    x: float,
    y: float,
    z: float,
    rotY: Literal[0, 90, 180, 270],
    placed_pieces: list[dict] | None = None,
    snap: bool = False,
    anchor: Literal["bottom", "center", "top"] = "center"
) -> dict:
    """
    Place a single piece at (x, y, z) with rotation rotY.
    
    Use this for individual pieces that don't fit composite tools:
    - Doors, arches, stairs
    - Decorations and furniture
    - One-off pieces needing precise placement
    
    For walls, floors, and roofs, prefer the composite tools (generate_wall,
    generate_floor_grid, generate_roof) which handle snapping internally.
    
    Args:
        prefab: Exact prefab name (e.g., "stone_floor_2x2")
        x, y, z: World position (piece center, or anchor point if anchor specified)
        rotY: Y-axis rotation in degrees (0, 90, 180, or 270)
        placed_pieces: List of pieces to snap to (only used if snap=True)
        snap: Whether to apply snap correction (default False - use for doors/decorations)
        anchor: Vertical anchor point - "bottom", "center" (default), or "top".
                When "bottom", Y specifies where the piece's lowest snap point should be.
                When "top", Y specifies where the piece's highest snap point should be.
    
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
    
    # Apply anchor offset to Y position
    anchor_offset = _get_anchor_y_offset(prefab, rotY, anchor)
    adjusted_y = y + anchor_offset
    
    final_x, final_y, final_z = x, adjusted_y, z
    snapped = False
    snap_distance = 0.0
    
    if snap and placed_pieces:
        final_x, final_y, final_z, snapped, snap_distance = _find_snap_correction(
            prefab, x, adjusted_y, z, rotY, placed_pieces
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


def replace_piece(
    prefab: str,
    x: float,
    y: float,
    z: float,
    rotY: Literal[0, 90, 180, 270],
    placed_pieces: list[dict]
) -> dict:
    """
    Replace the closest piece at a position with a new piece.
    
    Finds the single closest piece to (x, z) in placed_pieces (within 2m of y),
    removes it, and places the new piece at the removed piece's position.
    
    The new piece's Y position is derived from the removed piece's floor level
    (its bottom snap point), ensuring doors sit correctly on the floor.
    
    Args:
        prefab: New piece prefab name (e.g., "wood_door")
        x, y, z: Approximate position to find the piece to replace
                 (y is used to filter to bottom wall row only)
        rotY: Y-axis rotation (0, 90, 180, or 270)
        placed_pieces: List of existing pieces (will be modified in-place)
    
    Returns:
        Dict with keys: removed (the removed piece), placed (the new piece)
    """
    if not placed_pieces:
        return {"error": "No pieces to replace"}
    
    # Find the closest piece by 2D distance (x, z)
    # Also filter to pieces at similar Y level (within 2m) to avoid replacing
    # upper wall rows when placing a ground-level door
    best_idx = None
    best_dist = float("inf")
    
    for i, piece in enumerate(placed_pieces):
        # Skip pieces that are too far vertically (e.g., upper wall rows)
        dy = abs(piece["y"] - y)
        if dy > 2.0:
            continue
            
        dx = piece["x"] - x
        dz = piece["z"] - z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    if best_idx is None:
        return {"error": "No piece found to replace at similar Y level"}
    
    # Remove the closest piece
    removed = placed_pieces.pop(best_idx)
    
    # Calculate the removed piece's base Y (floor level) from its snap points
    # The removed wall's center is at removed["y"], but we need the floor level
    # which is where the wall's bottom snap point sits
    removed_details = get_prefab_details(removed["prefab"])
    if removed_details:
        removed_bottom_offset = _get_bottom_snap_offset(removed_details)
        # Floor level = piece center Y + bottom snap offset (which is negative)
        floor_y = removed["y"] + removed_bottom_offset
    else:
        # Fallback: assume standard 2m wall with center at y, bottom at y-1
        floor_y = removed["y"] - 1.0
    
    # Place the new piece AT THE REMOVED PIECE'S X/Z POSITION
    # Use floor_y as the base, with anchor="bottom" to position door correctly
    new_piece = place_piece(
        prefab=prefab,
        x=removed["x"],  # Use removed piece's X
        y=floor_y,       # Use calculated floor level
        z=removed["z"],  # Use removed piece's Z
        rotY=rotY,
        placed_pieces=None,
        snap=False,
        anchor="bottom"  # Always anchor to bottom since we calculated floor_y
    )
    
    return {
        "removed": removed,
        "placed": new_piece
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


def _get_roof_snap_offsets(prefab_details: dict) -> dict:
    """
    Get snap point offsets for a roof piece.
    
    Roof pieces have snap points at their edges:
    - "low_z": Z offset to the low/eave edge (positive Z, bottom of slope)
    - "high_z": Z offset to the high/ridge edge (negative Z, top of slope)
    - "low_y": Y offset at the low edge
    - "high_y": Y offset at the high edge
    - "left_x": X offset to left edge
    - "right_x": X offset to right edge
    
    For wood_roof: low edge at z=+1, y=0; high edge at z=-1, y=1
    """
    snap_points = prefab_details.get("snapPoints")
    if not snap_points:
        # Fallback to bounding box estimates
        w = prefab_details.get("width", 2.0) / 2
        d = prefab_details.get("depth", 2.0) / 2
        return {
            "low_z": d, "high_z": -d,
            "low_y": 0.0, "high_y": 1.0,
            "left_x": -w, "right_x": w
        }
    
    z_values = [sp["z"] for sp in snap_points]
    y_values = [sp["y"] for sp in snap_points]
    x_values = [sp["x"] for sp in snap_points]
    
    # Find snap points at low edge (max Z) and high edge (min Z)
    low_z = max(z_values)
    high_z = min(z_values)
    
    # Get Y values at those Z positions
    low_y = min(sp["y"] for sp in snap_points if abs(sp["z"] - low_z) < 0.1)
    high_y = max(sp["y"] for sp in snap_points if abs(sp["z"] - high_z) < 0.1)
    
    return {
        "low_z": low_z,      # Z offset to eave/bottom edge (e.g., +1.0)
        "high_z": high_z,    # Z offset to ridge/top edge (e.g., -1.0)
        "low_y": low_y,      # Y at eave edge (e.g., 0.0)
        "high_y": high_y,    # Y at ridge edge (e.g., 1.0)
        "left_x": min(x_values),   # X offset to left edge
        "right_x": max(x_values)   # X offset to right edge
    }


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
    
    # Get filler details if provided.
    filler_details = None
    filler_snap_w = 0.0
    filler_snap_h = snap_h
    filler_bottom_offset = 0.0
    if filler_prefab:
        filler_details = get_prefab_details(filler_prefab)
        if filler_details:
            filler_snap_w = _get_snap_spacing(filler_details, "x")
            filler_snap_h = _get_snap_height(filler_details)
            filler_bottom_offset = _get_bottom_snap_offset(filler_details)
    
    # Calculate how many filler rows needed per main row (for height matching)
    filler_rows_per_main = max(1, int(round(snap_h / filler_snap_h))) if filler_snap_h > 0.1 else 1
    
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
        # Fillers may have different snap height, so we stack them independently.
        remaining = length - covered
        if filler_details and remaining > 0.1 and filler_snap_w > 0.1:
            # For each filler sub-row within this main row's height
            for filler_row in range(filler_rows_per_main):
                # Calculate Y position using filler's own snap points
                filler_row_y = base_y - filler_bottom_offset + row * snap_h + filler_row * filler_snap_h
                filler_covered = covered
                filler_remaining = remaining
                last_filler_in_row = last_piece_in_row
                
                # Place fillers horizontally
                while filler_remaining >= filler_snap_w - 0.05:
                    center_offset = filler_covered + filler_snap_w / 2
                    filler_x = start_x + dir_x * center_offset
                    filler_z = start_z + dir_z * center_offset
                    filler_y = filler_row_y
                    
                    if last_filler_in_row:
                        filler_x, filler_y, filler_z, _ = _snap_to_piece(
                            filler_prefab, filler_x, filler_y, filler_z, rotY, last_filler_in_row
                        )
                    
                    filler_piece = {
                        "prefab": filler_prefab,
                        "x": round(filler_x, 3),
                        "y": round(filler_y, 3),
                        "z": round(filler_z, 3),
                        "rotY": rotY
                    }
                    pieces.append(filler_piece)
                    last_filler_in_row = filler_piece
                    filler_covered += filler_snap_w
                    filler_remaining = length - filler_covered
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
            # Determine wall direction and normalize to always iterate min->max
            if is_x_axis:
                wall_start = min(start_x, end_x)
                wall_end = max(start_x, end_x)
                fixed_coord = start_z  # Z is fixed for north/south walls
            else:
                wall_start = min(start_z, end_z)
                wall_end = max(start_z, end_z)
                fixed_coord = start_x  # X is fixed for east/west walls
            
            # Sort openings by position (ascending)
            sorted_openings = sorted(wall_openings, key=lambda o: o.get("position", 0))
            
            current_pos = wall_start
            
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
                if current_pos < open_start - 0.1:
                    if is_x_axis:
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
                    else:
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
                
                # Generate wall pieces ABOVE the opening if the opening is shorter than wall height
                open_height = open_details["height"] if open_details else 2.0
                if open_height < height - 0.1:
                    # Calculate height remaining above the opening
                    above_base_y = base_y + open_height
                    above_height = height - open_height
                    
                    # Generate wall segment above the opening
                    if is_x_axis:
                        above_pieces = generate_wall(
                            prefab=prefab,
                            start_x=open_start, start_z=fixed_coord,
                            end_x=open_end, end_z=fixed_coord,
                            base_y=above_base_y, height=above_height,
                            rotY=rotY,
                            filler_prefab=filler_prefab,
                            include_start_corner=False,
                            include_end_corner=False,
                            anchor_pieces=None
                        )
                    else:
                        above_pieces = generate_wall(
                            prefab=prefab,
                            start_x=fixed_coord, start_z=open_start,
                            end_x=fixed_coord, end_z=open_end,
                            base_y=above_base_y, height=above_height,
                            rotY=rotY,
                            filler_prefab=filler_prefab,
                            include_start_corner=False,
                            include_end_corner=False,
                            anchor_pieces=None
                        )
                    segment_pieces.extend(above_pieces)
                
                current_pos = open_end
            
            # Generate wall segment after last opening (if there's space)
            if current_pos < wall_end - 0.1:
                if is_x_axis:
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
                else:
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


def generate_roof(
    prefab: str,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    base_y: float,
    ridge_axis: Literal["x", "z"] = "x",
    ridge_prefab: str | None = None
) -> list[dict]:
    """
    Generate a complete gabled roof covering a rectangular building footprint.
    
    Creates both slopes of a gabled roof in a single call. The roof slopes down
    from the ridge toward the edges of the building. Ridge caps are auto-placed
    when needed based on geometry (when slopes don't fully meet at the center).
    
    Args:
        prefab: Roof slope prefab name (e.g., "wood_roof", "wood_roof_45", "darkwood_roof")
        x_min, x_max: X bounds of the building footprint
        z_min, z_max: Z bounds of the building footprint
        base_y: Y position of wall tops (where roof starts)
        ridge_axis: Which axis the ridge runs along:
            - "x": Ridge runs along X axis (roof slopes down toward z_min and z_max)
            - "z": Ridge runs along Z axis (roof slopes down toward x_min and x_max)
        ridge_prefab: Optional ridge cap prefab (e.g., "wood_roof_top"). When provided,
            ridge caps are placed along the ridge line if geometry requires them.
    
    Returns:
        List of piece dicts for the complete roof (slopes + ridge caps).
    
    Example:
        generate_roof(
            prefab="wood_roof",
            x_min=-8, x_max=8,
            z_min=-6, z_max=6,
            base_y=12,
            ridge_axis="x",
            ridge_prefab="wood_roof_top"
        )
    """
    details = get_prefab_details(prefab)
    if not details:
        return [{"error": f"Unknown prefab: {prefab}"}]
    
    ridge_details = None
    if ridge_prefab:
        ridge_details = get_prefab_details(ridge_prefab)
        if not ridge_details:
            return [{"error": f"Unknown ridge prefab: {ridge_prefab}"}]
    
    pieces = []
    
    # Get snap point offsets for the roof piece
    roof_snaps = _get_roof_snap_offsets(details)
    
    # Snap spacing along the row (X axis for ridge_axis="x")
    row_spacing = roof_snaps["right_x"] - roof_snaps["left_x"]  # e.g., 2.0 for wood_roof
    
    # Snap spacing perpendicular to row (Z depth from low to high edge)
    depth_spacing = roof_snaps["low_z"] - roof_snaps["high_z"]  # e.g., 2.0 for wood_roof
    
    # Y rise per row (height gain as we go up toward ridge)
    y_rise = roof_snaps["high_y"] - roof_snaps["low_y"]  # e.g., 1.0 for wood_roof
    
    # Offset from piece center to its low edge (eave) - used for initial placement
    low_edge_offset = roof_snaps["low_z"]   # e.g., +1.0 for wood_roof
    high_edge_offset = roof_snaps["high_z"]  # e.g., -1.0 for wood_roof
    low_y_offset = roof_snaps["low_y"]       # e.g., 0.0 for wood_roof
    left_x_offset = roof_snaps["left_x"]     # e.g., -1.0 for wood_roof
    
    if ridge_axis == "x":
        # Ridge runs along X axis, slopes face north and south
        # Rows extend along X (east-west), we stack rows from z_min toward center and z_max toward center
        
        building_width = x_max - x_min
        building_depth = z_max - z_min
        
        # Calculate number of pieces per row (along X)
        pieces_per_row = max(1, int(round(building_width / row_spacing)))
        
        # Calculate number of rows from edge to ridge (half the depth)
        half_depth = building_depth / 2
        
        # Ridge center Z position
        ridge_z = (z_min + z_max) / 2
        
        # Calculate how many slope rows fit from edge to center
        rows_per_slope = max(1, int(half_depth / depth_spacing))
        
        # === South slope (from z_min toward ridge, rotY=180) ===
        # With rotY=180, the piece is flipped: low_z (normally +Z) now points to -Z (south)
        # So piece center should be placed such that its rotated low edge aligns with z_min
        # After 180° rotation: low edge at piece_z - low_edge_offset = z_min
        # Therefore: piece_z = z_min + low_edge_offset
        for row in range(rows_per_slope):
            # First row: low edge at z_min, so center at z_min + low_edge_offset
            # Each subsequent row steps by depth_spacing toward ridge
            piece_z = z_min + low_edge_offset + row * depth_spacing
            piece_y = base_y - low_y_offset + row * y_rise
            
            for col in range(pieces_per_row):
                # First piece: left edge at x_min, so center at x_min - left_x_offset
                piece_x = x_min - left_x_offset + col * row_spacing
                
                roof_piece = {
                    "prefab": prefab,
                    "x": round(piece_x, 3),
                    "y": round(piece_y, 3),
                    "z": round(piece_z, 3),
                    "rotY": 180  # slope descends toward -Z (south/away from ridge)
                }
                pieces.append(roof_piece)
        
        # === Ridge caps (auto-determined based on geometry) ===
        # Check if slopes meet at center: last south slope's high edge reaches ridge_z
        # South slope last piece center at: z_min + low_edge_offset + (rows-1)*depth_spacing
        # Its high edge (after 180° rotation) is at: piece_z + low_edge_offset (since high_z becomes +Z after rotation)
        south_last_center = z_min + low_edge_offset + (rows_per_slope - 1) * depth_spacing
        south_high_edge = south_last_center + low_edge_offset  # After 180° rotation, high edge is at +Z
        slopes_meet = south_high_edge >= ridge_z - 0.1
        
        if ridge_prefab and ridge_details and not slopes_meet:
            ridge_y = base_y - low_y_offset + rows_per_slope * y_rise
            for col in range(pieces_per_row):
                piece_x = x_min - left_x_offset + col * row_spacing
                
                ridge_piece = {
                    "prefab": ridge_prefab,
                    "x": round(piece_x, 3),
                    "y": round(ridge_y, 3),
                    "z": round(ridge_z, 3),
                    "rotY": 0
                }
                pieces.append(ridge_piece)
        
        # === North slope (from z_max toward ridge, rotY=0) ===
        # With rotY=0, low edge (z=+1 in local) points to +Z (north)
        # So piece center should be placed such that low edge aligns with z_max
        # low edge at piece_z + low_edge_offset = z_max
        # Therefore: piece_z = z_max - low_edge_offset
        for row in range(rows_per_slope):
            piece_z = z_max - low_edge_offset - row * depth_spacing
            piece_y = base_y - low_y_offset + row * y_rise
            
            for col in range(pieces_per_row):
                piece_x = x_min - left_x_offset + col * row_spacing
                
                roof_piece = {
                    "prefab": prefab,
                    "x": round(piece_x, 3),
                    "y": round(piece_y, 3),
                    "z": round(piece_z, 3),
                    "rotY": 0  # slope descends toward +Z (north/away from ridge)
                }
                pieces.append(roof_piece)
    
    else:  # ridge_axis == "z"
        # Ridge runs along Z axis, slopes face east and west
        # For this orientation, pieces are rotated 90° or 270°
        # The Z-axis snap points become X-axis after rotation
        
        building_width = x_max - x_min
        building_depth = z_max - z_min
        
        # When rotated 90°, the piece's Z becomes world X, and piece's X becomes world Z
        # So row_spacing (along Z world) uses the original Z snap spacing
        # And depth_spacing (toward ridge, along X world) uses original Z snap spacing too
        
        # Calculate number of pieces per row (along Z)
        pieces_per_row = max(1, int(round(building_depth / row_spacing)))
        
        # Calculate number of rows from edge to ridge (half the width)
        half_width = building_width / 2
        
        # Ridge center X position
        ridge_x = (x_min + x_max) / 2
        
        # Calculate how many slope rows fit from edge to center
        rows_per_slope = max(1, int(half_width / depth_spacing))
        
        # === West slope (from x_min toward ridge, rotY=270) ===
        # With rotY=270, the piece's +Z (low edge) points to -X (west)
        # So piece center at: x_min + low_edge_offset (so low edge is at x_min)
        for row in range(rows_per_slope):
            piece_x = x_min + low_edge_offset + row * depth_spacing
            piece_y = base_y - low_y_offset + row * y_rise
            
            for col in range(pieces_per_row):
                # Piece's X axis (after 270° rotation) aligns with world Z
                piece_z = z_min - left_x_offset + col * row_spacing
                
                roof_piece = {
                    "prefab": prefab,
                    "x": round(piece_x, 3),
                    "y": round(piece_y, 3),
                    "z": round(piece_z, 3),
                    "rotY": 270  # slope descends toward -X (west/away from ridge)
                }
                pieces.append(roof_piece)
        
        # === Ridge caps (auto-determined based on geometry) ===
        west_last_center = x_min + low_edge_offset + (rows_per_slope - 1) * depth_spacing
        west_high_edge = west_last_center + low_edge_offset  # After 270° rotation
        slopes_meet = west_high_edge >= ridge_x - 0.1
        
        if ridge_prefab and ridge_details and not slopes_meet:
            ridge_y = base_y - low_y_offset + rows_per_slope * y_rise
            for col in range(pieces_per_row):
                piece_z = z_min - left_x_offset + col * row_spacing
                
                ridge_piece = {
                    "prefab": ridge_prefab,
                    "x": round(ridge_x, 3),
                    "y": round(ridge_y, 3),
                    "z": round(piece_z, 3),
                    "rotY": 90  # ridge runs along Z axis
                }
                pieces.append(ridge_piece)
        
        # === East slope (from x_max toward ridge, rotY=90) ===
        # With rotY=90, the piece's +Z (low edge) points to +X (east)
        # So piece center at: x_max - low_edge_offset (so low edge is at x_max)
        for row in range(rows_per_slope):
            piece_x = x_max - low_edge_offset - row * depth_spacing
            piece_y = base_y - low_y_offset + row * y_rise
            
            for col in range(pieces_per_row):
                piece_z = z_min - left_x_offset + col * row_spacing
                
                roof_piece = {
                    "prefab": prefab,
                    "x": round(piece_x, 3),
                    "y": round(piece_y, 3),
                    "z": round(piece_z, 3),
                    "rotY": 90  # slope descends toward +X (east/away from ridge)
                }
                pieces.append(roof_piece)
    
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
generate_roof) which handle snapping internally and are more efficient.

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
                },
                "anchor": {
                    "type": "string",
                    "enum": ["bottom", "center", "top"],
                    "description": "Vertical anchor point. 'bottom': Y is where lowest snap point sits. 'top': Y is where highest snap point sits. 'center' (default): Y is piece center."
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
        "name": "generate_wall",
        "description": """Generate a single wall segment along a line.

Use this for multi-volume buildings where you need individual wall control:
- When volumes connect and you need to skip shared walls
- L-shaped or non-rectangular footprints
- Custom wall arrangements

For simple rectangular buildings, prefer generate_floor_walls which creates all 4 walls at once.

The wall runs from (start_x, start_z) to (end_x, end_z) and stacks vertically to reach height.

Example - north wall only:
  generate_wall(prefab="stone_wall_4x2", start_x=-6, start_z=4, end_x=6, end_z=4,
                base_y=0, height=6, rotY=0, filler_prefab="stone_wall_2x1")

rotY determines which way the wall faces:
- 0: faces +Z (north) - use for north walls
- 90: faces +X (east) - use for east walls
- 180: faces -Z (south) - use for south walls
- 270: faces -X (west) - use for west walls""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Wall prefab name (e.g., 'stone_wall_4x2')"
                },
                "start_x": {
                    "type": "number",
                    "description": "X coordinate of wall start point"
                },
                "start_z": {
                    "type": "number",
                    "description": "Z coordinate of wall start point"
                },
                "end_x": {
                    "type": "number",
                    "description": "X coordinate of wall end point"
                },
                "end_z": {
                    "type": "number",
                    "description": "Z coordinate of wall end point"
                },
                "base_y": {
                    "type": "number",
                    "description": "Y position of the floor surface the wall sits on"
                },
                "height": {
                    "type": "number",
                    "description": "Target wall height in meters (use 6 or more)"
                },
                "rotY": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                    "description": "Wall facing direction: 0=north, 90=east, 180=south, 270=west"
                },
                "filler_prefab": {
                    "type": "string",
                    "description": "Optional smaller wall prefab to fill horizontal gaps"
                },
                "corner_prefab": {
                    "type": "string",
                    "description": "Optional pole/pillar prefab for corners"
                },
                "include_start_corner": {
                    "type": "boolean",
                    "description": "Place corner post at start point (default true)"
                },
                "include_end_corner": {
                    "type": "boolean",
                    "description": "Place corner post at end point (default true)"
                }
            },
            "required": ["prefab", "start_x", "start_z", "end_x", "end_z", "base_y", "height", "rotY"]
        }
    },
    {
        "name": "generate_roof",
        "description": """Generate a complete gabled roof for a rectangular building in a single call.

Creates both slopes of a gabled roof. Optionally places ridge caps along the ridge line
and corner caps at specified positions.

This is the PRIMARY roof generation tool. Use this instead of placing individual
roof pieces manually.

Example - basic roof:
  generate_roof(prefab="wood_roof", x_min=-8, x_max=8, z_min=-6, z_max=6,
                base_y=12, ridge_axis="x")

Example - roof with ridge caps:
  generate_roof(prefab="wood_roof", x_min=-8, x_max=8, z_min=-6, z_max=6,
                base_y=12, ridge_axis="x", ridge_prefab="wood_roof_top")

The ridge_axis determines roof orientation:
- "x": Ridge runs east-west, slopes face north and south
- "z": Ridge runs north-south, slopes face east and west

Ridge caps are auto-placed when geometry requires them (when slopes don't fully meet at center).""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "Roof slope prefab name (e.g., 'wood_roof', 'wood_roof_45', 'darkwood_roof')"
                },
                "x_min": {
                    "type": "number",
                    "description": "Minimum X bound of building footprint (west edge)"
                },
                "x_max": {
                    "type": "number",
                    "description": "Maximum X bound of building footprint (east edge)"
                },
                "z_min": {
                    "type": "number",
                    "description": "Minimum Z bound of building footprint (south edge)"
                },
                "z_max": {
                    "type": "number",
                    "description": "Maximum Z bound of building footprint (north edge)"
                },
                "base_y": {
                    "type": "number",
                    "description": "Y position of wall tops (where roof starts)"
                },
                "ridge_axis": {
                    "type": "string",
                    "enum": ["x", "z"],
                    "description": "Which axis the ridge runs along: 'x' (slopes face N/S) or 'z' (slopes face E/W)"
                },
                "ridge_prefab": {
                    "type": "string",
                    "description": "Ridge cap prefab (e.g., 'wood_roof_top'). When provided, ridge caps are auto-placed where needed."
                }
            },
            "required": ["prefab", "x_min", "x_max", "z_min", "z_max", "base_y", "ridge_axis"]
        }
    },
    {
        "name": "complete_build",
        "description": """Signal that the build is complete. Call this after all pieces have been placed.

This finalizes the blueprint and returns the accumulated pieces. Do NOT output JSON manually - 
just call this tool when you're done placing all floors, walls, roofs, and decorations.""",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "remove_piece",
        "description": """Remove a piece from the build by its index in the pieces list.

Use this to create gaps (e.g., remove wall segments before adding windows) or fix mistakes.
The pieces list is 0-indexed. After removal, indices of subsequent pieces shift down by 1.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Index of the piece to remove (0-based)"
                }
            },
            "required": ["index"]
        }
    },
    {
        "name": "replace_piece",
        "description": """Replace the closest piece at a position with a new piece.

Finds the closest piece to (x, z) in the build (within 2m of y), removes it, and places the
new piece at the removed piece's exact position. The door's Y is automatically derived from
the removed wall's floor level (bottom snap point).

Use this for doors/windows to swap out wall pieces without leaving overlapping geometry.

Example - replace wall with door on south wall at z=2:
  replace_piece(prefab="wood_door", x=6, y=1.5, z=2, rotY=180)
  # Finds closest wall to (6, 2), removes it, places door at wall's position on floor""",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefab": {
                    "type": "string",
                    "description": "New piece prefab name (e.g., 'wood_door')"
                },
                "x": {
                    "type": "number",
                    "description": "Approximate X position to find piece to replace"
                },
                "y": {
                    "type": "number",
                    "description": "Approximate Y (filters to bottom wall row, use ~1.5 for ground floor)"
                },
                "z": {
                    "type": "number",
                    "description": "Approximate Z position to find piece to replace"
                },
                "rotY": {
                    "type": "integer",
                    "enum": [0, 90, 180, 270],
                    "description": "Y-axis rotation in degrees"
                }
            },
            "required": ["prefab", "x", "y", "z", "rotY"]
        }
    }
]


def execute_placement_tool(name: str, args: dict, accumulator: list[dict] | None = None) -> str:
    """
    Execute a placement tool by name and return JSON result.
    
    Called by the build agent when Claude uses a placement tool.
    
    Args:
        name: Tool name to execute
        args: Tool arguments
        accumulator: Optional list to accumulate pieces into. If provided,
                     pieces are appended and a summary is returned instead
                     of the full piece array.
    
    Returns:
        JSON string with either full pieces (no accumulator) or summary (with accumulator).
        For complete_build, returns a special marker.
    """
    if name == "complete_build":
        # Signal completion - the agent loop handles this specially
        return json.dumps({"complete": True, "total_pieces": len(accumulator) if accumulator else 0})
    
    if name == "remove_piece":
        index = args.get("index")
        if accumulator is None:
            return json.dumps({"error": "No accumulator available for remove_piece"})
        if index is None or not isinstance(index, int):
            return json.dumps({"error": "index must be an integer"})
        if index < 0 or index >= len(accumulator):
            return json.dumps({"error": f"Index {index} out of range (0-{len(accumulator)-1})"})
        
        removed = accumulator.pop(index)
        return json.dumps({
            "removed": removed,
            "total_pieces": len(accumulator)
        })
    
    if name == "replace_piece":
        if accumulator is None:
            return json.dumps({"error": "No accumulator available for replace_piece"})
        
        result = replace_piece(
            prefab=args["prefab"],
            x=args["x"],
            y=args["y"],
            z=args["z"],
            rotY=args["rotY"],
            placed_pieces=accumulator
        )
        
        if result.get("error"):
            return json.dumps(result)
        
        # Add the new piece to the accumulator
        accumulator.append(result["placed"])
        
        return json.dumps({
            "removed": result["removed"],
            "placed": result["placed"],
            "total_pieces": len(accumulator)
        })
    
    if name == "place_piece":
        result = place_piece(
            prefab=args["prefab"],
            x=args["x"],
            y=args["y"],
            z=args["z"],
            rotY=args["rotY"],
            placed_pieces=args.get("placed_pieces"),
            snap=args.get("snap", False),
            anchor=args.get("anchor", "center")
        )
        # place_piece returns a single dict, not a list
        pieces = [result] if not result.get("error") else []
    elif name == "generate_floor_grid":
        pieces = generate_floor_grid(
            prefab=args["prefab"],
            width=args["width"],
            depth=args["depth"],
            y=args["y"],
            origin_x=args.get("origin_x", 0.0),
            origin_z=args.get("origin_z", 0.0)
        )
        result = pieces
    elif name == "generate_floor_walls":
        pieces = generate_floor_walls(
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
        result = pieces
    elif name == "generate_wall":
        pieces = generate_wall(
            prefab=args["prefab"],
            start_x=args["start_x"],
            start_z=args["start_z"],
            end_x=args["end_x"],
            end_z=args["end_z"],
            base_y=args["base_y"],
            height=args["height"],
            rotY=args["rotY"],
            filler_prefab=args.get("filler_prefab"),
            corner_prefab=args.get("corner_prefab"),
            corner_y=args.get("corner_y"),
            include_start_corner=args.get("include_start_corner", True),
            include_end_corner=args.get("include_end_corner", True),
            anchor_pieces=args.get("anchor_pieces")
        )
        result = pieces
    elif name == "generate_roof":
        pieces = generate_roof(
            prefab=args["prefab"],
            x_min=args["x_min"],
            x_max=args["x_max"],
            z_min=args["z_min"],
            z_max=args["z_max"],
            base_y=args["base_y"],
            ridge_axis=args["ridge_axis"],
            ridge_prefab=args.get("ridge_prefab")
        )
        result = pieces
    else:
        return json.dumps({"error": f"Unknown placement tool: {name}"})
    
    # Check for errors in result
    if isinstance(result, list) and result and result[0].get("error"):
        return json.dumps(result[0])
    if isinstance(result, dict) and result.get("error"):
        return json.dumps(result)
    
    # If accumulator provided, append pieces and return summary
    if accumulator is not None:
        if isinstance(pieces, list):
            accumulator.extend(pieces)
            added = len(pieces)
        else:
            # Single piece from place_piece
            accumulator.append(pieces)
            added = 1
        
        return json.dumps({
            "added": added,
            "total_pieces": len(accumulator)
        })
    
    # No accumulator - return full result (backwards compatibility)
    return json.dumps(result, indent=2)
