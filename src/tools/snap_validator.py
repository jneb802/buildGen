"""
Snap point validation and correction for blueprints.

This is Stage 3 of the pipeline. It runs locally without an LLM to correct
piece positions so they align at snap points. Valheim's building system
requires pieces to connect at specific points.
"""

import math
from dataclasses import dataclass

from src.models import Piece, Vector3
from src.tools.prefab_lookup import get_prefab_details


# Maximum allowed distance between snap points before correction.
SNAP_TOLERANCE = 0.1


# ============================================================================
# Vector Math Helpers
# ============================================================================

@dataclass
class Vec3:
    """Simple vector for internal calculations."""
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


def rotate_y(point: Vec3, degrees: float) -> Vec3:
    """
    Rotate a point around the Y axis by the given degrees.
    
    Used to transform local snap points to world space based on piece rotation.
    """
    rad = math.radians(degrees)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return Vec3(
        point.x * cos_r - point.z * sin_r,
        point.y,
        point.x * sin_r + point.z * cos_r
    )


def get_world_snap_points(piece: Piece) -> list[Vec3]:
    """
    Get the world-space positions of all snap points for a placed piece.
    
    Transforms local snap points by adding piece position and applying rotation.
    """
    details = get_prefab_details(piece.prefab)
    if not details or not details.get("snapPoints"):
        return []
    
    piece_pos = Vec3(piece.x, piece.y, piece.z)
    world_points = []
    
    for sp in details["snapPoints"]:
        local = Vec3(sp["x"], sp["y"], sp["z"])
        rotated = rotate_y(local, piece.rotY)
        world = piece_pos + rotated
        world_points.append(world)
    
    return world_points


# ============================================================================
# Validation Algorithm
# ============================================================================

@dataclass
class CorrectionResult:
    """Result of correcting a piece's position."""
    original: Piece
    corrected: Piece
    was_corrected: bool
    correction_distance: float


def find_closest_snap_pair(
    new_piece: Piece,
    placed_pieces: list[Piece]
) -> tuple[Vec3 | None, Vec3 | None, float]:
    """
    Find the closest pair of snap points between a new piece and all placed pieces.
    
    Returns (new_piece_snap, placed_piece_snap, distance) or (None, None, inf) if
    no snap points exist.
    """
    new_snaps = get_world_snap_points(new_piece)
    if not new_snaps:
        return None, None, float("inf")
    
    best_new = None
    best_placed = None
    best_dist = float("inf")
    
    for placed in placed_pieces:
        placed_snaps = get_world_snap_points(placed)
        for ns in new_snaps:
            for ps in placed_snaps:
                dist = ns.distance(ps)
                if dist < best_dist:
                    best_dist = dist
                    best_new = ns
                    best_placed = ps
    
    return best_new, best_placed, best_dist


def correct_piece_position(piece: Piece, offset: Vec3) -> Piece:
    """Create a new piece with position adjusted by the given offset."""
    return Piece(
        prefab=piece.prefab,
        x=piece.x + offset.x,
        y=piece.y + offset.y,
        z=piece.z + offset.z,
        rotY=piece.rotY
    )


def validate_and_correct(pieces: list[Piece]) -> tuple[list[Piece], list[CorrectionResult]]:
    """
    Validate and correct snap point alignment for all pieces.
    
    Algorithm:
    1. First piece is the anchor - placed as-is
    2. For each subsequent piece:
       - Find the closest snap point pair to any existing piece
       - If distance > tolerance, shift the new piece to align snap points
    3. Return corrected pieces and a report of changes
    
    This ensures all pieces connect properly at their snap points.
    """
    if not pieces:
        return [], []
    
    corrected_pieces = [pieces[0]]  # First piece is anchor.
    corrections = [CorrectionResult(pieces[0], pieces[0], False, 0.0)]
    
    for piece in pieces[1:]:
        new_snap, placed_snap, dist = find_closest_snap_pair(piece, corrected_pieces)
        
        if dist <= SNAP_TOLERANCE or new_snap is None or placed_snap is None:
            # Already aligned or no snap points - keep as-is.
            corrected_pieces.append(piece)
            corrections.append(CorrectionResult(piece, piece, False, 0.0))
        else:
            # Calculate offset needed to align snap points.
            # Move the new piece so its snap point lands on the placed snap point.
            offset = placed_snap - new_snap
            corrected = correct_piece_position(piece, offset)
            corrected_pieces.append(corrected)
            corrections.append(CorrectionResult(piece, corrected, True, dist))
    
    return corrected_pieces, corrections


def format_correction_report(corrections: list[CorrectionResult]) -> str:
    """Generate a human-readable report of what was corrected."""
    lines = ["# Snap Point Validation Report\n"]
    
    corrected_count = sum(1 for c in corrections if c.was_corrected)
    lines.append(f"Total pieces: {len(corrections)}")
    lines.append(f"Pieces corrected: {corrected_count}\n")
    
    for i, c in enumerate(corrections):
        if c.was_corrected:
            lines.append(f"Piece {i} ({c.original.prefab}):")
            lines.append(f"  Original: ({c.original.x:.3f}, {c.original.y:.3f}, {c.original.z:.3f})")
            lines.append(f"  Corrected: ({c.corrected.x:.3f}, {c.corrected.y:.3f}, {c.corrected.z:.3f})")
            lines.append(f"  Distance: {c.correction_distance:.3f}m\n")
    
    return "\n".join(lines)
