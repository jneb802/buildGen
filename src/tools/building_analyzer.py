"""
Building analyzer for extracting semantic structure from piece lists.

Analyzes a list of placed pieces to extract high-level building information
like bounds, floor levels, corners, and material. Used by the detail agent
to understand building structure for enhancement descriptions.
"""

from collections import Counter

from src.tools.prefab_lookup import get_prefab_details


def _detect_material(pieces: list[dict]) -> str:
    """Detect the primary material from piece prefab names."""
    material_patterns = {
        "darkwood": ["darkwood_"],
        "blackmarble": ["blackmarble_"],
        "stone": ["stone_"],
        "wood": ["wood_", "woodwall"],
    }
    
    counts: Counter[str] = Counter()
    
    for piece in pieces:
        prefab = piece.get("prefab", "")
        if not isinstance(prefab, str):
            continue
        
        for material, patterns in material_patterns.items():
            if any(pat in prefab for pat in patterns):
                counts[material] += 1
                break
    
    if not counts:
        return "wood"
    
    return counts.most_common(1)[0][0]


def _find_floor_pieces(pieces: list[dict]) -> list[dict]:
    """Find all floor pieces in the build."""
    floor_keywords = ["floor", "Floor"]
    return [
        p for p in pieces
        if any(kw in p.get("prefab", "") for kw in floor_keywords)
    ]


def _find_wall_pieces(pieces: list[dict]) -> list[dict]:
    """Find all wall pieces in the build."""
    wall_keywords = ["wall", "Wall"]
    return [
        p for p in pieces
        if any(kw in p.get("prefab", "") for kw in wall_keywords)
    ]


def _find_roof_pieces(pieces: list[dict]) -> list[dict]:
    """Find all roof pieces in the build."""
    roof_keywords = ["roof", "Roof"]
    return [
        p for p in pieces
        if any(kw in p.get("prefab", "") for kw in roof_keywords)
    ]


def _get_bounds(pieces: list[dict]) -> dict:
    """Extract X/Z bounds from pieces, accounting for piece dimensions."""
    if not pieces:
        return {"x_min": 0, "x_max": 0, "z_min": 0, "z_max": 0}
    
    x_vals = []
    z_vals = []
    
    for p in pieces:
        x = p.get("x", 0)
        z = p.get("z", 0)
        prefab = p.get("prefab", "")
        
        # Get piece dimensions to find actual extents
        details = get_prefab_details(prefab)
        if details:
            half_w = details.get("width", 2) / 2
            half_d = details.get("depth", 2) / 2
        else:
            half_w = 1
            half_d = 1
        
        x_vals.extend([x - half_w, x + half_w])
        z_vals.extend([z - half_d, z + half_d])
    
    return {
        "x_min": round(min(x_vals), 1),
        "x_max": round(max(x_vals), 1),
        "z_min": round(min(z_vals), 1),
        "z_max": round(max(z_vals), 1),
    }


def _detect_floors(floor_pieces: list[dict], wall_pieces: list[dict], roof_pieces: list[dict]) -> list[dict]:
    """
    Detect floor levels from floor, wall, and roof pieces.
    
    Returns list of floors with y position and estimated height.
    """
    if not floor_pieces:
        return [{"y": 0, "height": 6}]  # Default single floor
    
    # Group floors by Y coordinate (within tolerance)
    y_values = [p.get("y", 0) for p in floor_pieces]
    
    # Cluster Y values (floors within 0.5m are same level)
    unique_ys = []
    for y in sorted(set(y_values)):
        if not unique_ys or abs(y - unique_ys[-1]) > 0.5:
            unique_ys.append(y)
    
    # For multi-story buildings, the floor-to-floor distance tells us wall height.
    # For the top floor, use roof base Y if available, otherwise default to 6m.
    default_wall_height = 6.0
    
    # If we have multiple floors, infer typical wall height from floor spacing
    if len(unique_ys) >= 2:
        # Use the first floor-to-floor distance as typical wall height
        default_wall_height = unique_ys[1] - unique_ys[0]
    
    # Find roof base Y (lowest roof piece) for top floor height calculation
    roof_base_y = None
    if roof_pieces:
        roof_ys = [p.get("y", 0) for p in roof_pieces]
        if roof_ys:
            roof_base_y = min(roof_ys)
    
    floors = []
    for i, y in enumerate(unique_ys):
        if i + 1 < len(unique_ys):
            # Not top floor: height is distance to next floor
            height = unique_ys[i + 1] - y
        else:
            # Top floor: use roof base if available, else default
            if roof_base_y is not None and roof_base_y > y:
                height = roof_base_y - y
            else:
                height = default_wall_height
        
        floors.append({"y": round(y, 1), "height": round(height, 1)})
    
    return floors


def _get_wall_top_y(floors: list[dict]) -> float:
    """Calculate the wall top Y from floor data."""
    if not floors:
        return 6.0
    
    top_floor = floors[-1]
    return round(top_floor["y"] + top_floor["height"], 1)


def _get_roof_base_y(roof_pieces: list[dict], wall_top_y: float) -> float:
    """Find the base Y of the roof (lowest roof piece)."""
    if not roof_pieces:
        return wall_top_y
    
    roof_ys = [p.get("y", wall_top_y) for p in roof_pieces]
    return round(min(roof_ys), 1)


def _get_corners(bounds: dict) -> list[tuple[float, float]]:
    """Return corner coordinates as (x, z) tuples."""
    return [
        (bounds["x_min"], bounds["z_min"]),  # SW
        (bounds["x_max"], bounds["z_min"]),  # SE
        (bounds["x_min"], bounds["z_max"]),  # NW
        (bounds["x_max"], bounds["z_max"]),  # NE
    ]


def analyze_building(pieces: list[dict]) -> dict:
    """
    Extract semantic building structure from a list of placed pieces.
    
    Returns a dict with:
    - bounds: {x_min, x_max, z_min, z_max}
    - floors: [{y, height}, ...]
    - corners: [(x, z), ...] for SW, SE, NW, NE
    - wall_top_y: Y position of wall tops
    - roof_base_y: Y position where roof starts
    - material: Primary building material (wood, stone, etc.)
    
    This analysis is used by the detail agent to understand building
    structure when generating enhancement descriptions.
    """
    if not pieces:
        return {
            "bounds": {"x_min": 0, "x_max": 0, "z_min": 0, "z_max": 0},
            "floors": [{"y": 0, "height": 6}],
            "corners": [(0, 0), (0, 0), (0, 0), (0, 0)],
            "wall_top_y": 6,
            "roof_base_y": 6,
            "material": "wood",
        }
    
    # Categorize pieces
    floor_pieces = _find_floor_pieces(pieces)
    wall_pieces = _find_wall_pieces(pieces)
    roof_pieces = _find_roof_pieces(pieces)
    
    # Use floor pieces for bounds if available, otherwise all pieces
    bounds_pieces = floor_pieces if floor_pieces else pieces
    bounds = _get_bounds(bounds_pieces)
    
    # Detect floors and heights
    floors = _detect_floors(floor_pieces, wall_pieces, roof_pieces)
    wall_top_y = _get_wall_top_y(floors)
    roof_base_y = _get_roof_base_y(roof_pieces, wall_top_y)
    
    # Get corner positions
    corners = _get_corners(bounds)
    
    # Detect primary material
    material = _detect_material(pieces)
    
    return {
        "bounds": bounds,
        "floors": floors,
        "corners": corners,
        "wall_top_y": wall_top_y,
        "roof_base_y": roof_base_y,
        "material": material,
    }


def format_building_analysis(analysis: dict) -> str:
    """
    Format building analysis as human-readable text for LLM prompts.
    
    Produces output like:
    - Bounds: x=[-6, 6], z=[-8, 8]
    - Floor at y=0, wall height 6m, wall top at y=6
    - Material: wood
    - Corners: SW(-6,-8), SE(6,-8), NW(-6,8), NE(6,8)
    """
    bounds = analysis["bounds"]
    floors = analysis["floors"]
    corners = analysis["corners"]
    
    lines = []
    
    # Bounds
    lines.append(
        f"- Bounds: x=[{bounds['x_min']}, {bounds['x_max']}], "
        f"z=[{bounds['z_min']}, {bounds['z_max']}]"
    )
    
    # Floors
    if len(floors) == 1:
        f = floors[0]
        lines.append(
            f"- Single floor at y={f['y']}, wall height {f['height']}m, "
            f"wall top at y={analysis['wall_top_y']}"
        )
    else:
        lines.append(f"- {len(floors)} floors:")
        for i, f in enumerate(floors):
            lines.append(f"  - Floor {i+1}: y={f['y']}, height={f['height']}m")
        lines.append(f"  - Wall top at y={analysis['wall_top_y']}")
    
    # Roof
    if analysis["roof_base_y"] != analysis["wall_top_y"]:
        lines.append(f"- Roof base at y={analysis['roof_base_y']}")
    
    # Material
    lines.append(f"- Material: {analysis['material']}")
    
    # Corners
    sw, se, nw, ne = corners
    lines.append(
        f"- Corners: SW{sw}, SE{se}, NW{nw}, NE{ne}"
    )
    
    return "\n".join(lines)
