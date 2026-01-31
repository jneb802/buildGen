"""
Prefab lookup tools for Claude agents.

These functions let agents query the prefab database to find appropriate pieces
for building. They're exposed as Claude tools during the design and build stages.
"""

import json
from pathlib import Path
from functools import lru_cache

from src.models import Prefab, Vector3


# ============================================================================
# Data Loading
# ============================================================================

@lru_cache(maxsize=1)
def _load_prefabs() -> list[Prefab]:
    """Load prefabs from JSON file. Cached so we only read once."""
    data_path = Path(__file__).parent.parent / "data" / "prefabs.json"
    with open(data_path) as f:
        raw = json.load(f)
    
    prefabs = []
    for item in raw:
        # Convert snap points to Vector3 objects if present.
        snap_points = None
        if item.get("snapPoints"):
            snap_points = [Vector3(**sp) for sp in item["snapPoints"]]
        
        prefabs.append(Prefab(
            name=item["name"],
            englishName=item["englishName"],
            description=item["description"],
            biome=item["biome"],
            width=item["width"],
            height=item["height"],
            depth=item["depth"],
            snapPoints=snap_points
        ))
    
    return prefabs


@lru_cache(maxsize=1)
def _prefab_by_name() -> dict[str, Prefab]:
    """Build a lookup dict for quick access by prefab name."""
    return {p.name: p for p in _load_prefabs()}


# ============================================================================
# Material and Category Mapping
# ============================================================================

# Maps user-friendly material names to prefab name patterns.
MATERIAL_PATTERNS = {
    "wood": ["wood_", "woodwall"],
    "log": ["wood_pole_log", "wood_wall_log", "wood_log_"],
    "darkwood": ["darkwood_"],
    "stone": ["stone_"],
    "iron": ["iron_", "woodiron_"],
    "blackmarble": ["blackmarble_"],
    "dvergr": ["piece_dvergr_", "piece_hexagonal_"],
    "grausten": ["piece_grausten_", "Piece_grausten_"],
    "ashwood": ["ashwood_", "Ashwood_"],
    "flametal": ["piece_flametal_", "Piece_flametal_", "flametal_"],
}

# Maps categories to keywords in prefab names or descriptions.
CATEGORY_KEYWORDS = {
    "floor": ["floor", "Floor"],
    "wall": ["wall", "Wall"],
    "roof": ["roof", "Roof"],
    "pole": ["pole", "Pole", "pillar", "Pillar", "column", "Column"],
    "beam": ["beam", "Beam"],
    "door": ["door", "Door", "gate", "Gate"],
    "stair": ["stair", "Stair", "ladder", "stepladder"],
    "window": ["window", "Window", "shutter", "Shutter"],
    "furniture": ["bed", "chair", "bench", "table", "throne", "chest", "Bed", "Chair", "Bench", "Table", "Throne", "Chest"],
    "lighting": ["torch", "brazier", "lantern", "candle", "Torch", "Brazier", "Lantern", "Candle"],
    "crafting": ["workbench", "forge", "cauldron", "smelter", "Workbench", "Forge", "Cauldron", "Smelter"],
    "defense": ["stake", "sharp", "trap", "turret", "shield"],
}


# ============================================================================
# Tool Functions (exposed to Claude)
# ============================================================================

def list_materials() -> list[str]:
    """Return all available material types that can be used to filter prefabs."""
    return list(MATERIAL_PATTERNS.keys())


def list_categories() -> list[str]:
    """Return all available piece categories that can be used to filter prefabs."""
    return list(CATEGORY_KEYWORDS.keys())


def get_prefabs(material: str | None = None, category: str | None = None) -> list[dict]:
    """
    Get prefabs filtered by material and/or category.
    
    Returns a simplified list with name, englishName, description, and dimensions.
    Use get_prefab_details() to get full info including snap points.
    """
    prefabs = _load_prefabs()
    results = []
    
    for p in prefabs:
        # Filter by material if specified.
        if material:
            patterns = MATERIAL_PATTERNS.get(material.lower(), [])
            if not any(pat in p.name for pat in patterns):
                continue
        
        # Filter by category if specified.
        if category:
            keywords = CATEGORY_KEYWORDS.get(category.lower(), [])
            name_and_desc = p.name + " " + p.englishName + " " + p.description
            if not any(kw in name_and_desc for kw in keywords):
                continue
        
        results.append({
            "name": p.name,
            "englishName": p.englishName,
            "description": p.description,
            "biome": p.biome,
            "width": p.width,
            "height": p.height,
            "depth": p.depth,
        })
    
    return results


def get_prefab_details(name: str) -> dict | None:
    """
    Get full details for a specific prefab by its exact name.
    
    Returns all info including snap points, or None if not found.
    """
    lookup = _prefab_by_name()
    prefab = lookup.get(name)
    
    if not prefab:
        return None
    
    snap_points = None
    if prefab.snapPoints:
        snap_points = [{"x": sp.x, "y": sp.y, "z": sp.z} for sp in prefab.snapPoints]
    
    return {
        "name": prefab.name,
        "englishName": prefab.englishName,
        "description": prefab.description,
        "biome": prefab.biome,
        "width": prefab.width,
        "height": prefab.height,
        "depth": prefab.depth,
        "snapPoints": snap_points,
    }


# ============================================================================
# Claude Tool Definitions
# ============================================================================

# These are the tool schemas passed to Claude's API.

PREFAB_TOOLS = [
    {
        "name": "list_materials",
        "description": "Get all available material types (wood, stone, darkwood, etc.) for filtering prefabs.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_categories",
        "description": "Get all available piece categories (floor, wall, roof, door, etc.) for filtering prefabs.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_prefabs",
        "description": "Query prefabs by material and/or category. Returns name, description, and dimensions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "Material type: wood, log, darkwood, stone, iron, blackmarble, dvergr, grausten, ashwood, flametal"
                },
                "category": {
                    "type": "string",
                    "description": "Piece category: floor, wall, roof, pole, beam, door, stair, window, furniture, lighting, crafting, defense"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_prefab_details",
        "description": "Get full details for a specific prefab including dimensions and snap points. Use exact prefab name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact prefab name (e.g. 'stone_wall_4x2', 'wood_floor')"
                }
            },
            "required": ["name"]
        }
    }
]


def execute_tool(name: str, args: dict) -> str:
    """
    Execute a prefab tool by name and return JSON result.
    
    Called by the pipeline when Claude uses a tool.
    """
    if name == "list_materials":
        result = list_materials()
    elif name == "list_categories":
        result = list_categories()
    elif name == "get_prefabs":
        result = get_prefabs(args.get("material"), args.get("category"))
    elif name == "get_prefab_details":
        result = get_prefab_details(args.get("name", ""))
    else:
        result = {"error": f"Unknown tool: {name}"}
    
    return json.dumps(result, indent=2)
