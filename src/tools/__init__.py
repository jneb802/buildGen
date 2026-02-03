# Tool implementations for Claude agents and validation.

from src.tools.prefab_lookup import (
    DESIGN_TOOLS,
    BUILD_TOOLS,
    execute_tool,
    get_prefab_details,
    get_prefabs,
    list_materials,
    list_categories,
)

from src.tools.placement_tools import (
    PLACEMENT_TOOLS,
    execute_placement_tool,
    place_piece,
    generate_floor_grid,
    generate_floor_walls,
    generate_wall,  # Internal, used by generate_floor_walls
    generate_roof,
)

__all__ = [
    # Prefab lookup
    "DESIGN_TOOLS",
    "BUILD_TOOLS",
    "execute_tool",
    "get_prefab_details",
    "get_prefabs",
    "list_materials",
    "list_categories",
    # Placement tools
    "PLACEMENT_TOOLS",
    "execute_placement_tool",
    # Primitive action
    "place_piece",
    # Composite actions
    "generate_floor_grid",
    "generate_floor_walls",
    "generate_wall",  # Internal helper
    "generate_roof",
]
