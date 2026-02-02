# Valheim Blueprint Generator - Python CLI Specification

## Overview

Build a Python CLI tool that generates Valheim blueprints from natural language descriptions. The tool uses an agentic LLM pipeline with the Anthropic Claude API.

## Pipeline Stages

| Stage | Name | LLM | Input | Output |
|-------|------|-----|-------|--------|
| 1 | Design | Yes | User prompt + prefab data | Design document (markdown) |
| 2 | Build | Yes | Design doc + prefab data | Blueprint JSON (snap correction handled inline) |

## CLI Interface

```bash
# Basic usage
valheim-blueprint "a small stone watchtower with two floors"

# With options
valheim-blueprint "viking longhouse" --output ./builds --model claude-sonnet-4-20250514
```

### Arguments
- `prompt` (positional): Building description
- `--output` / `-o`: Output directory (default: `./output`)
- `--model` / `-m`: Claude model (default: `claude-sonnet-4-20250514`)
- `--verbose` / `-v`: Show detailed progress

### Outputs
All outputs go to `{output_dir}/{timestamp}/`:
- `design.md` - Design document from Stage 1
- `blueprint.json` - Final validated blueprint
- `log.txt` - Pipeline execution log (if verbose)

---

## Reference Files

These C# files contain logic and data to port to Python:

### 1. Prefab Database
**File:** `ValheimPrefabDatabase.cs`

Contains all Valheim building prefabs with:
- `name`: Internal prefab name (use exactly)
- `englishName`: Display name
- `description`: What it is
- `biome`: Where it's unlocked (Meadows, BlackForest, etc.)
- `width`, `height`, `depth`: Bounding box in meters
- `snapPoints`: Array of Vector3 offsets from piece center

**Action:** Export to `prefabs.json` for the Python tool.

Example prefab structure:
```json
{
  "name": "stone_wall_4x2",
  "englishName": "Stone Wall 4x2",
  "description": "Large stone wall section",
  "biome": "BlackForest",
  "width": 4.28,
  "height": 2.303,
  "depth": 1.558,
  "snapPoints": [
    {"x": 2.0, "y": 1.0, "z": 0.5},
    {"x": 2.0, "y": 1.0, "z": -0.5},
    {"x": 2.0, "y": -1.0, "z": 0.5},
    {"x": 2.0, "y": -1.0, "z": -0.5},
    {"x": -2.0, "y": 1.0, "z": 0.5},
    {"x": -2.0, "y": 1.0, "z": -0.5},
    {"x": -2.0, "y": -1.0, "z": 0.5},
    {"x": -2.0, "y": -1.0, "z": -0.5}
  ]
}
```

Prefab categories in the database:
- WoodPieces (floors, walls, roofs, poles, beams, doors, stairs)
- LogPieces (corewood poles and beams)
- DarkwoodPieces (tar-treated roofs, poles, beams)
- StonePieces (walls, floors, pillars, arches)
- IronPieces (cage walls/floors, iron gates)
- BlackMarblePieces (Mistlands stone)
- DvergrPieces (dwarven structures)
- GraustenPieces (Ashlands stone)
- AshwoodPieces (Ashlands wood)
- FlametalPieces (Ashlands metal)
- DefensePieces (stakewalls, traps)
- FurniturePieces (beds, chairs, tables, chests)
- LightingPieces (torches, braziers)
- CraftingPieces (workbenches, forges, smelters)
- UtilityPieces (portals, wards, signs)
- VehiclePieces (ships, carts)

---

### 2. Blueprint Schema
**File:** `BlueprintData.cs`

Defines the output format:

```json
{
  "name": "Building Name",
  "creator": "BlueprintGenerator",
  "description": "Optional description",
  "category": "Misc",
  "pieces": [
    {
      "prefab": "stone_floor_2x2",
      "x": 1.0,
      "y": 0.5,
      "z": 1.0,
      "rotY": 0
    }
  ]
}
```

**Piece fields:**
- `prefab`: Exact prefab name from database
- `x`, `y`, `z`: Position (center of piece) in meters
- `rotY`: Rotation around Y axis in degrees (0, 90, 180, 270)

---

### 3. Design Agent Prompt
**File:** `LLM/SemanticKernel/Agents/DesignAgentPrompt.cs`

System prompt for Stage 1. Key sections:

- Role: Valheim building architect
- Input: User's build request + prefab reference
- Output: Structured design document with:
  - Overview (name, dimensions, materials, floors)
  - Prefabs to use (specific names from database)
  - Foundation (floor grid layout)
  - Walls (per floor, with Y positions calculated)
  - Roof (style, angle, positioning)
  - Stairs (if multi-story)

**Critical instruction:** Calculate Y positions from snap point spacing:
```
row_y = (row_number × vertical_snap) + (vertical_snap / 2)

Example with stone_wall_4x2 (2m vertical snap):
  Row 0: y = 0 × 2 + 1 = 1
  Row 1: y = 1 × 2 + 1 = 3
  Row 2: y = 2 × 2 + 1 = 5
```

---

### 4. Build Agent Prompt
**File:** `LLM/SemanticKernel/Agents/BuildAgentPrompt.cs`

System prompt for Stage 2. Key sections:

- Role: Blueprint generator
- Input: Design document + prefab reference
- Output: JSON array of pieces

**Coordinate system:**
- Y is UP (vertical)
- X and Z are horizontal
- Units in meters
- Position = center of piece

**Wall rotations:**
- North (positive Z): rotY = 0
- East (positive X): rotY = 90
- South (negative Z): rotY = 180
- West (negative X): rotY = 270

**Roof rotations:**
- South slope (toward -Z): rotY = 0
- North slope (toward +Z): rotY = 180
- Ridge along X: rotY = 90

---

## Example Outputs

### Design Document Example
See: `LLMResponses/Generated_Blueprint_20260131_115822/design.md`

```markdown
# STONE TOWER DESIGN DOCUMENT

## OVERVIEW
- Building name: Small Stone Watchtower
- Overall dimensions: 6m x 6m x 12m
- Primary materials: Stone walls, wood floors, darkwood roof
- Number of floors: 2 floors plus ground level

## FOUNDATION
- Floor dimensions: 6m x 6m stone floor
- Material: stone_floor_2x2 pieces (9 pieces in 3x3 grid)
- Position: y=0 (ground level)

## WALLS
### Ground Floor (y=0 to y=4)
- North wall: 6m length, 4m height, stone_wall_4x2 vertical stack
- South wall: door opening at center
...
```

### Blueprint JSON Example
See: `LLMResponses/Generated_Blueprint_20260131_115822/blueprint.json`

```json
{
  "pieces": [
    { "prefab": "stone_floor_2x2", "x": 1, "y": 0.5, "z": 1, "rotY": 0 },
    { "prefab": "stone_floor_2x2", "x": 3, "y": 0.5, "z": 1, "rotY": 0 },
    { "prefab": "stone_wall_4x2", "x": 1, "y": 1, "z": 6, "rotY": 0 },
    { "prefab": "wood_door", "x": 3, "y": 1, "z": 0, "rotY": 180 },
    { "prefab": "wood_roof_45", "x": 1, "y": 10, "z": 1, "rotY": 0 },
    { "prefab": "wood_roof_top_45", "x": 1, "y": 12, "z": 3, "rotY": 90 }
  ]
}
```

---

## Suggested Python Project Structure

```
valheim-blueprint-cli/
├── pyproject.toml
├── README.md
├── src/
│   └── valheim_blueprint/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (click or argparse)
│       ├── pipeline.py         # Orchestrates the 3 stages
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── design_agent.py # Stage 1: Design document generation
│       │   └── build_agent.py  # Stage 2: Blueprint JSON generation
│       ├── tools/
│       │   ├── __init__.py
│       │   └── prefab_lookup.py    # Query prefab database
│       ├── data/
│       │   └── prefabs.json    # Exported from ValheimPrefabDatabase.cs
│       └── models.py           # Pydantic models for Prefab, Piece, Blueprint
└── tests/
    └── ...
```

---

## Implementation Notes

### Anthropic SDK Usage

```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

# With tools
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=system_prompt,
    tools=[
        {
            "name": "get_prefabs",
            "description": "Query Valheim prefabs by material and category",
            "input_schema": {
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "category": {"type": "string"}
                }
            }
        }
    ],
    messages=[{"role": "user", "content": user_prompt}]
)

# Handle tool use
for block in response.content:
    if block.type == "tool_use":
        # Execute tool, send result back
        ...
```

### Tool Definitions for Agents

**Design Agent Tools:**
1. `get_prefabs(material, category)` - Query prefabs by filter
2. `get_prefab_details(name)` - Get specific prefab info
3. `list_materials()` - List available materials
4. `list_categories()` - List available categories

**Build Agent Tools:**
1. `get_prefab_details(name)` - Get dimensions and snap points
2. `calculate_snap_position(...)` - Calculate exact placement

---

## Data Export

The prefab database needs to be exported from the C# source. Two options:

### Option 1: Unity Editor Script
Add this to a Unity Editor script and run from menu:

```csharp
[MenuItem("Tools/Export Prefabs JSON")]
static void ExportPrefabs()
{
    var prefabs = ValheimPrefabDatabase.AllPrefabs.Select(p => new {
        name = p.name,
        englishName = p.englishName,
        description = p.description,
        biome = p.biome.ToString(),
        width = p.width,
        height = p.height,
        depth = p.depth,
        snapPoints = p.snapPoints?.Select(sp => new { x = sp.x, y = sp.y, z = sp.z }).ToArray()
    });
    var json = JsonConvert.SerializeObject(prefabs, Formatting.Indented);
    File.WriteAllText("prefabs.json", json);
    Debug.Log($"Exported {ValheimPrefabDatabase.AllPrefabs.Count} prefabs");
}
```

### Option 2: Parse C# Source
The Python tool can include a build script that parses `ValheimPrefabDatabase.cs` using regex to extract prefab data. The file format is consistent:

```
new PrefabInfo("name", "English Name", "description", Biome.X, width, height, depth,
    new Vector3[] { new Vector3(x, y, z), ... })
```

---

## Dependencies

```toml
[project]
dependencies = [
    "anthropic>=0.40.0",
    "click>=8.0",
    "pydantic>=2.0",
    "rich>=13.0",  # For nice CLI output
]
```

---

## Testing

Test with these prompts:
1. "a simple wooden cabin with one room"
2. "a stone watchtower with two floors"
3. "a viking longhouse with a central hearth"
4. "a blackmarble temple entrance"

Validate outputs by:
1. Checking all prefab names exist in database
2. Verifying snap point connections
3. Ensuring no overlapping pieces at same position
