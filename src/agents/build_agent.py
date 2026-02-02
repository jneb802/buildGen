"""
Build Agent - Stage 2 of the blueprint pipeline.

This agent takes a design document and produces the actual blueprint JSON
with exact piece positions and rotations.
"""

import json
import re
import anthropic

from src.tools.prefab_lookup import PREFAB_TOOLS, execute_tool as execute_prefab_tool
from src.tools.placement_tools import PLACEMENT_TOOLS, execute_placement_tool


def _extract_json(text: str) -> str:
    """
    Extract JSON from Claude's response, handling various formats.
    
    Claude might return:
    - Pure JSON
    - JSON wrapped in markdown code blocks
    - JSON with explanatory text before/after
    """
    # Remove markdown code blocks if present.
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(code_block_pattern, text)
    if match:
        return match.group(1).strip()
    
    # Try to find JSON object by locating outermost braces.
    start = text.find("{")
    if start == -1:
        return text
    
    # Find matching closing brace by counting depth.
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    # Fallback: return from first brace to end.
    return text[start:]


# ============================================================================
# System Prompt
# ============================================================================

BUILD_SYSTEM_PROMPT = """You are a Valheim blueprint generator. Your job is to convert design
documents into precise JSON piece arrays.

## Your Task

Given a design document, output a JSON array of pieces with exact positions and rotations.

## IMPORTANT: Use Procedural Placement Tools

ALWAYS use the procedural placement tools instead of calculating coordinates manually.
These tools generate pieces with correct positions deterministically:

- generate_floor_grid: Create complete floor coverage for a rectangular area
- generate_wall_line: Place walls along a line with optional filler pieces for gaps and corner posts
- generate_roof_slope: Create a row of sloped roof pieces

When using generate_wall_line, provide a filler_prefab (smaller wall piece) to fill any remaining gap.
For example, use stone_wall_4x2 with filler_prefab=stone_wall_1x1 to ensure complete wall coverage.

Call these tools, collect the returned pieces, then combine them into the final JSON.

## Output Format

Return ONLY valid JSON (no markdown code blocks) in this format:

{
  "name": "Building Name",
  "pieces": [
    {"prefab": "stone_floor_2x2", "x": 1.0, "y": 0.5, "z": 1.0, "rotY": 0},
    {"prefab": "stone_wall_2x1", "x": 0.0, "y": 1.25, "z": 2.0, "rotY": 0},
    ...
  ]
}

## Coordinate System

- Y is UP (vertical axis)
- X and Z are horizontal
- Units are in meters
- Position = CENTER of the piece (not corner)

## Wall Rotations (rotY values)

Walls face outward from the building center:
- North wall (faces +Z direction): rotY = 0
- East wall (faces +X direction): rotY = 90  
- South wall (faces -Z direction): rotY = 180
- West wall (faces -X direction): rotY = 270

## Roof Rotations

For sloped roof pieces:
- Slopes toward -Z (south): rotY = 0
- Slopes toward +Z (north): rotY = 180
- Ridge runs along X axis: rotY = 90 or 270

## Critical Rules

1. Use EXACT prefab names from the design document
2. Query get_prefab_details() to get piece dimensions before positioning
3. Place pieces so they connect at snap points
4. rotY must be 0, 90, 180, or 270 - no other values

## ARCHITECTURAL RULE 1: Wall Y-Positioning

Walls must sit ON TOP of floors, not inside them:
- First, place the floor at y = 0.5 (floor center)
- Floor top surface is at y = 0.5 + floor_height/2
- Wall base starts at floor top surface
- Wall center y = floor_top + wall_height/2

Example with stone_floor_2x2 (height ~0.25) and stone_wall_2x1 (height ~1.0):
- Floor center: y = 0.5
- Floor top: y = 0.5 + 0.125 = 0.625 (approximately 0.5 for simplicity)
- Wall center: y = 0.5 + 1.0/2 = 1.0 for first row
- Second wall row: y = 1.0 + 1.0 = 2.0

For second floor:
- Second floor foundation y = first_floor_wall_top
- Second floor walls start on top of second floor foundation

## ARCHITECTURAL RULE 2: Floor Coverage

Floors must tile completely with NO GAPS:
- Calculate exact number of tiles needed: (building_width / tile_width) × (building_depth / tile_depth)
- Place tiles in a complete grid pattern
- VERIFY: total floor pieces = width_tiles × depth_tiles

Example for 4m × 4m building with 2m × 2m tiles:
- Need 2 × 2 = 4 floor tiles
- Positions: (1,0.5,1), (3,0.5,1), (1,0.5,3), (3,0.5,3)
- All 4 tiles present, no gaps

## ARCHITECTURAL RULE 3: Wall Openings

Every wall opening MUST have a door, arch, or window piece:
- If design mentions "entrance" or "door", place a door/arch prefab there
- NEVER leave an empty gap in walls
- Door/arch pieces REPLACE wall segments at that position
- If no door piece available, use a full wall instead

## ARCHITECTURAL RULE 4: Roof Placement

Roof pieces attach to the EXTERIOR edge of walls, not interior:
- North-sloping roof (rotY=180): place at z = max_wall_z + roof_depth/2
- South-sloping roof (rotY=0): place at z = min_wall_z - roof_depth/2
- Roof pieces should NOT overlap each other
- Two opposing slopes meet at the ridge line in the CENTER of the building

Example for 4m deep building:
- South wall at z = 0, North wall at z = 4
- South roof at z = -0.5 to 1.5 (outside south wall)
- North roof at z = 2.5 to 4.5 (outside north wall)
- Roofs meet at z = 2.0 (building center)

## ARCHITECTURAL RULE 5: Roof Completion

Only add ridge/roof_top pieces when slopes don't meet:
- For a simple gabled roof where two slopes meet at apex: NO ridge piece needed
- For a 4m wide building with 2m roof pieces from each side: slopes meet, no ridge
- For 6m+ wide building: slopes may not meet, add ridge pieces to fill gap

Calculate: if (building_width / 2) <= roof_piece_width, slopes meet at center.

## Complete Example: 4×4m Two-Story Stone Tower

### Floor 1 Foundation (stone_floor_2x2)
- Tile size: 2m × 2m, height 0.25m
- Need: 2 × 2 = 4 tiles
- Positions (center coords):
  - (1, 0.5, 1), (3, 0.5, 1), (1, 0.5, 3), (3, 0.5, 3)

### Floor 1 Walls (stone_wall_2x1, 2m wide, 1m tall)
- Floor top at y ≈ 0.5
- Wall center y = 0.5 + 0.5 = 1.0
- North wall (z=4): pieces at x=1 and x=3, rotY=0
- South wall (z=0): piece at x=3 rotY=180, door at x=1
- East wall (x=4): pieces at z=1 and z=3, rotY=90
- West wall (x=0): pieces at z=1 and z=3, rotY=270
- Door piece (stone_arch) at x=1, z=0, y=1.0, rotY=180

### Floor 2 Foundation
- Wall top at y = 1.0 + 0.5 = 1.5
- Floor 2 foundation y = 1.5 + 0.125 ≈ 1.5
- Same 4-tile pattern: (1, 1.5, 1), (3, 1.5, 1), (1, 1.5, 3), (3, 1.5, 3)

### Floor 2 Walls
- Floor 2 top at y ≈ 1.5
- Wall center y = 1.5 + 0.5 = 2.0
- Complete walls on all 4 sides (no door on second floor)

### Roof (2m roof pieces, 45 degree)
- Wall top at y = 2.0 + 0.5 = 2.5
- Roof base y = 2.5
- South slope (rotY=0) at z = -0.5, x = 1 and x = 3
- North slope (rotY=180) at z = 4.5, x = 1 and x = 3
- Building is 4m wide, roof pieces are 2m: they meet at center
- NO roof_top ridge piece needed
"""


# ============================================================================
# Agent Execution
# ============================================================================

def run_build_agent(
    design_doc: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False
) -> dict:
    """
    Run the build agent to convert a design document into blueprint JSON.
    
    Args:
        design_doc: The design document markdown from the design agent
        model: Claude model to use
        verbose: Whether to print debug info
    
    Returns a dict with 'name' and 'pieces' keys.
    """
    client = anthropic.Anthropic()
    
    user_message = f"""Convert this design document into a blueprint JSON:

{design_doc}

Remember to:
1. Use the procedural placement tools (generate_floor_grid, generate_wall_line, etc.)
2. Use get_prefab_details() to check dimensions when needed
3. Combine all generated pieces into the final JSON
4. Output ONLY valid JSON, no markdown"""

    messages = [{"role": "user", "content": user_message}]
    
    # Combine prefab lookup tools with placement tools.
    all_tools = PREFAB_TOOLS + PLACEMENT_TOOLS
    
    # Track consecutive identical errors to detect infinite loops.
    last_error = None
    consecutive_error_count = 0
    MAX_CONSECUTIVE_ERRORS = 3
    
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8192,  # Larger for potentially many pieces.
            system=BUILD_SYSTEM_PROMPT,
            tools=all_tools,
            messages=messages
        )
        
        if verbose:
            print(f"[Build Agent] Stop reason: {response.stop_reason}")
        
        if response.stop_reason == "tool_use":
            tool_results = []
            assistant_content = []
            had_error_this_round = False
            current_error = None
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    if verbose:
                        print(f"[Build Agent] Tool call: {block.name}({block.input})")
                    
                    # Dispatch to the appropriate tool executor.
                    placement_tool_names = [
                        "place_piece",
                        "generate_floor_grid",
                        "generate_wall_line", 
                        "generate_roof_slope"
                    ]
                    if block.name in placement_tool_names:
                        result = execute_placement_tool(block.name, block.input)
                    else:
                        result = execute_prefab_tool(block.name, block.input)
                    
                    # Check if result contains an error.
                    if '"error"' in result:
                        had_error_this_round = True
                        current_error = result
                    
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Track consecutive identical errors.
            if had_error_this_round:
                if current_error == last_error:
                    consecutive_error_count += 1
                else:
                    consecutive_error_count = 1
                    last_error = current_error
                
                if consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"Build agent stuck in error loop. "
                        f"Same error occurred {MAX_CONSECUTIVE_ERRORS} times: {last_error}"
                    )
            else:
                # Reset on successful tool calls.
                consecutive_error_count = 0
                last_error = None
            
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Extract final text and parse as JSON.
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    
                    # Try to extract JSON from the response, handling various formats.
                    json_text = _extract_json(text)
                    
                    if verbose:
                        print(f"[Build Agent] Extracted JSON: {json_text[:500]}...")
                    
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"[Build Agent] JSON parse error: {e}")
                            print(f"[Build Agent] Raw text: {text[:500]}...")
                        # Return empty blueprint on parse failure.
                        return {"name": "Parse Error", "pieces": [], "raw_response": text}
            
            return {"name": "Empty Response", "pieces": []}
