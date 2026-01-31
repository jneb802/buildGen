"""
Build Agent - Stage 2 of the blueprint pipeline.

This agent takes a design document and produces the actual blueprint JSON
with exact piece positions and rotations.
"""

import json
import anthropic

from src.tools.prefab_lookup import PREFAB_TOOLS, execute_tool


# ============================================================================
# System Prompt
# ============================================================================

BUILD_SYSTEM_PROMPT = """You are a Valheim blueprint generator. Your job is to convert design
documents into precise JSON piece arrays.

## Your Task

Given a design document, output a JSON array of pieces with exact positions and rotations.

## Output Format

Return ONLY valid JSON (no markdown code blocks) in this format:

{
  "name": "Building Name",
  "pieces": [
    {"prefab": "stone_floor_2x2", "x": 1.0, "y": 0.5, "z": 1.0, "rotY": 0},
    {"prefab": "stone_wall_4x2", "x": 0.0, "y": 1.0, "z": 3.0, "rotY": 0},
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
5. Calculate positions carefully:
   - Floor tiles: place in a grid, y is typically 0.5 for ground level
   - Walls: y = row_height/2 + (row_number × row_height)
   - Roofs: start above the top wall row

## Example Calculation

For a 6x6m building with stone_floor_2x2 (2m tiles):
- Need 3x3 grid of floor tiles
- Positions: (1,0.5,1), (3,0.5,1), (5,0.5,1), (1,0.5,3), ...

For stone_wall_4x2 (4m wide, 2m tall):
- Row 0: y = 1.0 (half the height)
- Row 1: y = 3.0 (1.0 + 2.0)
- Row 2: y = 5.0 (1.0 + 4.0)
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
    
    Returns a dict with 'name' and 'pieces' keys.
    """
    client = anthropic.Anthropic()
    
    user_message = f"""Convert this design document into a blueprint JSON:

{design_doc}

Remember to:
1. Use get_prefab_details() to check dimensions
2. Calculate positions precisely  
3. Output ONLY valid JSON, no markdown"""

    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8192,  # Larger for potentially many pieces.
            system=BUILD_SYSTEM_PROMPT,
            tools=PREFAB_TOOLS,
            messages=messages
        )
        
        if verbose:
            print(f"[Build Agent] Stop reason: {response.stop_reason}")
        
        if response.stop_reason == "tool_use":
            tool_results = []
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    if verbose:
                        print(f"[Build Agent] Tool call: {block.name}({block.input})")
                    
                    result = execute_tool(block.name, block.input)
                    
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
            
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Extract final text and parse as JSON.
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    
                    # Strip markdown code blocks if present.
                    if text.startswith("```"):
                        lines = text.split("\n")
                        # Remove first and last lines (``` markers).
                        text = "\n".join(lines[1:-1])
                    
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"[Build Agent] JSON parse error: {e}")
                            print(f"[Build Agent] Raw text: {text[:500]}...")
                        # Return empty blueprint on parse failure.
                        return {"name": "Parse Error", "pieces": []}
            
            return {"name": "Empty Response", "pieces": []}
