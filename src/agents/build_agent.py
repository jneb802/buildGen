"""
Build Agent - Stage 2 of the blueprint pipeline.

This agent takes a design document and produces the actual blueprint JSON
with exact piece positions and rotations.
"""

import json
import re
import anthropic

from src.agents.design_agent import AgentResult
from src.tools.prefab_lookup import BUILD_TOOLS, execute_tool as execute_prefab_tool
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

## IMPORTANT: Use Composite Placement Tools

ALWAYS use composite tools for structural elements. They handle snapping internally
and are much more efficient than placing pieces one by one:

- generate_floor_grid: Create complete floor coverage (pieces snap to each other)
- generate_wall_line: Place walls along a line (walls chain-snap to each other)
- generate_roof_slope: Create row of roof pieces (pieces chain-snap to each other)
- place_piece: ONLY for individual pieces (doors, stairs, decorations)

## Snapping with anchor_pieces

Composite tools handle snapping internally. To connect new pieces to existing structure,
pass anchor_pieces - the tool will snap the first piece to the anchors, then chain-snap
the rest:

Example - walls snapping to floor edges:
  floor_pieces = generate_floor_grid(prefab="stone_floor_2x2", width=8, depth=8, y=0.5)
  wall_pieces = generate_wall_line(
      prefab="stone_wall_4x2", start_x=0, start_z=8, end_x=8, end_z=8,
      y=1.65, rotY=0, anchor_pieces=floor_pieces  # Snaps first wall to floor edge
  )

Example - roof snapping to walls:
  roof_pieces = generate_roof_slope(
      prefab="wood_roof_45", start_x=0, start_z=0, y=5.0, count=4,
      direction="east", rotY=0, anchor_pieces=wall_pieces  # Snaps to wall tops
  )

## Reading Design Documents

Design documents specify CONSTRAINTS, not coordinates. You must translate them to tool calls:

### Floor Sections
Design says:
  - surface_y: 0.5
  - bounds: x=[-4 to 4], z=[-4 to 4]
  - prefab: stone_floor_2x2

You call:
  generate_floor_grid(prefab="stone_floor_2x2", width=8, depth=8, y=0.5, origin_x=-4, origin_z=-4)

### Wall Sections
Design says:
  - North: z=4, x=[-4 to 4], prefab=stone_wall_4x2, filler=stone_wall_1x1
  - surface_y: 0.5

You call:
  1. get_prefab_details("stone_wall_4x2") to get wall height
  2. Calculate wall_center_y = surface_y + wall_height/2
  3. generate_wall_line(prefab="stone_wall_4x2", start_x=-4, start_z=4, end_x=4, end_z=4, 
                        y=wall_center_y, rotY=0, filler_prefab="stone_wall_1x1",
                        anchor_pieces=floor_pieces)  # Pass floor pieces for snapping

### Wall Openings
Design says:
  - South: z=-4, x=[-4 to 4], opening=stone_arch at x=0

You call:
  1. generate_wall_line for left segment: x=[-4 to -1]
  2. place_piece for the arch at x=0 (use snap=true with nearby wall pieces if needed)
  3. generate_wall_line for right segment: x=[1 to 4]

## Output Format

Return ONLY valid JSON (no markdown code blocks) in this format:

{
  "name": "Building Name",
  "pieces": [
    {"prefab": "stone_floor_2x2", "x": 1.0, "y": 0.5, "z": 1.0, "rotY": 0},
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
- North wall (at +Z edge, faces +Z): rotY = 0
- East wall (at +X edge, faces +X): rotY = 90  
- South wall (at -Z edge, faces -Z): rotY = 180
- West wall (at -X edge, faces -X): rotY = 270

## Critical Rules

1. Use EXACT prefab names from the design document
2. Query get_prefab_details() to get piece dimensions for Y calculations
3. rotY must be 0, 90, 180, or 270 - no other values
4. ALWAYS use filler_prefab in generate_wall_line when design specifies one

## Y Position Calculations

Design documents provide surface_y (where things sit ON TOP of a floor).
You must calculate piece center Y:

For floors:
  - floor_center_y = surface_y (generate_floor_grid handles this)

For walls on a floor with surface_y:
  - wall_center_y = surface_y + (wall_height / 2)

For second floor surface:
  - floor2_surface_y = floor1_surface_y + wall_height + floor_thickness

## Wall Opening Handling

When design specifies an opening (door, arch, window):
1. Split the wall line at the opening
2. Place wall segments on either side
3. Place the opening piece at the specified position
4. Opening piece uses same wall_center_y calculation

## Roof Placement

- base_y from design = where roof pieces start (top of walls)
- For gabled roofs, place slopes on opposite sides meeting at ridge
- Use ridge_prefab only if slopes don't meet in center
- Use corner_prefab at roof corners if specified

## Example Workflow

Given this design section:
```
## FLOORS
- Ground floor: surface_y=0.5, bounds x=[0 to 8], z=[0 to 8], prefab=stone_floor_2x2

## WALLS
### Ground Floor Walls (surface_y = 0.5)
- North: z=8, x=[0 to 8], prefab=stone_wall_4x2
```

Your process:
1. Generate floors first (they are the anchor for walls):
   floor_pieces = generate_floor_grid(prefab="stone_floor_2x2", width=8, depth=8, y=0.5)

2. Get wall dimensions:
   get_prefab_details("stone_wall_4x2") → height=2.3

3. Calculate wall Y and generate with floor as anchor:
   wall_y = 0.5 + 2.3/2 = 1.65
   north_walls = generate_wall_line(
       prefab="stone_wall_4x2", start_x=0, start_z=8, end_x=8, end_z=8,
       y=1.65, rotY=0, anchor_pieces=floor_pieces
   )

4. Collect ALL returned pieces into final JSON

## Efficiency Tips

- Use generate_wall_line for entire walls, NOT individual place_piece calls
- One generate_wall_line call replaces 4-8 place_piece calls
- Pass anchor_pieces to connect walls to floors, roofs to walls
- Reserve place_piece for doors, stairs, and decorations only
"""


# ============================================================================
# Agent Execution
# ============================================================================

def run_build_agent(
    design_doc: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False
) -> AgentResult:
    """
    Run the build agent to convert a design document into blueprint JSON.
    
    Args:
        design_doc: The design document markdown from the design agent
        model: Claude model to use
        verbose: Whether to print debug info
    
    Returns an AgentResult with the blueprint dict and usage stats.
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
    
    # Combine prefab detail lookup with placement tools.
    all_tools = BUILD_TOOLS + PLACEMENT_TOOLS
    
    # Track tool calls and usage for logging.
    tool_call_log: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_write = 0
    api_call_count = 0
    
    # Track consecutive identical errors to detect infinite loops.
    last_error = None
    consecutive_error_count = 0
    MAX_CONSECUTIVE_ERRORS = 3
    
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=8192,  # Larger for potentially many pieces.
            system=[
                {
                    "type": "text",
                    "text": BUILD_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=all_tools,
            messages=messages
        )
        
        # Track usage from this API call.
        api_call_count += 1
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_read += getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        total_cache_write += getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        
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
                    # Log this tool call.
                    tool_call_str = f"{block.name}({block.input})"
                    tool_call_log.append(tool_call_str)
                    
                    if verbose:
                        print(f"[Build Agent] Tool call: {tool_call_str}")
                    
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
            blueprint = {"name": "Empty Response", "pieces": []}
            
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    
                    # Try to extract JSON from the response, handling various formats.
                    json_text = _extract_json(text)
                    
                    if verbose:
                        print(f"[Build Agent] Extracted JSON: {json_text[:500]}...")
                    
                    try:
                        blueprint = json.loads(json_text)
                    except json.JSONDecodeError as e:
                        if verbose:
                            print(f"[Build Agent] JSON parse error: {e}")
                            print(f"[Build Agent] Raw text: {text[:500]}...")
                        blueprint = {"name": "Parse Error", "pieces": [], "raw_response": text}
                    break
            
            return AgentResult(
                result=blueprint,
                tool_calls=tool_call_log,
                api_calls=api_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_write_tokens=total_cache_write
            )
