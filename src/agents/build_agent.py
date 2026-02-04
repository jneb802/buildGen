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

BUILD_SYSTEM_PROMPT = """You are a Valheim blueprint generator. Convert design documents into piece placements.

## Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| generate_floor_grid | Floor coverage | prefab, width, depth, y, origin_x, origin_z |
| generate_floor_walls | ALL 4 walls for a floor | prefab, x_min, x_max, z_min, z_max, base_y, height, filler_prefab, openings |
| generate_wall | Single wall segment | prefab, start_x, start_z, end_x, end_z, base_y, height, rotY, filler_prefab |
| generate_roof | Complete gabled roof | prefab, x_min, x_max, z_min, z_max, base_y, ridge_axis |
| replace_piece | Swap wall piece for door | prefab, x, y, z, rotY, anchor |
| place_piece | Single pieces only | prefab, x, y, z, rotY, snap, anchor_pieces |
| get_prefab_details | Lookup dimensions | prefab_name |
| complete_build | Finalize the build | (no params) |

## Design Doc → Tool Calls

### Simple Rectangular Buildings

**Floor:** bounds x=[-5 to 5], z=[-5 to 5], surface_y=0.5, prefab=stone_floor_2x2
```
generate_floor_grid(prefab="stone_floor_2x2", width=10, depth=10, y=0.5, origin_x=-5, origin_z=-5)
```

**All walls for a floor (one call generates N/E/S/W):**
```
generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                     base_y=0.5, height=6, filler_prefab="stone_wall_2x1")
```

**Walls with door opening on south wall:**
```
generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                     base_y=0.5, height=6, filler_prefab="stone_wall_2x1",
                     openings=[{"wall": "south", "position": 0, "prefab": "stone_arch"}])
```

### Multi-Volume Buildings (when design has COMPOSITION/VOLUMES sections)

When the design specifies multiple volumes with `omit_walls`, use `generate_wall` for individual walls:

**Example: main_hall (omit_walls: east) connected to tower (omit_walls: west)**
```
# main_hall: bounds x=-6 to 6, z=-4 to 4, omit east wall
generate_wall(prefab="wood_wall", start_x=-6, start_z=-4, end_x=-6, end_z=4, base_y=0, height=6, rotY=270)  # west
generate_wall(prefab="wood_wall", start_x=-6, start_z=4, end_x=6, end_z=4, base_y=0, height=6, rotY=0)     # north
generate_wall(prefab="wood_wall", start_x=6, start_z=-4, end_x=-6, end_z=-4, base_y=0, height=6, rotY=180) # south
# east wall omitted - connects to tower

# tower: bounds x=6 to 12, z=-3 to 3, omit west wall
generate_wall(prefab="wood_wall", start_x=6, start_z=3, end_x=12, end_z=3, base_y=0, height=6, rotY=0)     # north
generate_wall(prefab="wood_wall", start_x=12, start_z=3, end_x=12, end_z=-3, base_y=0, height=6, rotY=90)  # east
generate_wall(prefab="wood_wall", start_x=12, start_z=-3, end_x=6, end_z=-3, base_y=0, height=6, rotY=180) # south
# west wall omitted - connects to main_hall
```

**Roof:** Complete gabled roof in ONE call. base_y = top of walls. Slopes meet at peak (no ridge cap needed).
```
generate_roof(prefab="wood_roof", x_min=-5, x_max=5, z_min=-5, z_max=5,
              base_y=6.5, ridge_axis="x")  # "x" = ridge runs E-W, "z" = ridge runs N-S
```

**Doors/Windows in multi-volume buildings:** When design says "openings: south = wood_door", use replace_piece after generating walls. The tool finds the closest wall piece and places the door at that piece's position.

For a door on a wall, use the wall's CENTER coordinates (midpoint of x_min to x_max for north/south walls, midpoint of z_min to z_max for east/west walls):
```
# Volume bounds: x=2 to 10, z=2 to 8, opening: south = wood_door
# South wall is at z=2, center X = (2+10)/2 = 6
generate_wall(prefab="woodwall", start_x=10, start_z=2, end_x=2, end_z=2, base_y=0.5, height=6, rotY=180)
# Replace wall piece near center with door
replace_piece(prefab="wood_door", x=6, y=1.5, z=2, rotY=180, anchor="bottom")
```

The replace_piece tool will find the closest wall piece to (6, 2) and place the door at THAT piece's exact position.

## Workflow

1. Read the COMPOSITION/VOLUMES sections to understand the building structure
2. For each volume, generate floor and walls (skipping omit_walls)
3. Generate roofs for each volume or spanning multiple volumes
4. Each tool returns a summary: {"added": N, "total_pieces": M}
5. When ALL pieces are placed, call complete_build() to finalize
6. Do NOT output JSON manually - the system accumulates pieces automatically

## Coordinate System

- Y = UP, X/Z = horizontal, units = meters, position = piece CENTER
- rotY: North (+Z) = 0, East (+X) = 90, South (-Z) = 180, West (-X) = 270

## Rules

1. Use EXACT prefab names from design document
2. Walls must be height ≥ 6 meters (wall prefabs are ~2m, tool stacks automatically)
3. rotY must be 0, 90, 180, or 270
4. Use filler_prefab when design specifies one
5. For simple rectangles: use generate_floor_walls (all 4 walls at once)
6. For multi-volume buildings: use generate_wall for individual walls, skip omit_walls
7. Second floor: surface_y = floor1_surface_y + wall_height + floor_thickness
8. ALWAYS call complete_build() when done - do not output JSON
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

    # Extract building name from design doc (first # heading or fallback).
    building_name = "Generated Building"
    for line in design_doc.split("\n"):
        if line.startswith("# "):
            building_name = line[2:].strip()
            break
    
    user_message = f"""Convert this design document into piece placements:

{design_doc}

Remember to:
1. Use generate_floor_walls to create ALL 4 walls per floor in ONE call
2. Use height=6 or more for walls - this is CRITICAL for proper building scale
3. Call complete_build() when all pieces are placed"""

    messages = [{"role": "user", "content": user_message}]
    
    # Combine prefab detail lookup with placement tools.
    all_tools = BUILD_TOOLS + PLACEMENT_TOOLS
    
    # Server-side piece accumulator - pieces are collected here, not in Claude's output.
    accumulated_pieces: list[dict] = []
    
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
            max_tokens=16384,  # Larger for potentially many pieces.
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
            build_complete = False
            
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
                    placement_tool_names = {tool["name"] for tool in PLACEMENT_TOOLS}
                    if block.name in placement_tool_names:
                        result = execute_placement_tool(block.name, block.input, accumulated_pieces)
                        
                        # Check if this was complete_build
                        if block.name == "complete_build":
                            build_complete = True
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
            
            # If complete_build was called, return accumulated pieces directly.
            if build_complete:
                if verbose:
                    print(f"[Build Agent] Build complete with {len(accumulated_pieces)} pieces")
                
                # Add final exchange to messages for logging.
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})
                
                blueprint = {"name": building_name, "pieces": accumulated_pieces}
                
                return AgentResult(
                    result=blueprint,
                    tool_calls=tool_call_log,
                    conversation=messages,
                    api_calls=api_call_count,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_read_tokens=total_cache_read,
                    cache_write_tokens=total_cache_write
                )
            
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
            # Claude stopped without calling complete_build.
            # Use accumulated pieces if available, otherwise try to parse output.
            if accumulated_pieces:
                if verbose:
                    print(f"[Build Agent] Using {len(accumulated_pieces)} accumulated pieces (no complete_build call)")
                blueprint = {"name": building_name, "pieces": accumulated_pieces}
            else:
                # Fallback: try to extract JSON from response (legacy behavior).
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
                conversation=messages,
                api_calls=api_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_write_tokens=total_cache_write
            )
