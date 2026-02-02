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

BUILD_SYSTEM_PROMPT = """You are a Valheim blueprint generator. Convert design documents into JSON piece arrays.

## Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| generate_floor_grid | Floor coverage | prefab, width, depth, y, origin_x, origin_z |
| generate_wall | Walls (auto-stacks rows) | prefab, start_x/z, end_x/z, base_y, height, rotY, filler_prefab, anchor_pieces |
| generate_roof_slope | Roof rows | prefab, start_x, start_z, y, count, direction, rotY, anchor_pieces |
| place_piece | Single pieces only | prefab, x, y, z, rotY, snap, anchor_pieces |
| get_prefab_details | Lookup dimensions | prefab_name |

Pass anchor_pieces to snap new structures to existing ones. Tools handle internal snapping automatically.

## Design Doc → Tool Calls

**Floor:** bounds x=[-4 to 4], z=[-4 to 4], surface_y=0.5, prefab=stone_floor_2x2
```
floor = generate_floor_grid(prefab="stone_floor_2x2", width=8, depth=8, y=0.5, origin_x=-4, origin_z=-4)
```

**Wall:** North z=4, x=[-4 to 4], prefab=stone_wall_4x2, filler=stone_wall_1x1, height=6
```
walls = generate_wall(prefab="stone_wall_4x2", start_x=-4, start_z=4, end_x=4, end_z=4,
                      base_y=0.5, height=6, rotY=0, filler_prefab="stone_wall_1x1", anchor_pieces=floor)
```

**Wall with opening:** South z=-4, opening=stone_arch at x=0
```
left_wall = generate_wall(prefab="stone_wall_4x2", start_x=-4, start_z=-4, end_x=-1, end_z=-4, ...)
arch = place_piece(prefab="stone_arch", x=0, z=-4, ...)
right_wall = generate_wall(prefab="stone_wall_4x2", start_x=1, start_z=-4, end_x=4, end_z=-4, ...)
```

**Roof:** base_y=6.5 (top of walls)
```
roof = generate_roof_slope(prefab="wood_roof_45", start_x=0, start_z=0, y=6.5, count=4,
                           direction="east", rotY=0, anchor_pieces=walls)
```

## Output Format

Return ONLY valid JSON (no markdown):
```
{"name": "Building Name", "pieces": [{"prefab": "stone_floor_2x2", "x": 1.0, "y": 0.5, "z": 1.0, "rotY": 0}, ...]}
```

## Coordinate System

- Y = UP, X/Z = horizontal, units = meters, position = piece CENTER
- rotY: North (+Z) = 0, East (+X) = 90, South (-Z) = 180, West (-X) = 270

## Rules

1. Use EXACT prefab names from design document
2. Walls must be height ≥ 6 meters (wall prefabs are ~2m, tool stacks automatically)
3. rotY must be 0, 90, 180, or 270
4. Use filler_prefab when design specifies one
5. Use composite tools for structures, place_piece only for doors/stairs/decorations
6. Second floor: surface_y = floor1_surface_y + wall_height + floor_thickness
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
1. Use the procedural placement tools (generate_floor_grid, generate_wall, etc.)
2. Use height=6 or more for walls - this is CRITICAL for proper building scale
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
                        "generate_wall", 
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
