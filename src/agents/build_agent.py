"""
Build Agent - Stage 2 of the blueprint pipeline.

This agent takes a design document and produces the actual blueprint JSON
with exact piece positions and rotations.
"""

import json
import re
from pathlib import Path
import anthropic

from src.agents.design_agent import AgentResult
from src.tools.prefab_lookup import BUILD_TOOLS, execute_tool as execute_prefab_tool
from src.tools.placement_tools import PLACEMENT_TOOLS, execute_placement_tool
from src.tools.blueprint_parser import parse_blueprint


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
| generate_floor_walls | ALL 4 walls for a floor | prefab, x_min, x_max, z_min, z_max, base_y, height, filler_prefab, openings |
| generate_roof | Complete gabled roof | prefab, ridge_prefab, x_min, x_max, z_min, z_max, base_y, ridge_axis |
| place_piece | Single pieces only | prefab, x, y, z, rotY, snap, anchor_pieces |
| get_prefab_details | Lookup dimensions | prefab_name |

## Design Doc → Tool Calls

**Floor:** bounds x=[-5 to 5], z=[-5 to 5], surface_y=0.5, prefab=stone_floor_2x2
```
floor = generate_floor_grid(prefab="stone_floor_2x2", width=10, depth=10, y=0.5, origin_x=-5, origin_z=-5)
```

**All walls for a floor (one call generates N/E/S/W):**
```
walls = generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                             base_y=0.5, height=6, filler_prefab="stone_wall_2x1")
```

**Walls with door opening on south wall:**
```
walls = generate_floor_walls(prefab="stone_wall_4x2", x_min=-5, x_max=5, z_min=-5, z_max=5,
                             base_y=0.5, height=6, filler_prefab="stone_wall_2x1",
                             openings=[{"wall": "south", "position": 0, "prefab": "stone_arch"}])
```

**Roof:** Complete gabled roof in ONE call. base_y = top of walls.
```
roof = generate_roof(prefab="wood_roof", ridge_prefab="wood_roof_top",
                     x_min=-5, x_max=5, z_min=-5, z_max=5, base_y=6.5,
                     ridge_axis="x")  # "x" = ridge runs E-W, "z" = ridge runs N-S
```

For a 3-floor building: 3 generate_floor_grid calls + 3 generate_floor_walls calls + 1 generate_roof call.

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
# Example Context Loading
# ============================================================================

# Track loading status for diagnostics
_EXAMPLE_LOAD_ERROR: str | None = None


def _load_example_blueprints() -> str:
    """
    Load example blueprint as JSON to show the LLM the expected output format.

    Returns JSON-formatted example blueprint, or empty string if unavailable.
    """
    global _EXAMPLE_LOAD_ERROR

    # Look for example blueprints relative to the project root
    project_root = Path(__file__).parent.parent.parent
    example_path = project_root / "blueprints" / "simple-starter-house.blueprint"

    if not example_path.exists():
        _EXAMPLE_LOAD_ERROR = f"Example blueprint not found: {example_path}"
        return ""

    try:
        parsed = parse_blueprint(str(example_path))

        # Convert to the JSON format the LLM should output
        blueprint_json = {
            "name": parsed.name,
            "pieces": [
                {
                    "prefab": p.prefab,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "z": round(p.z, 2),
                    "rotY": round(p.rotY)
                }
                for p in parsed.pieces
            ]
        }

        json_str = json.dumps(blueprint_json, indent=2)
        _EXAMPLE_LOAD_ERROR = None  # Success
        return f"\n\n## Example Output\n\nHere is an example of a complete blueprint JSON output:\n\n```json\n{json_str}\n```"
    except Exception as e:
        _EXAMPLE_LOAD_ERROR = f"Failed to load example blueprint: {type(e).__name__}: {e}"
        return ""


# Load example context once at module import time
_EXAMPLE_CONTEXT = _load_example_blueprints()

# Report loading status at import time
if _EXAMPLE_LOAD_ERROR:
    import sys
    print(f"[Build Agent Warning] {_EXAMPLE_LOAD_ERROR}", file=sys.stderr)


# ============================================================================
# Agent Execution
# ============================================================================

def run_build_agent(
    design_doc: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False,
    use_examples: bool = True
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

    # Log example context status
    if verbose:
        if use_examples and _EXAMPLE_CONTEXT:
            print(f"[Build Agent] Using example context ({len(_EXAMPLE_CONTEXT)} chars)")
        elif use_examples and not _EXAMPLE_CONTEXT:
            print(f"[Build Agent] Examples enabled but no context loaded")
            if _EXAMPLE_LOAD_ERROR:
                print(f"[Build Agent] Load error: {_EXAMPLE_LOAD_ERROR}")
        else:
            print(f"[Build Agent] Examples disabled")

    user_message = f"""Convert this design document into a blueprint JSON:

{design_doc}

Remember to:
1. Use generate_floor_walls to create ALL 4 walls per floor in ONE call
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
        # Combine system prompt with example context if enabled
        full_system_prompt = BUILD_SYSTEM_PROMPT + (_EXAMPLE_CONTEXT if use_examples else "")
        
        response = client.messages.create(
            model=model,
            max_tokens=16384,  # Larger for potentially many pieces.
            system=[
                {
                    "type": "text",
                    "text": full_system_prompt,
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
                    placement_tool_names = {tool["name"] for tool in PLACEMENT_TOOLS}
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
                conversation=messages,
                api_calls=api_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_write_tokens=total_cache_write
            )
