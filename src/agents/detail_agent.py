"""
Detail Agent - Stage 3 of the blueprint pipeline.

This agent takes the original prompt and the accumulated pieces from the build agent,
then enhances the building with structural architectural details (beams, poles, windows).
"""

import json

import anthropic

from src.agents.design_agent import AgentResult
from src.tools.prefab_lookup import (
    DESIGN_TOOLS,
    execute_tool as execute_prefab_tool,
)
from src.tools.placement_tools import execute_placement_tool


# ============================================================================
# Tool Definitions for Detail Agent
# ============================================================================

# Detail agent gets:
# - get_prefabs (from DESIGN_TOOLS) - to discover available detail pieces
# - get_prefab_details (from BUILD_TOOLS) - for dimensions/snap points
# - place_piece, remove_piece, complete_build (subset of placement tools)

_PLACE_PIECE_TOOL = {
    "name": "place_piece",
    "description": """Place a single piece at exact coordinates.

Use for individual detail pieces: beams, poles, windows, trim pieces.
Set snap=true to auto-align with nearby pieces.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "prefab": {
                "type": "string",
                "description": "Prefab name (e.g., 'wood_beam', 'wood_pole2')"
            },
            "x": {"type": "number", "description": "X position (meters)"},
            "y": {"type": "number", "description": "Y position (meters)"},
            "z": {"type": "number", "description": "Z position (meters)"},
            "rotY": {
                "type": "integer",
                "enum": [0, 90, 180, 270],
                "description": "Rotation: 0=North, 90=East, 180=South, 270=West"
            },
            "snap": {
                "type": "boolean",
                "description": "If true, snap to nearby pieces (default false)"
            }
        },
        "required": ["prefab", "x", "y", "z", "rotY"]
    }
}

_REMOVE_PIECE_TOOL = {
    "name": "remove_piece",
    "description": """Remove a piece by its index in the pieces list.

Use to create gaps for windows or fix placement errors.
After removal, subsequent piece indices shift down by 1.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index of the piece to remove (0-based)"
            }
        },
        "required": ["index"]
    }
}

_COMPLETE_BUILD_TOOL = {
    "name": "complete_build",
    "description": "Signal that detailing is complete. Call when all details have been added.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Combine tools for the detail agent
# Only get_prefabs from DESIGN_TOOLS (skip list_materials, list_categories)
_GET_PREFABS_TOOL = next(t for t in DESIGN_TOOLS if t["name"] == "get_prefabs")

DETAIL_TOOLS = [
    _GET_PREFABS_TOOL,
    _PLACE_PIECE_TOOL,
    _REMOVE_PIECE_TOOL,
    _COMPLETE_BUILD_TOOL,
]


# ============================================================================
# System Prompt
# ============================================================================

DETAIL_SYSTEM_PROMPT = """You are a Valheim architectural detail specialist. Your job is to enhance buildings with structural details.

## Your Role

You receive a building with basic structure (floors, walls, roof) already placed. Add architectural details to make it more visually interesting and structurally authentic.

## Tools

| Tool | Purpose |
|------|---------|
| get_prefabs | Find available detail pieces by material/category (batch all categories in ONE call) |
| place_piece | Add a single detail piece at exact coordinates |
| remove_piece | Remove a piece by index (for creating window gaps) |
| complete_build | Finalize when done |

IMPORTANT: Call get_prefabs ONCE with all needed categories as an array:
```
get_prefabs(material="wood", category=["beam", "pole", "window"])
```
Do NOT make separate calls for each category.

## What to Add

Focus on STRUCTURAL details only:
- **Beams**: Horizontal members along wall tops, under floors, roof supports
- **Poles/Pillars**: Vertical supports at corners, doorways, load-bearing points
- **Cross-braces**: Diagonal supports in wall corners or under overhangs
- **Window gaps**: Remove wall segments and add window frames where appropriate
- **Trim pieces**: Half-walls or quarter pieces to fill gaps or add visual interest

Do NOT add furniture, lighting, or decorations - that's a separate agent's job.

## Coordinate System

- Y = UP, X/Z = horizontal
- rotY: 0 = North (+Z), 90 = East (+X), 180 = South (-Z), 270 = West (-X)
- Positions are piece centers

## Workflow

1. Review the existing pieces to understand the building layout
2. Call get_prefabs ONCE with all needed categories (beam, pole, window, etc.)
3. Place beams along wall tops (y = wall_base + wall_height)
4. Add corner poles where walls meet
5. Consider adding windows by removing wall pieces and placing window frames
6. Call complete_build when done

## Rules

1. Match materials - use wood beams for wood buildings, stone for stone, etc.
2. Don't duplicate existing pieces
3. Keep details proportional to the building scale
4. Call complete_build() when finished
"""


# ============================================================================
# Agent Execution
# ============================================================================

def run_detail_agent(
    prompt: str,
    base_pieces: list[dict],
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False
) -> AgentResult:
    """
    Run the detail agent to enhance a building with architectural details.
    
    Args:
        prompt: Original user building description (for style context)
        base_pieces: Pieces from the build agent to enhance
        model: Claude model to use
        verbose: Whether to print debug info
    
    Returns an AgentResult with the enhanced blueprint dict and usage stats.
    """
    client = anthropic.Anthropic()
    
    # Extract building name from first piece's context or use default
    building_name = "Detailed Building"
    
    # Format pieces summary for the prompt
    pieces_json = json.dumps(base_pieces, indent=2)
    
    user_message = f"""Enhance this building with structural details.

## Original Design Request
{prompt}

## Current Pieces ({len(base_pieces)} total)
```json
{pieces_json}
```

Add beams, poles, and other structural details appropriate for this building style.
Call complete_build() when done."""

    messages = [{"role": "user", "content": user_message}]
    
    # Server-side piece accumulator - starts with base pieces
    accumulated_pieces: list[dict] = list(base_pieces)  # Copy to avoid modifying original
    
    # Track tool calls and usage for logging
    tool_call_log: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_write = 0
    api_call_count = 0
    
    # Track consecutive identical errors to detect infinite loops
    last_error = None
    consecutive_error_count = 0
    MAX_CONSECUTIVE_ERRORS = 3
    
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=16384,
            system=[
                {
                    "type": "text",
                    "text": DETAIL_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=DETAIL_TOOLS,
            messages=messages
        )
        
        # Track usage from this API call
        api_call_count += 1
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_read += getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        total_cache_write += getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        
        if verbose:
            print(f"[Detail Agent] Stop reason: {response.stop_reason}")
        
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
                    # Log this tool call
                    tool_call_str = f"{block.name}({block.input})"
                    tool_call_log.append(tool_call_str)
                    
                    if verbose:
                        print(f"[Detail Agent] Tool call: {tool_call_str}")
                    
                    # Dispatch to the appropriate tool executor
                    if block.name in ("place_piece", "remove_piece", "complete_build"):
                        result = execute_placement_tool(
                            block.name, block.input, accumulated_pieces
                        )
                        if block.name == "complete_build":
                            build_complete = True
                    else:
                        # Prefab lookup tools
                        result = execute_prefab_tool(block.name, block.input)
                    
                    # Check if result contains an error
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
            
            # If complete_build was called, return accumulated pieces
            if build_complete:
                if verbose:
                    pieces_added = len(accumulated_pieces) - len(base_pieces)
                    print(f"[Detail Agent] Complete with {len(accumulated_pieces)} pieces ({pieces_added:+d} from base)")
                
                # Add final exchange to messages for logging
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
            
            # Track consecutive identical errors
            if had_error_this_round:
                if current_error == last_error:
                    consecutive_error_count += 1
                else:
                    consecutive_error_count = 1
                    last_error = current_error
                
                if consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"Detail agent stuck in error loop. "
                        f"Same error occurred {MAX_CONSECUTIVE_ERRORS} times: {last_error}"
                    )
            else:
                # Reset on successful tool calls
                consecutive_error_count = 0
                last_error = None
            
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Claude stopped without calling complete_build
            # Use accumulated pieces
            if verbose:
                pieces_added = len(accumulated_pieces) - len(base_pieces)
                print(f"[Detail Agent] Using {len(accumulated_pieces)} pieces ({pieces_added:+d}, no complete_build)")
            
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
