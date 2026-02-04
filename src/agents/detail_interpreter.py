"""
Detail Interpreter - Stage 3b of the blueprint pipeline.

Converts natural language enhancement descriptions from the Detail Agent
into concrete piece placements using the placement tools.

Uses a hybrid approach:
1. Pattern matcher for common descriptions (corner poles, wall beams, etc.)
2. LLM fallback for creative/novel descriptions

The pattern matcher is fast and reliable for known patterns.
The LLM fallback handles edge cases with full tool access.
"""

import json
import re
from typing import Literal

import anthropic

from src.agents.design_agent import AgentResult
from src.tools.placement_tools import (
    place_piece,
    generate_corner_poles,
    generate_wall_beams,
    generate_interior_beams,
    PLACEMENT_TOOLS,
    execute_placement_tool,
)
from src.tools.prefab_lookup import get_prefabs


# ============================================================================
# Pattern Matchers
# ============================================================================

def _extract_prefab(text: str, material: str, category: str) -> str | None:
    """
    Extract a prefab name from text, or find a matching one.
    
    First looks for explicit prefab names in parentheses like (wood_pole2).
    Falls back to finding a prefab of the right material and category.
    """
    # Look for explicit prefab name in parentheses
    paren_match = re.search(r'\(([a-z_0-9]+)\)', text.lower())
    if paren_match:
        return paren_match.group(1)
    
    # Look for explicit prefab name mentioned directly
    prefabs = get_prefabs(material=material, category=category)
    for p in prefabs:
        if p["name"].lower() in text.lower():
            return p["name"]
    
    # Fall back to first matching prefab
    if prefabs:
        return prefabs[0]["name"]
    
    return None


def _extract_height_range(text: str, analysis: dict) -> tuple[float, float]:
    """
    Extract height range from text like "from floor (y=0) to wall top (y=6)".
    
    Returns (base_y, top_y).
    """
    # Look for explicit y= values
    y_matches = re.findall(r'y\s*=\s*(\d+(?:\.\d+)?)', text)
    if len(y_matches) >= 2:
        ys = [float(y) for y in y_matches]
        return min(ys), max(ys)
    
    # Look for keywords
    base_y = analysis["floors"][0]["y"] if analysis["floors"] else 0
    top_y = analysis["wall_top_y"]
    
    text_lower = text.lower()
    
    if "floor" in text_lower:
        base_y = analysis["floors"][0]["y"] if analysis["floors"] else 0
    
    if "wall top" in text_lower or "ceiling" in text_lower:
        top_y = analysis["wall_top_y"]
    elif "roof" in text_lower:
        top_y = analysis["roof_base_y"]
    
    return base_y, top_y


def _extract_walls(text: str) -> list[Literal["north", "south", "east", "west"]]:
    """Extract which walls are referenced in the text."""
    walls: list[Literal["north", "south", "east", "west"]] = []
    text_lower = text.lower()
    
    if "all" in text_lower or "every" in text_lower:
        return ["north", "south", "east", "west"]
    
    if "north" in text_lower:
        walls.append("north")
    if "south" in text_lower:
        walls.append("south")
    if "east" in text_lower:
        walls.append("east")
    if "west" in text_lower:
        walls.append("west")
    
    # Default to all if none specified
    return walls if walls else ["north", "south", "east", "west"]


def _extract_axis(text: str) -> Literal["x", "z"]:
    """Extract axis from text like 'running east-west' or 'north-south'."""
    text_lower = text.lower()
    
    if "east-west" in text_lower or "east west" in text_lower:
        return "x"
    if "north-south" in text_lower or "north south" in text_lower:
        return "z"
    
    # Default based on other keywords
    if "east" in text_lower or "west" in text_lower:
        return "x"
    if "north" in text_lower or "south" in text_lower:
        return "z"
    
    return "x"  # Default


def _extract_spacing(text: str) -> float | None:
    """Extract spacing value from text like 'spaced 4m apart'."""
    match = re.search(r'spaced?\s+(\d+(?:\.\d+)?)\s*m', text.lower())
    if match:
        return float(match.group(1))
    return None


# ============================================================================
# Pattern Handlers
# ============================================================================

def _handle_corner_poles(desc: str, analysis: dict) -> list[dict] | None:
    """Handle descriptions about corner poles."""
    keywords = ["corner pole", "corner post", "corner pillar", "corners"]
    if not any(kw in desc.lower() for kw in keywords):
        return None
    
    # Find appropriate prefab
    prefab = _extract_prefab(desc, analysis["material"], "pole")
    if not prefab:
        return None
    
    # Get height range
    base_y, top_y = _extract_height_range(desc, analysis)
    height = top_y - base_y
    
    bounds = analysis["bounds"]
    
    pieces = generate_corner_poles(
        prefab=prefab,
        x_min=bounds["x_min"],
        x_max=bounds["x_max"],
        z_min=bounds["z_min"],
        z_max=bounds["z_max"],
        base_y=base_y,
        height=height
    )
    
    return pieces


def _handle_wall_beams(desc: str, analysis: dict) -> list[dict] | None:
    """Handle descriptions about wall beams / horizontal beams along walls."""
    keywords = ["wall beam", "horizontal beam", "beam along", "beams along"]
    if not any(kw in desc.lower() for kw in keywords):
        return None
    
    # Find appropriate prefab
    prefab = _extract_prefab(desc, analysis["material"], "beam")
    if not prefab:
        return None
    
    # Get Y position (default to wall top)
    _, top_y = _extract_height_range(desc, analysis)
    
    bounds = analysis["bounds"]
    
    pieces = generate_wall_beams(
        prefab=prefab,
        x_min=bounds["x_min"],
        x_max=bounds["x_max"],
        z_min=bounds["z_min"],
        z_max=bounds["z_max"],
        y=top_y
    )
    
    return pieces


def _handle_interior_beams(desc: str, analysis: dict) -> list[dict] | None:
    """Handle descriptions about interior beams / rafters."""
    keywords = ["interior beam", "interior rafter", "rafter", "ceiling beam", 
                "spanning", "across the ceiling", "running"]
    if not any(kw in desc.lower() for kw in keywords):
        return None
    
    # Find appropriate prefab
    prefab = _extract_prefab(desc, analysis["material"], "beam")
    if not prefab:
        return None
    
    # Get Y position
    _, top_y = _extract_height_range(desc, analysis)
    
    # Get axis
    axis = _extract_axis(desc)
    
    # Get spacing
    spacing = _extract_spacing(desc)
    
    bounds = analysis["bounds"]
    
    pieces = generate_interior_beams(
        prefab=prefab,
        x_min=bounds["x_min"],
        x_max=bounds["x_max"],
        z_min=bounds["z_min"],
        z_max=bounds["z_max"],
        y=top_y,
        axis=axis,
        spacing=spacing
    )
    
    return pieces


# All pattern handlers in order of specificity
PATTERN_HANDLERS = [
    _handle_corner_poles,
    _handle_wall_beams,
    _handle_interior_beams,
]


# ============================================================================
# LLM Fallback Interpreter
# ============================================================================

LLM_INTERPRETER_SYSTEM = """You are a Valheim building piece placement assistant. 
Convert natural language descriptions into tool calls that place building pieces.

You have access to placement tools that handle the coordinate math for you.
Just call the appropriate tool with the right parameters based on the description.

## Building Analysis
{building_analysis}

## Available Prefabs
{prefab_list}

## Guidelines
- Use generate_corner_poles for corner posts/pillars
- Use generate_wall_beams for horizontal beams along walls
- Use generate_interior_beams for rafters/ceiling beams
- Use place_piece for individual decorative pieces

Call the appropriate tool(s) to place the described pieces.
"""


def _run_llm_interpreter(
    description: str,
    analysis: dict,
    model: str,
    verbose: bool
) -> tuple[list[dict], int, int]:
    """
    Use LLM with tools to interpret a description.
    
    Returns (pieces, input_tokens, output_tokens).
    """
    client = anthropic.Anthropic()
    
    # Format building analysis
    from src.tools.building_analyzer import format_building_analysis
    analysis_text = format_building_analysis(analysis)
    
    # Get prefab list
    material = analysis["material"]
    prefabs = get_prefabs(material=material, category=["beam", "pole"])
    prefab_list = "\n".join(f"- {p['name']}" for p in prefabs)
    
    system_prompt = LLM_INTERPRETER_SYSTEM.format(
        building_analysis=analysis_text,
        prefab_list=prefab_list
    )
    
    # Subset of tools for detail placement
    detail_tools = [
        t for t in PLACEMENT_TOOLS 
        if t["name"] in ["place_piece", "generate_corner_poles", 
                         "generate_wall_beams", "generate_interior_beams"]
    ]
    
    messages = [{"role": "user", "content": f"Place pieces for: {description}"}]
    
    total_input = 0
    total_output = 0
    pieces: list[dict] = []
    
    # Tool loop
    for _ in range(5):  # Max iterations
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=detail_tools,
            messages=messages
        )
        
        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens
        
        # Check if done
        if response.stop_reason == "end_turn":
            break
        
        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"[LLM Interpreter] Tool: {block.name}({block.input})")
                
                # Execute the tool
                result_json = execute_placement_tool(block.name, block.input)
                result = json.loads(result_json)
                
                # Accumulate pieces
                if isinstance(result, list):
                    pieces.extend(result)
                elif isinstance(result, dict) and not result.get("error"):
                    pieces.append(result)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json
                })
        
        if not tool_results:
            break
        
        # Add assistant response and tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    
    return pieces, total_input, total_output


# ============================================================================
# Main Interpreter
# ============================================================================

def run_detail_interpreter(
    descriptions: list[str],
    building_analysis: dict,
    base_pieces: list[dict],
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False
) -> AgentResult:
    """
    Convert natural language descriptions to piece placements.
    
    Uses pattern matching for common cases, falls back to LLM for others.
    
    Args:
        descriptions: List of enhancement descriptions from detail agent
        building_analysis: Analyzed building structure
        base_pieces: Existing pieces from build agent
        model: Claude model for LLM fallback
        verbose: Whether to print debug info
    
    Returns:
        AgentResult with result being a blueprint dict containing
        base_pieces + newly placed detail pieces.
    """
    all_pieces = list(base_pieces)
    tool_calls = []
    total_input = 0
    total_output = 0
    api_calls = 0
    
    for desc in descriptions:
        if verbose:
            print(f"[Interpreter] Processing: {desc}")
        
        # Try pattern matchers first
        matched = False
        for handler in PATTERN_HANDLERS:
            pieces = handler(desc, building_analysis)
            if pieces is not None:
                if verbose:
                    print(f"[Interpreter] Pattern matched: {handler.__name__}, {len(pieces)} pieces")
                all_pieces.extend(pieces)
                tool_calls.append(f"pattern:{handler.__name__}({len(pieces)} pieces)")
                matched = True
                break
        
        # Fall back to LLM if no pattern matched
        if not matched:
            if verbose:
                print(f"[Interpreter] No pattern match, using LLM fallback")
            
            pieces, inp_tok, out_tok = _run_llm_interpreter(
                desc, building_analysis, model, verbose
            )
            
            if pieces:
                all_pieces.extend(pieces)
                tool_calls.append(f"llm_fallback({len(pieces)} pieces)")
            else:
                tool_calls.append(f"llm_fallback(0 pieces - unresolved)")
            
            total_input += inp_tok
            total_output += out_tok
            api_calls += 1
    
    if verbose:
        added = len(all_pieces) - len(base_pieces)
        print(f"[Interpreter] Total: {len(all_pieces)} pieces ({added:+d} added)")
    
    blueprint = {
        "name": "Detailed Building",
        "pieces": all_pieces
    }
    
    return AgentResult(
        result=blueprint,
        tool_calls=tool_calls,
        conversation=[],  # Not tracking full conversation for interpreter
        api_calls=api_calls,
        input_tokens=total_input,
        output_tokens=total_output,
        cache_read_tokens=0,
        cache_write_tokens=0
    )
