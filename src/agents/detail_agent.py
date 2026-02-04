"""
Detail Agent - Stage 3a of the blueprint pipeline.

This agent takes the original prompt and building analysis, then describes
architectural enhancements in natural language. The Detail Interpreter
(Stage 3b) converts these descriptions to concrete piece placements.

This design separates creative intent from coordinate math:
- Detail Agent: "Add corner poles at all four corners from floor to wall top"
- Detail Interpreter: Resolves to actual pieces with correct positions
"""

import anthropic

from src.agents.design_agent import AgentResult
from src.tools.prefab_lookup import get_prefabs
from src.tools.building_analyzer import analyze_building, format_building_analysis


# ============================================================================
# System Prompt
# ============================================================================

DETAIL_SYSTEM_PROMPT_TEMPLATE = """You are a Valheim architectural detail specialist. Your job is to describe enhancements for buildings using natural language.

## Your Role

You receive a building with basic structure (floors, walls, roof) already placed. Describe additional architectural details that would make it more visually interesting and structurally authentic.

## Output Format

Output a list of enhancements, one per line, starting with a dash (-). Be specific about:
- What prefab to use (from the available list)
- Where to place it (reference the building analysis for coordinates)
- How many / what pattern

Example output:
```
- Add corner poles (wood_pole2) at all four corners, stacking from floor (y=0) to wall top (y=6)
- Add horizontal beams (wood_beam) along the north and south walls at wall top height (y=6)
- Add interior rafters (wood_beam) running east-west across the ceiling, spaced 4m apart
- Add decorative X-bracing with wood_pole2 on the south gable between y=6 and y=10
```

## Building Analysis

{building_analysis}

## Available Detail Prefabs

{prefab_list}

## Detail Types You Can Add

**Structural:**
- Corner poles: Vertical supports at building corners
- Wall beams: Horizontal beams along wall tops
- Interior rafters: Beams spanning across the ceiling
- Cross-bracing: Diagonal supports for visual interest

**Decorative:**
- Gable details: X-patterns or sunburst designs on gable ends
- Trim beams: Accent beams along roof edges
- Support columns: Interior pillars

## Guidelines

1. Match materials to the building (use {material} prefabs)
2. Reference specific coordinates from the building analysis
3. Use EXACT prefab names from the available list
4. Describe patterns clearly (e.g., "at all four corners", "spaced 4m apart")
5. Focus on structural authenticity - make it look like it could stand up
6. Don't add furniture, lighting, or decorations - structural details only

## Rules

- Output ONLY the enhancement descriptions (dash-prefixed list)
- Be specific enough that someone could place the pieces
- Reference the building analysis for positions
"""


# ============================================================================
# Helper Functions
# ============================================================================

def _format_prefab_list(material: str) -> str:
    """Get a formatted list of available detail prefabs for a material."""
    prefabs = get_prefabs(material=material, category=["beam", "pole"])
    if not prefabs:
        # Fallback to wood
        prefabs = get_prefabs(material="wood", category=["beam", "pole"])
    
    lines = []
    for p in prefabs:
        try:
            lines.append(f"- {p['name']}: {p['description']} ({p['width']:.1f}x{p['height']:.1f}x{p['depth']:.1f}m)")
        except (KeyError, TypeError):
            continue
    
    return "\n".join(lines) if lines else "No prefabs found"


def _extract_descriptions(text: str) -> list[str]:
    """Extract enhancement descriptions from the response text."""
    descriptions = []
    
    for line in text.split("\n"):
        line = line.strip()
        # Look for lines starting with dash or bullet
        if line.startswith("-") or line.startswith("•"):
            # Remove the leading dash/bullet and whitespace
            desc = line.lstrip("-•").strip()
            if desc:
                descriptions.append(desc)
    
    return descriptions


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
    Run the detail agent to describe architectural enhancements.
    
    This agent outputs natural language descriptions, NOT coordinates.
    The Detail Interpreter (run_detail_interpreter) converts these
    descriptions to actual piece placements.
    
    Args:
        prompt: Original user building description (for style context)
        base_pieces: Pieces from the build agent (for analysis)
        model: Claude model to use
        verbose: Whether to print debug info
    
    Returns:
        AgentResult with result being a dict containing:
        - descriptions: list of natural language enhancement descriptions
        - building_analysis: the analyzed building structure
    """
    client = anthropic.Anthropic()
    
    # Analyze the building structure
    building_analysis = analyze_building(base_pieces)
    material = building_analysis["material"]
    
    if verbose:
        print(f"[Detail Agent] Analyzed building: {material}, bounds={building_analysis['bounds']}")
    
    # Format building analysis for the prompt
    analysis_text = format_building_analysis(building_analysis)
    
    # Get available prefabs for this material
    prefab_list = _format_prefab_list(material)
    
    # Build system prompt
    system_prompt = DETAIL_SYSTEM_PROMPT_TEMPLATE.format(
        building_analysis=analysis_text,
        prefab_list=prefab_list,
        material=material
    )
    
    # User message
    user_message = f"""Describe architectural enhancements for this building.

## Original Design Request
{prompt}

## Building Summary
{len(base_pieces)} pieces placed. Structure analyzed above.

Describe what structural details to add (corner poles, beams, rafters, etc.).
Use the building analysis to reference specific positions."""

    messages = [{"role": "user", "content": user_message}]
    
    # Single API call
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=messages
    )
    
    # Track usage
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
    cache_write = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
    
    # Extract response text
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break
    
    if verbose:
        print(f"[Detail Agent] Response:\n{response_text}")
    
    # Parse descriptions from response
    descriptions = _extract_descriptions(response_text)
    
    if verbose:
        print(f"[Detail Agent] Extracted {len(descriptions)} enhancement descriptions")
        for desc in descriptions:
            print(f"  - {desc}")
    
    # Add response to messages for logging
    messages.append({"role": "assistant", "content": response_text})
    
    # Return descriptions and analysis (not pieces - that's the interpreter's job)
    result = {
        "descriptions": descriptions,
        "building_analysis": building_analysis,
    }
    
    return AgentResult(
        result=result,
        tool_calls=[f"nl_descriptions({len(descriptions)} items)"],
        conversation=messages,
        api_calls=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write
    )
