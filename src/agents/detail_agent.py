"""
Detail Agent - Stage 3 of the blueprint pipeline.

This agent takes the original prompt and the accumulated pieces from the build agent,
then enhances the building with structural architectural details (beams, poles, windows).

This version uses NO TOOLS - the agent directly outputs JSON pieces to add.
"""

import json
import re

import anthropic

from src.agents.design_agent import AgentResult
from src.tools.prefab_lookup import get_prefabs


# ============================================================================
# System Prompt (No Tools Version)
# ============================================================================

DETAIL_SYSTEM_PROMPT_TEMPLATE = """You are a Valheim architectural detail specialist. Your job is to enhance buildings with structural details.

## Your Role

You receive a building with basic structure (floors, walls, roof) already placed. Add architectural details to make it more visually interesting and structurally authentic.

## Output Format

Output ONLY a JSON array of additional pieces to add. Each piece needs:
- prefab: Exact prefab name (see available prefabs below)
- x, y, z: Position in meters
- rotY: Rotation (0, 90, 180, or 270)

Example output:
```json
[
  {{"prefab": "wood_pole2", "x": -6, "y": 1, "z": -8, "rotY": 0}},
  {{"prefab": "wood_pole2", "x": 6, "y": 1, "z": -8, "rotY": 0}},
  {{"prefab": "wood_beam", "x": -4, "y": 6, "z": -8, "rotY": 0}},
  {{"prefab": "wood_beam", "x": 0, "y": 6, "z": -8, "rotY": 0}}
]
```

## Available Detail Prefabs

{prefab_list}

## Positioning Rules

- Y = UP, X/Z = horizontal plane
- Positions are piece CENTERS
- For poles: place center at (base_y + height/2) so they sit on the floor
- For beams: place at wall top height
- rotY: 0=North(+Z), 90=East(+X), 180=South(-Z), 270=West(-X)
- Beams are ~2m long, poles are ~2m tall

## What to Add

Focus on STRUCTURAL details only:
- **Corner poles**: At building corners, from floor to ceiling
- **Wall beams**: Along wall tops
- **Interior rafters**: Spanning across ceiling

Do NOT add furniture, lighting, or decorations.

## Extracting Bounds from Existing Pieces

Look at the pieces to find:
- min/max X and Z from floor pieces = building bounds
- Y values from floor pieces = floor level
- Wall pieces typically stack to 6m height

## Rules

1. Match materials (wood beams for wood buildings)
2. Use EXACT prefab names from the list above
3. Output ONLY the JSON array, no other text
"""


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from text that may contain markdown or other content."""
    # Try to find JSON in code blocks first
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(code_block_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to find a JSON array directly
    start = text.find("[")
    if start == -1:
        return []
    
    # Find matching closing bracket
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return []
    
    return []


def _get_material_from_pieces(pieces: list[dict]) -> str:
    """Detect the primary material from existing pieces."""
    prefab_names = []
    for p in pieces:
        if isinstance(p, dict):
            prefab_names.append(p.get("prefab", ""))
    
    # Count material patterns
    materials = {
        "wood": 0,
        "stone": 0,
        "darkwood": 0,
        "blackmarble": 0,
    }
    
    for name in prefab_names:
        if not isinstance(name, str):
            continue
        if "darkwood" in name:
            materials["darkwood"] += 1
        elif "wood" in name:
            materials["wood"] += 1
        elif "stone" in name:
            materials["stone"] += 1
        elif "blackmarble" in name:
            materials["blackmarble"] += 1
    
    # Return most common
    return max(materials, key=materials.get) if any(materials.values()) else "wood"


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
    
    This version uses NO TOOLS - the agent outputs JSON pieces directly.
    
    Args:
        prompt: Original user building description (for style context)
        base_pieces: Pieces from the build agent to enhance
        model: Claude model to use
        verbose: Whether to print debug info
    
    Returns an AgentResult with the enhanced blueprint dict and usage stats.
    """
    client = anthropic.Anthropic()
    
    building_name = "Detailed Building"
    
    # Detect material and get available prefabs
    try:
        material = _get_material_from_pieces(base_pieces)
    except Exception as e:
        raise RuntimeError(f"Failed to detect material: {e}") from e
    
    try:
        prefab_list = _format_prefab_list(material)
    except Exception as e:
        raise RuntimeError(f"Failed to format prefab list for {material}: {e}") from e
    
    # Build system prompt with prefab list
    system_prompt = DETAIL_SYSTEM_PROMPT_TEMPLATE.format(prefab_list=prefab_list)
    
    # Format pieces for the prompt
    pieces_json = json.dumps(base_pieces, indent=2)
    
    user_message = f"""Enhance this building with structural details.

## Original Design Request
{prompt}

## Current Pieces ({len(base_pieces)} total)
```json
{pieces_json}
```

Output a JSON array of additional pieces to add (beams, poles, etc.).
Use EXACT prefab names from the available list."""

    messages = [{"role": "user", "content": user_message}]
    
    # Single API call - no tool loop needed
    response = client.messages.create(
        model=model,
        max_tokens=16384,
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
        print(f"[Detail Agent] Response length: {len(response_text)} chars")
        print(f"[Detail Agent] Response preview: {response_text[:500]}...")
    
    # Parse the JSON array of new pieces
    new_pieces = _extract_json_array(response_text)
    
    if verbose:
        print(f"[Detail Agent] Parsed {len(new_pieces)} new pieces")
    
    # Validate and add new pieces
    valid_pieces = []
    for p in new_pieces:
        try:
            if not isinstance(p, dict):
                continue
            if not all(k in p for k in ("prefab", "x", "y", "z", "rotY")):
                continue
            valid_pieces.append({
                "prefab": str(p["prefab"]),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "z": float(p["z"]),
                "rotY": int(p["rotY"])
            })
        except (TypeError, ValueError, KeyError):
            # Skip malformed pieces
            continue
    
    # Combine base pieces with new pieces
    all_pieces = list(base_pieces) + valid_pieces
    
    if verbose:
        print(f"[Detail Agent] Total pieces: {len(all_pieces)} ({len(valid_pieces):+d} added)")
    
    # Add response to messages for logging
    messages.append({"role": "assistant", "content": response_text})
    
    blueprint = {"name": building_name, "pieces": all_pieces}
    
    return AgentResult(
        result=blueprint,
        tool_calls=[f"direct_json_output({len(valid_pieces)} pieces)"],
        conversation=messages,
        api_calls=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write
    )
