"""
Design Agent - Stage 1 of the blueprint pipeline.

This agent takes a user's building description and produces a structured design
document. It uses prefab lookup tools to find appropriate pieces and calculates
positioning based on snap point spacing.
"""

import anthropic

from src.tools.prefab_lookup import PREFAB_TOOLS, execute_tool


# ============================================================================
# System Prompt
# ============================================================================

DESIGN_SYSTEM_PROMPT = """You are an expert Valheim building architect. Your job is to create
detailed design documents for buildings based on user descriptions.

## Your Task

Given a building request, produce a structured markdown design document that specifies:
1. What prefabs to use (query them using the available tools)
2. Exact dimensions and layout
3. How pieces connect at each floor level

## Output Format

Start your response DIRECTLY with the markdown document. Do not include any preamble or explanation.

Your design document MUST follow this structure:

```markdown
# [BUILDING NAME] DESIGN DOCUMENT

## OVERVIEW
- Building name: [name]
- Overall dimensions: [X]m x [Y]m x [Z]m (width x height x depth)
- Primary materials: [list materials]
- Number of floors: [count]

## PREFABS TO USE
List each prefab by its exact internal name (from the database):
- Floors: [prefab names]
- Walls: [prefab names]  
- Roof: [prefab names]
- Other: [prefab names]

## FOUNDATION
- Floor dimensions: [X]m x [Z]m
- Material: [prefab name]
- Grid layout: [rows] x [columns] pieces
- Position: y = [value] (ground level)

## WALLS
### Floor 1 (y = 0 to y = [height])
For each wall direction:
- North wall: [length]m, [height]m, using [prefab name]
- East wall: ...
- South wall: ... (note any door openings)
- West wall: ...

### Floor 2 (if applicable)
[Same format, with correct y positions]

## ROOF
- Style: [26 degree / 45 degree]
- Material: [prefab name]
- Ridge direction: [along X / along Z]
- Starting y position: [value]

## STAIRS (if multi-floor)
- Location: [description]
- Prefab: [name]
- Connects: floor [N] to floor [M]

## CONSTRUCTION SEQUENCE
Specify the build order to ensure structural stability and accessibility:
1. [Phase name]: [what to build and why this order]
2. [Phase name]: ...
...

Example phases (adapt to your design):
1. Foundation: All floor pieces (required for structural support)
2. Exterior walls: Ground floor outer walls (establishes footprint)
3. Interior walls: Ground floor partitions and door frames
4. Upper floors: Second floor structure (needs ground floor support)
5. Roof frame: Ridge and support beams
6. Roof covering: Thatch/wood tiles (weather protection)
7. Stairs: Vertical connections between floors
8. Furnishings: Interior items (placed last to avoid blocking access)
```

## Critical Rules

1. ALWAYS query prefabs using the tools before specifying them
2. Use EXACT prefab names from the database (e.g., "stone_wall_4x2" not "stone wall")
3. Calculate y positions based on piece heights and snap point spacing:
   - For a piece with 2m vertical snap spacing: row_y = (row_number × 2) + 1
   - Example: Row 0 = y:1, Row 1 = y:3, Row 2 = y:5
4. Position values are piece CENTER, not corner
5. Be specific about dimensions - the build agent needs exact numbers
6. ALWAYS include a construction sequence - build order matters for:
   - Structural stability (foundations before walls, walls before roof)
   - Accessibility (don't block areas needed for later placements)
   - Efficiency (complete each phase before moving to the next)

## Available Tools

Use these tools to find the right prefabs:
- list_materials(): See available material types
- list_categories(): See available piece categories  
- get_prefabs(material, category): Find prefabs matching filters
- get_prefab_details(name): Get exact dimensions and snap points
"""


# ============================================================================
# Agent Execution
# ============================================================================

def run_design_agent(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = False
) -> str:
    """
    Run the design agent to generate a design document from a user prompt.
    
    Handles the tool use loop - Claude may call tools multiple times to
    research prefabs before generating the final design.
    
    Returns the design document as a markdown string.
    """
    client = anthropic.Anthropic()
    
    messages = [{"role": "user", "content": prompt}]
    
    # Keep looping until Claude produces a final text response without tool calls.
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=DESIGN_SYSTEM_PROMPT,
            tools=PREFAB_TOOLS,
            messages=messages
        )
        
        if verbose:
            print(f"[Design Agent] Stop reason: {response.stop_reason}")
        
        # Check if Claude wants to use tools.
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response.
            tool_results = []
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    if verbose:
                        print(f"[Design Agent] Tool call: {block.name}({block.input})")
                    
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
            
            # Add assistant's response and tool results to conversation.
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # No more tool calls - extract the final text response.
            for block in response.content:
                if block.type == "text":
                    return block.text
            
            # Fallback if no text block found.
            return ""
