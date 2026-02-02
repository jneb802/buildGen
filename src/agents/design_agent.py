"""
Design Agent - Stage 1 of the blueprint pipeline.

This agent takes a user's building description and produces a structured design
document. It uses prefab lookup tools to find appropriate pieces and calculates
positioning based on snap point spacing.
"""

from dataclasses import dataclass, field
import anthropic

from src.tools.prefab_lookup import PREFAB_TOOLS, execute_tool


@dataclass
class AgentResult:
    """Result from an agent run, including logging and usage info."""
    result: str | dict
    tool_calls: list[str] = field(default_factory=list)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# ============================================================================
# System Prompt
# ============================================================================

DESIGN_SYSTEM_PROMPT = """You are an expert Valheim building architect. Your job is to create
detailed design documents for buildings based on user descriptions.

## Your Task

Given a building request, produce a structured markdown design document that specifies:
1. What prefabs to use (query them using the available tools)
2. Overall dimensions and layout constraints
3. Build tool parameters for each structural element

## Key Principle: Specify Constraints, Not Coordinates

Do NOT calculate piece positions or counts. Instead, specify:
- Boundaries (x_min, x_max, z_min, z_max)
- Floor surface Y values
- Which prefabs to use (primary + filler)

The build agent has procedural tools that calculate positions automatically.

## Output Format

Start your response DIRECTLY with the markdown document. Do not include any preamble.

```markdown
# [BUILDING NAME] DESIGN DOCUMENT

## OVERVIEW
- Building name: [name]
- Footprint: [X]m x [Z]m (width x depth)
- Height: [Y]m (to roof peak)
- Primary materials: [list materials]
- Number of floors: [count]

## PREFABS TO USE
- Floors: [prefab name] (primary), [prefab name] (filler if needed)
- Walls: [prefab name] (primary), [prefab name] (filler for gaps)
- Roof: [prefab name], [ridge prefab], [corner prefab]
- Doors/Arches: [prefab name]
- Stairs: [prefab name]

## BUILDING BOUNDS
Define the building envelope (all coordinates relative to center at origin):
- X range: [x_min] to [x_max]
- Z range: [z_min] to [z_max]
- Ground floor Y: [value] (floor surface level)

## FLOORS
### Ground Floor
- surface_y: [value]
- tool: generate_floor_grid
- prefab: [name]
- bounds: x=[x_min to x_max], z=[z_min to z_max]

### Floor 2 (if applicable)
- surface_y: [value]
- tool: generate_floor_grid
- prefab: [name]
- bounds: x=[x_min to x_max], z=[z_min to z_max]

## WALLS
### Ground Floor Walls (surface_y = [value])
- North: z=[z_max], x=[x_min to x_max], prefab=[name], filler=[name]
- East: x=[x_max], z=[z_min to z_max], prefab=[name], filler=[name]
- South: z=[z_min], x=[x_min to x_max], prefab=[name], filler=[name], opening=[door prefab] at x=0
- West: x=[x_min], z=[z_min to z_max], prefab=[name], filler=[name]

### Floor 2 Walls (surface_y = [value])
[Same format]

## ROOF
- style: [26 or 45] degree
- base_y: [value] (where roof starts, typically top of highest wall)
- ridge_direction: [X or Z] axis
- prefab: [name]
- ridge_prefab: [name] (if needed)
- corner_prefab: [name] (if needed)

## STAIRS
- location: [corner/side description]
- prefab: [name]
- floor_1_to_2: position near ([x], [z])
- floor_2_to_3: position near ([x], [z]) (if applicable)

## CONSTRUCTION SEQUENCE
1. Foundation: [description]
2. Ground floor walls: [description]
3. [Continue in logical build order...]
```

## Build Tools (format output for these)

The build agent uses these procedural tools. Your design should map cleanly to them:

- generate_floor_grid(prefab, width, depth, y, origin_x, origin_z)
  Creates a complete floor from bounds. Calculates piece count automatically.

- generate_wall_line(prefab, start_x, start_z, end_x, end_z, y, rotY, filler_prefab)
  Fills a wall line with pieces. Handles gaps with filler automatically.
  rotY: 0=north-facing, 90=east-facing, 180=south-facing, 270=west-facing

- generate_roof_slope(prefab, start_x, start_z, y, count, direction, rotY)
  Creates a row of roof pieces.

- place_piece(prefab, x, y, z, rotY)
  For individual pieces like doors, stairs, decorations.

## Critical Rules

1. ALWAYS query prefabs using tools before specifying them
2. Use EXACT prefab names from the database
3. Specify floor SURFACE y values only (where things sit on top)
   - The build agent calculates piece center Y from surface Y
4. Always specify a filler_prefab for walls to handle partial coverage
5. Wall rotY must face OUTWARD from building center
6. Do NOT calculate individual piece positions—that's the build agent's job

## Available Prefab Tools

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
) -> AgentResult:
    """
    Run the design agent to generate a design document from a user prompt.
    
    Handles the tool use loop - Claude may call tools multiple times to
    research prefabs before generating the final design.
    
    Returns an AgentResult with the design document and usage stats.
    """
    client = anthropic.Anthropic()
    
    messages = [{"role": "user", "content": prompt}]
    
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
    
    # Keep looping until Claude produces a final text response without tool calls.
    while True:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": DESIGN_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=PREFAB_TOOLS,
            messages=messages
        )
        
        # Track usage from this API call.
        api_call_count += 1
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cache_read += getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        total_cache_write += getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
        
        if verbose:
            print(f"[Design Agent] Stop reason: {response.stop_reason}")
        
        # Check if Claude wants to use tools.
        if response.stop_reason == "tool_use":
            # Process all tool calls in this response.
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
                        print(f"[Design Agent] Tool call: {tool_call_str}")
                    
                    result = execute_tool(block.name, block.input)
                    
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
                        f"Design agent stuck in error loop. "
                        f"Same error occurred {MAX_CONSECUTIVE_ERRORS} times: {last_error}"
                    )
            else:
                # Reset on successful tool calls.
                consecutive_error_count = 0
                last_error = None
            
            # Add assistant's response and tool results to conversation.
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # No more tool calls - extract the final text response.
            design_doc = ""
            for block in response.content:
                if block.type == "text":
                    design_doc = block.text
                    break
            
            return AgentResult(
                result=design_doc,
                tool_calls=tool_call_log,
                api_calls=api_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_write_tokens=total_cache_write
            )
