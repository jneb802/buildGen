"""
Design Agent - Stage 1 of the blueprint pipeline.

This agent takes a user's building description and produces a structured design
document. It uses prefab lookup tools to find appropriate pieces and calculates
positioning based on snap point spacing.
"""

from dataclasses import dataclass, field
import anthropic

from src.tools.prefab_lookup import DESIGN_TOOLS, execute_tool


@dataclass
class AgentResult:
    """Result from an agent run, including logging and usage info."""
    result: str | dict
    tool_calls: list[str] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# ============================================================================
# System Prompt
# ============================================================================

DESIGN_SYSTEM_PROMPT = """You are a Valheim building architect. Create design documents that specify constraints (not coordinates) for the build agent.

## Tools

- list_materials(): Available material types
- list_categories(): Available piece categories
- get_prefabs(material, category): Find prefabs. Both params accept arrays for batching.

Batch queries: call get_prefabs ONCE with all needed categories as an array.

## Design Philosophy

Good buildings combine 2-3 volumes, not just one box:
1. Mix shape types (rectangle, L-shape, octagon, angled-corner)
2. Arrange asymmetrically with offset positions and varying heights
3. Specify where volumes connect (walls omitted at connections)

## Output Format

Output markdown directly (no preamble):

```markdown
# [NAME] DESIGN DOCUMENT

## OVERVIEW
- Total footprint: [X]m x [Z]m, Height: [Y]m
- Materials: [list]

## PREFABS
- Floors: [primary], [filler]
- Walls: [primary], [filler]
- Roof slope: [name] (e.g., wood_roof or wood_roof_45)
- Roof ridge: [name] (e.g., wood_roof_top or wood_roof_top_45)
- Roof corners: ocorner=[name], icorner=[name] (for outer/inner corners)
- Openings: [door/arch]
- Stairs: [prefab]

## COMPOSITION
Describe 2-3 volumes that form the building skeleton:
- [volume_name]: [shape_type] [dimensions] at ([x], [y], [z]), [purpose]
- [volume_name]: [shape_type] [dimensions] at ([x], [y], [z]), connects to [other] on [side]

## VOLUMES

### [volume_name]
[Use one of the shape formats below based on type]

## ROOF
- style: [26/45] degree
- prefabs: slope=[name], ridge=[name], ocorner=[name], icorner=[name]

For each volume with a roof:
### [volume_name]
- bounds: x=[min] to [max], z=[min] to [max]
- base_y: [top of walls]
- ridge_axis: [x | z] (x = ridge runs east-west, z = ridge runs north-south)
- ridge_cap: [yes | no]
- corner_caps: (optional) position: ([x], [z]), type: [ocorner | icorner], rotY: [0/90/180/270]

## STAIRS
- [from]_to_[to]: prefab=[name], position near ([x], [z])
```

## Volume Shape Formats

### Rectangle (default)
```
### [volume_name]
- type: rectangle
- bounds: x=[min] to [max], z=[min] to [max]
- ground_y: [value]
- floors: [count]
- wall_height: [value per floor, typically 6]
- omit_walls: [none | list of sides: north/east/south/west]
- openings: [wall] = [prefab] (e.g., "south = wood_door")
```

### L-Shape / T-Shape
```
### [volume_name]
- type: [L-shape | T-shape]
- bounds_main: x=[min] to [max], z=[min] to [max]
- bounds_wing: x=[min] to [max], z=[min] to [max]
- wing_side: [north | east | south | west]
- ground_y: [value]
- floors: [count]
- wall_height: [value per floor, typically 6]
- omit_walls: [sides where connected to other volumes]
- openings: [wall] = [prefab]
```

### Octagon
```
### [volume_name]
- type: octagon
- center: ([x], [z])
- radius: [value, typically 4-6m for towers]
- ground_y: [value]
- floors: [count]
- wall_height: [value per floor, typically 6]
- openings: [direction] = [prefab]
```

### Angled Corner (rectangle with 45° corner cut)
```
### [volume_name]
- type: angled-corner
- bounds: x=[min] to [max], z=[min] to [max]
- cut_corner: [ne | nw | se | sw]
- ground_y: [value]
- floors: [count]
- wall_height: [value per floor, typically 6]
- omit_walls: [sides where connected to other volumes]
- openings: [wall] = [prefab]
```

## Rules

1. Query prefabs with tools before using; use exact names
2. Define 2-3 volumes in COMPOSITION, even for simple buildings
3. wall_height ≥ 6 meters (critical for interior scale)
4. Always specify filler_prefab for walls
5. Specify omit_walls where volumes connect
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
            tools=DESIGN_TOOLS,
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
                conversation=messages,
                api_calls=api_call_count,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_read_tokens=total_cache_read,
                cache_write_tokens=total_cache_write
            )
