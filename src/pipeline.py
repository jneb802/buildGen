"""
Pipeline orchestrator for blueprint generation.

Coordinates the two stages:
    Design Agent -> Build Agent.
Handles output file creation and logging.
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.models import Piece, Blueprint
from src.agents.design_agent import run_design_agent, AgentResult
from src.agents.build_agent import run_build_agent
from src.tools.blueprint_converter import save_blueprint_file


def _format_usage_stats(result: AgentResult) -> list[str]:
    """Format usage stats from an AgentResult for logging."""
    lines = []
    lines.append(f"API calls: {result.api_calls}")
    lines.append(f"Tool calls: {len(result.tool_calls)}")
    for tc in result.tool_calls:
        lines.append(f"  - {tc}")
    
    cache_info = ""
    if result.cache_read_tokens > 0:
        cache_info = f" ({result.cache_read_tokens:,} cached)"
    lines.append(f"Tokens: {result.input_tokens:,} input{cache_info} / {result.output_tokens:,} output")
    
    return lines


console = Console()

# Random words for memorable blueprint names.
_RANDOM_WORDS = [
    "amber", "azure", "bolt", "brass", "cedar", "cobalt", "coral", "crimson",
    "dusk", "ember", "falcon", "fern", "frost", "gale", "grove", "haze",
    "iron", "jade", "lunar", "maple", "moss", "nova", "oak", "onyx",
    "peak", "pine", "quartz", "raven", "sage", "slate", "solar", "spark",
    "stone", "storm", "thorn", "tide", "timber", "vale", "wolf", "zinc"
]


def run_pipeline(
    prompt: str,
    output_dir: Path,
    model: str = "claude-sonnet-4-20250514",
    copy_to: Path | None = None,
    verbose: bool = False,
    use_examples: bool = True
) -> Path:
    """
    Run the complete blueprint generation pipeline.
    
    Stages:
    1. Design Agent - Creates structured design document from prompt
    2. Build Agent - Converts design to blueprint JSON (with inline snap correction)
    
    Returns the path to the output directory containing all files.
    """
    # Create timestamped output directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_lines = [f"Pipeline run: {timestamp}", f"Prompt: {prompt}", f"Model: {model}", ""]
    
    # ========================================================================
    # Stage 1: Design Agent
    # ========================================================================
    
    console.print(Panel("Stage 1: Design Agent", style="bold blue"))
    console.print(f"[dim]Generating design document from prompt...[/dim]")
    
    design_result: AgentResult | None = None
    try:
        design_result = run_design_agent(prompt, model=model, verbose=verbose)
        design_doc = design_result.result
        log_lines.append("")
        log_lines.append("=== Stage 1: Design Agent ===")
        log_lines.extend(_format_usage_stats(design_result))
        log_lines.append("Result: SUCCESS")
    except Exception as e:
        console.print(f"[red]Stage 1 failed: {e}[/red]")
        log_lines.append("")
        log_lines.append("=== Stage 1: Design Agent ===")
        log_lines.append(f"Result: FAILED - {e}")
        design_doc = f"# Error\n\nDesign generation failed: {e}"
    
    # Save design document.
    design_path = run_dir / "design.md"
    design_path.write_text(design_doc)
    console.print(f"[green]✓[/green] Saved design to {design_path}")
    
    if verbose:
        console.print(Panel(design_doc[:1000] + "..." if len(design_doc) > 1000 else design_doc, 
                           title="Design Document Preview"))
    
    # ========================================================================
    # Stage 2: Build Agent
    # ========================================================================
    
    console.print(Panel("Stage 2: Build Agent", style="bold blue"))
    console.print(f"[dim]Converting design to blueprint JSON...[/dim]")
    
    build_result: AgentResult | None = None
    try:
        build_result = run_build_agent(design_doc, model=model, verbose=verbose, use_examples=use_examples)
        raw_blueprint = build_result.result
        piece_count = len(raw_blueprint.get("pieces", []))
        log_lines.append("")
        log_lines.append("=== Stage 2: Build Agent ===")
        log_lines.extend(_format_usage_stats(build_result))

        # Check for parse error and log raw response
        if raw_blueprint.get("name") == "Parse Error":
            log_lines.append(f"Result: PARSE ERROR - 0 pieces")
            if "raw_response" in raw_blueprint:
                log_lines.append("")
                log_lines.append("=== Raw LLM Response (Parse Failed) ===")
                log_lines.append(raw_blueprint["raw_response"][:5000])  # Limit to 5k chars
        else:
            log_lines.append(f"Result: SUCCESS - {piece_count} pieces")
    except Exception as e:
        console.print(f"[red]Stage 2 failed: {e}[/red]")
        log_lines.append("")
        log_lines.append("=== Stage 2: Build Agent ===")
        log_lines.append(f"Result: FAILED - {e}")
        raw_blueprint = {"name": "Error", "pieces": []}
    
    piece_count = len(raw_blueprint.get("pieces", []))
    console.print(f"[green]✓[/green] Generated {piece_count} pieces")
    
    # ========================================================================
    # Create Final Blueprint
    # ========================================================================
    
    # Convert raw pieces to Piece models.
    raw_pieces = raw_blueprint.get("pieces", [])
    pieces = []
    for p in raw_pieces:
        # Normalize rotY to valid values.
        rot_y = int(p.get("rotY", 0))
        if rot_y not in (0, 90, 180, 270):
            rot_y = int(round(rot_y / 90) * 90) % 360
            if rot_y not in (0, 90, 180, 270):
                rot_y = 0
        
        pieces.append(Piece(
            prefab=p["prefab"],
            x=float(p["x"]),
            y=float(p["y"]),
            z=float(p["z"]),
            rotY=rot_y
        ))
    
    blueprint = Blueprint(
        name=raw_blueprint.get("name", "Generated Blueprint"),
        creator="BlueprintGenerator",
        description=prompt,
        category="Misc",
        pieces=pieces
    )
    
    # Save blueprint JSON (intermediate format for debugging).
    json_path = run_dir / "blueprint.json"
    blueprint_dict = {
        "name": blueprint.name,
        "creator": blueprint.creator,
        "description": blueprint.description,
        "category": blueprint.category,
        "pieces": [
            {"prefab": p.prefab, "x": p.x, "y": p.y, "z": p.z, "rotY": p.rotY}
            for p in blueprint.pieces
        ]
    }
    json_path.write_text(json.dumps(blueprint_dict, indent=2))
    console.print(f"[green]✓[/green] Saved JSON to {json_path}")
    
    # Save .blueprint file (final Valheim-compatible format).
    random_word = random.choice(_RANDOM_WORDS)
    blueprint_filename = f"{random_word}_{blueprint.name.replace(' ', '_')}_{timestamp}.blueprint"
    blueprint_path = run_dir / blueprint_filename
    save_blueprint_file(blueprint, blueprint_path)
    console.print(f"[green]✓[/green] Saved blueprint to {blueprint_path}")
    
    # Copy to destination if specified.
    if copy_to:
        copy_to.mkdir(parents=True, exist_ok=True)
        dest_path = copy_to / blueprint_filename
        shutil.copy(blueprint_path, dest_path)
        console.print(f"[green]✓[/green] Copied to {dest_path}")
    
    # Add totals to log.
    log_lines.append("")
    log_lines.append("=== Totals ===")
    
    total_api_calls = 0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    
    if design_result:
        total_api_calls += design_result.api_calls
        total_input += design_result.input_tokens
        total_output += design_result.output_tokens
        total_cache_read += design_result.cache_read_tokens
    
    if build_result:
        total_api_calls += build_result.api_calls
        total_input += build_result.input_tokens
        total_output += build_result.output_tokens
        total_cache_read += build_result.cache_read_tokens
    
    log_lines.append(f"API calls: {total_api_calls}")
    log_lines.append(f"Total tokens: {total_input:,} input / {total_output:,} output")
    
    if total_input > 0:
        cache_pct = (total_cache_read / total_input) * 100
        log_lines.append(f"Cache efficiency: {cache_pct:.1f}% of input tokens served from cache")
    
    # Always save log.
    log_path = run_dir / "log.txt"
    log_path.write_text("\n".join(log_lines))
    console.print(f"[green]✓[/green] Saved log to {log_path}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    console.print()
    console.print(Panel(
        f"[bold green]Blueprint generated successfully![/bold green]\n\n"
        f"Name: {blueprint.name}\n"
        f"Pieces: {len(blueprint.pieces)}\n"
        f"Output: {run_dir}",
        title="Complete"
    ))
    
    return run_dir
