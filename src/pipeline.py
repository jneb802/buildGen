"""
Pipeline orchestrator for blueprint generation.

Coordinates the three stages:
    Design Agent -> Build Agent -> Snap Validator.
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
from src.agents.design_agent import run_design_agent
from src.agents.build_agent import run_build_agent
from src.tools.snap_validator import validate_and_correct, format_correction_report
from src.tools.blueprint_converter import save_blueprint_file


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
    verbose: bool = False
) -> Path:
    """
    Run the complete blueprint generation pipeline.
    
    Stages:
    1. Design Agent - Creates structured design document from prompt
    2. Build Agent - Converts design to blueprint JSON  
    3. Snap Validator - Corrects positions to align snap points
    
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
    
    try:
        design_doc = run_design_agent(prompt, model=model, verbose=verbose)
        log_lines.append("Stage 1: SUCCESS")
    except Exception as e:
        console.print(f"[red]Stage 1 failed: {e}[/red]")
        log_lines.append(f"Stage 1: FAILED - {e}")
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
    
    try:
        raw_blueprint = run_build_agent(design_doc, model=model, verbose=verbose)
        log_lines.append(f"Stage 2: SUCCESS - {len(raw_blueprint.get('pieces', []))} pieces")
    except Exception as e:
        console.print(f"[red]Stage 2 failed: {e}[/red]")
        log_lines.append(f"Stage 2: FAILED - {e}")
        raw_blueprint = {"name": "Error", "pieces": []}
    
    piece_count = len(raw_blueprint.get("pieces", []))
    console.print(f"[green]✓[/green] Generated {piece_count} pieces")
    
    # ========================================================================
    # Stage 3: Snap Validator
    # ========================================================================
    
    console.print(Panel("Stage 3: Snap Validator", style="bold blue"))
    console.print(f"[dim]Correcting snap point alignment...[/dim]")
    
    try:
        # Convert raw pieces to Piece models.
        raw_pieces = raw_blueprint.get("pieces", [])
        pieces = []
        for p in raw_pieces:
            # Normalize rotY to valid values.
            rot_y = p.get("rotY", 0)
            if rot_y not in (0, 90, 180, 270):
                rot_y = round(rot_y / 90) * 90 % 360
                if rot_y not in (0, 90, 180, 270):
                    rot_y = 0
            
            pieces.append(Piece(
                prefab=p["prefab"],
                x=float(p["x"]),
                y=float(p["y"]),
                z=float(p["z"]),
                rotY=rot_y
            ))
        
        corrected_pieces, corrections = validate_and_correct(pieces)
        correction_count = sum(1 for c in corrections if c.was_corrected)
        
        log_lines.append(f"Stage 3: SUCCESS - {correction_count} pieces corrected")
        
        # Generate correction report.
        report = format_correction_report(corrections)
        if verbose:
            console.print(Panel(report, title="Correction Report"))
        
    except Exception as e:
        console.print(f"[red]Stage 3 failed: {e}[/red]")
        log_lines.append(f"Stage 3: FAILED - {e}")
        corrected_pieces = pieces if 'pieces' in dir() else []
        correction_count = 0
    
    console.print(f"[green]✓[/green] Corrected {correction_count} piece positions")
    
    # ========================================================================
    # Create Final Blueprint
    # ========================================================================
    
    blueprint = Blueprint(
        name=raw_blueprint.get("name", "Generated Blueprint"),
        creator="BlueprintGenerator",
        description=prompt,
        category="Misc",
        pieces=corrected_pieces
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
    
    # Save log if verbose.
    if verbose:
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
