"""
Command-line interface for Valheim blueprint generation.

Usage:
    valheim-blueprint "a small stone watchtower with two floors"
    valheim-blueprint "viking longhouse" --output ./builds --verbose
"""

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from src.pipeline import run_pipeline


# Load environment variables from .env file (for ANTHROPIC_API_KEY).
load_dotenv()


console = Console()


@click.command()
@click.argument("prompt")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Output directory for generated files."
)
@click.option(
    "--model", "-m",
    default="claude-sonnet-4-20250514",
    help="Claude model to use."
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed progress and debug info."
)
def main(prompt: str, output: Path, model: str, verbose: bool):
    """
    Generate a Valheim blueprint from a natural language description.
    
    PROMPT: Description of the building to generate (e.g., "a small stone tower")
    """
    console.print(f"[bold]Valheim Blueprint Generator[/bold]")
    console.print(f"Prompt: {prompt}")
    console.print(f"Model: {model}")
    console.print()
    
    try:
        run_pipeline(prompt, output, model=model, verbose=verbose)
    except Exception as e:
        console.print(f"[red bold]Error:[/red bold] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
