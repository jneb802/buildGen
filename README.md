# Valheim Blueprint Generator

Generate Valheim blueprints from natural language descriptions using Claude AI.

## Installation

```bash
pip install -e .
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Usage

```bash
# Basic usage
valheim-blueprint "a small stone watchtower with two floors"

# With options
valheim-blueprint "viking longhouse" --output ./builds --model claude-sonnet-4-20250514
```

### Arguments

- `prompt` (positional): Building description
- `--output` / `-o`: Output directory (default: `./output`)
- `--model` / `-m`: Claude model (default: `claude-sonnet-4-20250514`)
- `--verbose` / `-v`: Show detailed progress

### Outputs

All outputs go to `{output_dir}/{timestamp}/`:
- `design.md` - Design document from Stage 1
- `blueprint.json` - Final validated blueprint
- `log.txt` - Pipeline execution log (if verbose)

## Pipeline

1. **Design Agent** - Creates structured design document from prompt
2. **Build Agent** - Converts design to blueprint JSON with piece positions (snap correction handled inline by placement tools)
