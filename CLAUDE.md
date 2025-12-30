# Claude Code Instructions

This file contains project-specific instructions for Claude Code.

## Project Overview

FloorPlan Set Transformer - A Set Transformer implementation for panoptic symbol spotting on CAD floor plans (FloorPlanCAD dataset).

## Development Commands

```bash
# Format
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/

# Test
uv run pytest tests/

# Run all checks
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest tests/
```

## Training

```bash
# Full training with all optimizations
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py

# With synthetic data (for testing)
uv run python scripts/train.py --synthetic

# Disable AMP for debugging
uv run python scripts/train.py --no-amp
```

## Architecture Notes

- **Input**: CAD primitives (lines, arcs, curves) as unordered sets
- **Model**: Set Transformer with ISAB for O(NM) complexity
- **Output**: Semantic labels (36 classes) + instance embeddings
- **Training**: Dynamic batching based on element count for memory efficiency

## Key Files

- `scripts/train.py` - Main training script
- `scripts/evaluate.py` - Evaluation and visualization
- `src/models/` - Set Transformer implementation
- `src/data/` - Dataset and preprocessing
- `src/training/` - Training loop, losses, metrics
- `configs/default.yaml` - Hyperparameters

## Data

FloorPlanCAD dataset should be placed in `data/floorplancad/`. The dataset is not included in the repository.
