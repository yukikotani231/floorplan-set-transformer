# FloorPlan Set Transformer

Set Transformer for Panoptic Symbol Spotting on FloorPlanCAD dataset.

## Overview

This project implements a Set Transformer-based approach for panoptic symbol spotting on CAD floor plans. Unlike traditional approaches that rely on graph neural networks or require explicit graph construction, this method treats CAD primitives (lines, arcs, curves) as an unordered set and uses Set Transformer's permutation-invariant architecture to process them.

### Key Features

- **Permutation-Invariant**: No need for ordering or graph construction
- **Primitive-Level Processing**: Each CAD primitive (line, arc, curve) is treated as a single element
- **Panoptic Segmentation**: Joint semantic classification and instance segmentation
- **Dynamic Batching**: Efficient GPU memory usage for variable-length inputs
- **Mixed Precision Training**: AMP support for faster training

## Architecture

```
Input: N CAD primitives (N × d_input)
    ↓
Input Embedding (Linear → d_model)
    ↓
Set Transformer Encoder
  - ISAB × L layers (with inducing points for O(NM) complexity)
  - Permutation-equivariant self-attention
    ↓
Element-wise Prediction Heads
  - Semantic classification (36 classes)
  - Instance embedding (for clustering)
    ↓
Output: Per-primitive predictions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yukikotani/floorplan-set-transformer.git
cd floorplan-set-transformer

# Install dependencies with uv
uv sync

# Install dev dependencies
uv sync --dev
```

## Dataset

This project uses the [FloorPlanCAD](https://floorplancad.github.io/) dataset.

1. Download the dataset from the official website
2. Place the SVG files in `data/floorplancad/`

## Usage

### Training

```bash
# Train with default config
uv run python scripts/train.py

# Train with synthetic data (for testing)
uv run python scripts/train.py --synthetic

# Disable mixed precision (for debugging)
uv run python scripts/train.py --no-amp

# Disable torch.compile
uv run python scripts/train.py --no-compile
```

### Evaluation

```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

## Configuration

Edit `configs/default.yaml` to customize:

- Model architecture (d_model, num_heads, num_layers, etc.)
- Training parameters (learning rate, batch size, epochs)
- Data processing options

## Development

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/
```

## Project Structure

```
floorplan-set-transformer/
├── configs/
│   └── default.yaml          # Training configuration
├── scripts/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset
│   │   ├── preprocess.py     # SVG parsing
│   │   ├── features.py       # Feature extraction
│   │   └── sampler.py        # Dynamic batch sampler
│   ├── models/
│   │   ├── modules.py        # MAB, SAB, ISAB, PMA
│   │   ├── set_transformer.py # Set Transformer encoder
│   │   └── panoptic_head.py  # Prediction heads
│   ├── training/
│   │   ├── train.py          # Trainer class
│   │   ├── loss.py           # Loss functions
│   │   └── metrics.py        # PQ, IoU metrics
│   └── utils/
│       └── visualization.py  # Plotting utilities
└── tests/
    └── test_model.py         # Model tests
```

## References

- [Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://arxiv.org/abs/1810.00825)
- [FloorPlanCAD: A Large-Scale CAD Drawing Dataset for Panoptic Symbol Spotting](https://arxiv.org/abs/2105.07147)

## License

MIT License
