# GHOST: Guarded Hybrid Obfuscation with Split Token Execution for Secure Edge Inference

This repository is organized for clean, reviewer-facing execution and reproduction.

## Repository structure

- `src/ghost/` — reusable model, defense, and utility implementations.
- `src/ghost/attacks/` — attack suites for extraction, privacy, gradients, and side channels.
- `scripts/` — training and evaluation entry points.
- `configs/` — default experimental configuration.
- `data/` — datasets and generated artifacts.
- `outputs/` — model checkpoints and experiment summaries.
- `logs/` — run logs.
- `archive/` — legacy or superseded scripts kept for reference only.

## Main entry points

- `python scripts/train.py` — train the GHOST and baseline models.
- `python scripts/run_experiments.py` — run the full evaluation suite.
- `python scripts/run_attack_suite.py` — run the privacy and extraction attack evaluation.
- `python scripts/run_model_inversion.py` — run the model inversion experiment.

## Quick start

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the models:
   ```bash
   python scripts/train.py
   ```
3. Run the benchmark suite:
   ```bash
   python scripts/run_experiments.py
   ```
4. For model inversion analysis:
   ```bash
   python scripts/run_model_inversion.py
   ```

## Hardware notes

- CPU: Intel Core i9-12900H
- GPU: NVIDIA GeForce RTX 3070 Ti
- Python: 3.9+ recommended
- Random seed: 42

## Notes for reviewers

- The active experiment pipeline is under `scripts/`.
- The reusable model implementation lives in `src/ghost/`.
- Legacy or non-primary utilities are stored in `archive/` for transparency and reproducibility.
