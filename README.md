# GHOST: Guarded Hybrid Obfuscation with Split Token Execution

Reference implementation and experiment suite for the manuscript
*GHOST: Guarded Hybrid Obfuscation with Split Token Execution for Secure Edge Inference*
(submitted to IEEE Transactions on Information Forensics and Security). The manuscript source is in `paper/`;
the earlier implementation that this code replaces is kept in `archive/v1/`.

This code replaces the previous GHOST implementation. Every number, table and data
figure in the TIFS manuscript (`ghost_tifs.tex`) is produced by the scripts below and read
from `results/*.json` by `make_tables.py`. No number in the manuscript is typed by hand.

## What changed from the previous implementation

| Previous code | GHOST v2 |
|---|---|
| Fixed spatial permutations hard-coded in `models.py` (seed 1337 + layer) and stored in the checkpoint; the network is trained on scrambled activations (accuracy 86% -> 61%) | Exact obfuscation applied after training: two-sided mixing `W~ = P A [W; F] (B^-1 (x) I)` with decoy filters `F`; secrets live only in the TEE object; predictions identical to the original model |
| Decoy path = one extra plain convolution selected with `key=0` | Decoy filters mixed into every stored filter (`decoy=0.25` of the output channels) |
| Token gate = `if` statement returning zeros | HMAC-SHA-256 token bound to (counter, SHA-256 of the input), replay protection, query budget, checked inside the TEE before any operand is released; authorisation covers one request |
| No edge/cloud split in the model; AES-GCM only in evaluation scripts | Split hook encrypts the split activation with AES-GCM inside the TEE; suffix executes in the cloud |
| MemGuard / Purifier / ModelGuard "conceptual" stand-ins | Removed. Output-side protection is an explicit release policy (probabilities / label + rounded confidence / label only) |
| Extraction "10.00%" from the input-scrambling bug | Knockoff-style extraction with a ResNet-20 surrogate against the exact model, as a function of the TEE budget |

## Files

- `common.py` data splits (40k members / 10k adversary pool / 10k test), ResNet-20 and VGG-11 (half width), training loops.
- `ghost.py` the construction: BN folding, `TEE` (secrets, masks, HMAC tokens, budget), `ObfConv` (levels `full`, `left`, `ps`), `GhostDeployment` (edge prefix, split, AES-GCM, timers, REE view recorder).
- `matching.py` adaptive weight-recovery attacks: direction matching, least-squares un-mixing (permutation and bilinear), row-space projection.
- `train_victim.py ARCH EPOCHS SEED` trains a victim on the member set.
- `finetune_public.py` builds the public-model scenario victim (seed-1 model fine-tuned on the members).
- `exact_overhead.py ARCH` exactness for rho in {0,1,10,100,1e3,1e4} and per-request cost for three split points.
- `steal.py ARCH N [conds]` model stealing from device memory (see the docstring for the condition names).
- `inversion.py ARCH` input reconstruction from the operands released by the TEE, plus the mean-image baseline.
- `extract_mia.py ARCH [mia|extract]` query-based extraction against the budget and membership inference under release policies.
- `token_tests.py` behaviour of the token gate.
- `make_tables.py OUTDIR` writes `results_macros.tex`, `tab_*.tex`, `fig_*.tex`, `inv_rho*.pdf` for the manuscript.
- `run_all.sh` the complete CPU pipeline in the order used for the manuscript.

## Data

`data/c10_train.npz`, `data/c10_test.npz`, `data/c100_test.npz` are produced by `python prep_data.py`, which downloads
CIFAR-10 and CIFAR-100 with torchvision (or reads HuggingFace parquet files from a directory given as its argument).
The trained victims used in the manuscript are in `checkpoints/` (ResNet-20 seeds 0 and 1, the fine-tuned ResNet-20, VGG-11 seed 0),
so the attack experiments can be run without retraining.

## Extending to GPU-scale backbones

1. Add the architecture to `ARCHS` in `common.py` (ResNet-18/50 for CIFAR).
2. Add its edge-convolution naming to `edge_conv_names` and `prefix_modules` (the ResNet-20 logic applies to any
   BasicBlock/Bottleneck ResNet with the module names `conv1`, `blocks.k.conv{1,2}`, `blocks.k.short.0`).
3. Depthwise convolutions (MobileNet) cannot use input mixing across channels: use `level='left'` with a
   channel permutation shared between input and output, and report it as such.
4. Run the same scripts with `NT` set to the number of CPU threads (or move tensors to CUDA in `common.py`),
   then rerun `make_tables.py`. New rows appear in the tables automatically once their JSON files exist.

## Reproducing the manuscript

```bash
python prep_data.py
./run_all.sh                       # about 9 hours on 2 CPU threads
python make_tables.py paper/generated
python fig_tensors.py                  # tiles for Figure 1 (paper/figures/arch_assets)
cd paper && pdflatex ghost_tifs.tex && pdflatex ghost_tifs.tex
```
