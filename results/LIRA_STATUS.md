# LiRA membership inference: status

`src/ghost/attacks/attack_lira.py` implements offline LiRA (Carlini et al.,
"Membership Inference Attacks From First Principles", IEEE S&P 2022) against
the GHOST target. **No result from this script should be cited or used in the
manuscript yet** -- calibration against the GHOST architecture is unresolved.

## What's been ruled out

A sanity control is built into the script: it scores one trained shadow model
against the other shadows' OUT statistics, using architecture-matched
(shadow-vs-shadow) data with known ground-truth membership. This control has
passed in every run (AUC 0.87-0.92), which means the scoring pipeline itself
(z-score computation, OUT-calibration, AUC direction) is implemented
correctly.

Despite that, four independent full-scale runs (16 shadows each) against the
real GHOST target have all produced an AUC well below 0.5 (0.11-0.33), which
is not a "strong defense" result -- it indicates the score is inverted
relative to true membership, not merely uninformative. Each run fixed a
different, real, individually-validated issue:

1. Device selection silently fell back to CPU instead of MPS.
2. OOD non-member labels (CIFAR-100) indexed out of the target's 10-way
   softmax.
3. Shadow IN/OUT masks were randomly generated, unrelated to which images
   each shadow actually trained on.
4. Shadow models used a different architecture family (plain ResNet-18)
   than the GHOST target (permutations + decoys + token gates), which
   confounds LiRA's assumption that shadows approximate "the target without
   this example."
5. Shadows were trained for far fewer epochs on far less data than the
   target (regime mismatch).
6. Shadows never randomized the decoy path during training, unlike the
   target's ensemble-style training procedure, so 7 of 8 decoy branches went
   untrained in every shadow.

Fixes 4-6 were each validated at small scale (a handful of shadows, a few
hundred samples) before being run at full scale, and each showed a
correct-direction signal in isolation -- but the full-scale result still
inverted every time. The remaining cause has not been identified.

## What this means for the manuscript

The manuscript's Section IV-C already states: *"Stronger white-box
formulations such as LiRA are not evaluated here and are identified as a
direction for future work."* That statement should remain accurate rather
than being replaced with an unresolved or possibly-inverted number.

## For whoever picks this up next

- Don't trust any single `results/lira_*.json` output without also checking
  `sanity_control_auc` in the same file is clearly above 0.5.
- The small-scale diagnostic pattern used to validate fixes 4-6 (a few
  shadows, ~500 samples, a couple of minutes) is worth reusing and iterating
  on *repeatedly* before committing to another multi-hour full-scale run --
  a single small-scale success was mistaken for a fix once in this process
  and did not hold up at scale.
- Worth checking next: whether `_confidence_logit`'s fixed `key=0` at
  evaluation time (for both target and shadows) interacts badly with a
  target that was trained across all 8 decoy paths versus shadows evaluated
  the same way -- this wasn't isolated as a variable on its own.
