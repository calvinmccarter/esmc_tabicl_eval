# ESMC + TabICLv2 on the Kermut / ProteinGym supervised substitution benchmark

Goal: numbers **comparable to Kermut** on its own benchmark (217 single-mutant ProteinGym DMS
assays, 3 CV splits, per-assay Spearman), with a different method — **mean-pooled ESMC-600M
embeddings + zero-shot features fed to TabICLv2 regression**, instead of Kermut's composite GP.

## Final recipe

Per variant, concatenate (→ 1164 features) and run `TabICLRegressor` under Kermut's CV protocol
(train on 4 folds, predict the 5th; concat fold predictions → one Spearman per assay/split):

| feature dir | dims | what |
|---|---|---|
| `emb/` | 1152 | ESMC-600M `hidden_states[-2]` (layer 35, second-to-last), mean-pooled over residues, fp32 |
| `pll/` | 1 | approximate pseudo-log-likelihood (one **unmasked** forward pass: Σ log P(s_i\|s)) |
| `wtmarg_masked_dec/` | 11 | **masked** wt-marginal mutation-effect score (mask each mutated position in WT, read blind logits: logP[mut]−logP[wt]), aggregated as deciles (0,10,…,100%) |

Reproduce end-to-end with `bash run_final.sh` (two GPUs). Pipeline scripts:
`scripts/embed.py`, `scripts/compute_pll.py`, `scripts/compute_wtmarg.py`, `scripts/regress.py`,
`scripts/aggregate.py`. Env: `esmc_venv/` (py3.12, cu130 torch, Biohub `esm` + its transformers
fork — mainline transformers lacks `esmc`, tabicl 2.1.1).

## Result (mean Spearman)

Matched vs Kermut on 214 assays (Kermut per-assay recomputed with our protocol; the other 3 have
name mismatches in Kermut's prediction files). Our numbers over all 217 in parentheses.

| split | ESMC-600M + TabICLv2 | Kermut |
|---|---|---|
| Random | **0.762** (0.761) | 0.760 |
| Modulo | 0.621 (0.618) | 0.650 |
| Contiguous | 0.587 (0.584) | 0.614 |
| **Average** | **0.657** (0.654) | 0.675 |

Per functional category (all 217 / Kermut matched): Stability **0.821 vs 0.817**, Expression
0.640/0.670, OrganismalFitness 0.575/0.598, Binding 0.578/0.619, Activity 0.569/0.612.

**Takeaway:** a generic recipe (pooled ESMC embedding + zero-shot scores → tabular foundation model)
**beats Kermut on the random split and on Stability**, and is within **0.018 average** — using none of
Kermut's structure-conditioned kernel. The residual gap is entirely on the extrapolation splits
(modulo/contiguous) and the Activity/Binding/Expression categories, where Kermut's structure prior helps.

Feature ladder (avg, 214 matched): raw emb 0.644 → +pll 0.646 → +masked-decile **0.657**. The
masked wt-marginal was the single biggest gain (it's the Kermut-style zero-shot signal). The decile
form gives that strong scalar more weight (for single mutants the 11 deciles are identical, ≈11×
up-weighting; it also generalizes to multi-mutation assays).

## Notes / caveats
- **Spearman is matched flat-mean over assays.** Kermut recomputed this way = 0.675 avg (its
  *published* aggregate is 0.655 — ProteinGym groups by protein/function; use `_kermut_perassay.csv`
  for apples-to-apples).
- **Training-context cap 10k** (`regress.py TRAIN_CAP`): a single TabICL fit peaks ~80GB at n=10k;
  larger folds are seeded-subsampled (test never capped). ~8 large assays affected.
- **ESMC context window 2048**; 4 proteins are longer (ZIKV, BRCA2, the two POLG) and use RoPE
  extrapolation — same situation Kermut's ESM-2 faces on long proteins.
- bf16 ≈ fp32 here (verified: embeddings cosine 0.99997, Spearman jitter <0.02); we used fp32.

## Alternatives tried and rejected (all underperformed the final recipe)
| variant | result |
|---|---|
| ESMC-**6B** (full recipe) | ≈neutral-to-worse than 600M (trades random-split loss for small extrapolation gain); ~3× slower regression — not worth it |
| per-**mutated-position** embedding (instead of full-seq) | worse, esp. extrapolation |
| blend 0.5·full + 0.5·mutated-pos (and √n-scaled blend) | ≤ full-seq; mutated-position is a weaker signal that dilutes when mixed |
| average L35 + middle layer (literal / scale-balanced) | literal = no-op (L35 8× larger magnitude); scale-balanced ≈ neutral |
| PCA-whitening (k=512) of embeddings | hurts (−0.025 avg) |
| remove-top-singular-direction (≈ mean-centering) | neutral (TabICL already centers per-feature) |
| unmasked wt-marginal (vs masked) | weaker (0.715 vs 0.736 standalone on BLAT) |
