# ProteinGym submission drafts — ESMC-TabICL (supervised DMS substitutions)

## Pull Request

**Title:** Add ESMC-TabICL supervised baseline (DMS substitutions)

**Body:**

This PR adds **ESMC-TabICL**, a supervised variant-effect predictor for the DMS substitution
benchmark: mean-pooled **ESMC-600M** embeddings plus two zero-shot scores, fed to a **TabICLv2**
tabular foundation-model regressor under the standard 5-fold CV protocol. No GP, MSA, or structure.

Features per variant (1164 dims): ESMC-600M second-to-last-layer embedding mean-pooled over residues
(1152); an approximate (single unmasked pass) pseudo-log-likelihood (1); and the masked wt-marginal
mutation-effect score aggregated as deciles (11).

Changes:
- `proteingym/baselines/esmc_tabicl/` — self-contained scoring script (`compute_scores.py`), README,
  `requirements.txt`, and `score_all.sh`.
- `config.json` — new `ESMC-TabICL` entry under `model_list_supervised_substitutions_DMS`.
- `proteingym/constants.json` — clean name, references, model details, model type.
- `benchmarks/DMS_supervised/substitutions/{Spearman,MSE}/` — Summary + DMS-level files regenerated
  with ESMC-TabICL included (computed with `performance_DMS_supervised_benchmarks.py`).

Per-mutant score files for all 217 assays × 3 CV splits are available at: **[Zenodo/Drive link —
TODO add]** (layout `<cv_scheme>/esmc_tabicl/<DMS_id>.csv`, columns `mutant,y,y_pred,fold`).

Both ESMC-600M (`biohub/ESMC-600M`) and TabICLv2 are open source. The method scores all mutants in
the substitution benchmark (217/217 assays).

**Official performance (your `performance_DMS_supervised_benchmarks.py`):** Average Spearman **0.630**
(random 0.749, modulo 0.591, contiguous 0.550) → **rank #2**, behind Kermut (0.657) and ahead of
ProteinNPT (0.619). ESMC-TabICL has the **best random-split score on the board (0.749 > Kermut 0.745)**
and leads the **Stability** category (0.821 vs 0.817).

---

## Issue (label: `new model`)

**Title:** [new model] ESMC-TabICL — supervised DMS substitutions

**Body:**

Requesting inclusion of **ESMC-TabICL** in the supervised DMS substitution benchmark. PR: #<TODO>.

**Method:** ESMC-600M (`biohub/ESMC-600M`) embeddings (second-to-last layer, mean-pooled) +
approximate pseudo-LL + masked wt-marginal deciles → TabICLv2 regression, standard 5-fold CV.

**Reproduce:**
1. `pip install torch "esm@git+https://github.com/Biohub/esm.git@main" tabicl scipy pandas numpy`
   (`biohub/ESMC-600M` auto-downloads from HuggingFace; TabICLv2 checkpoint auto-downloads).
2. `cd proteingym/baselines/esmc_tabicl && DMS_FOLDER=<cv_folds_singles_substitutions> ./score_all.sh`
3. Merge + score with `proteingym/merge_supervised.py` and `proteingym/performance_DMS_supervised_benchmarks.py`.

**Open source:** ESMC-600M — MIT/Biohub; TabICLv2 — open source (soda-inria/tabicl). **Scores all
mutants:** 217/217 substitution assays, all variants.

**Performance (official script):** Avg Spearman 0.630 (random 0.749 / modulo 0.591 / contiguous 0.550).
Per category: Activity 0.564, Binding 0.567, Expression 0.640, OrganismalFitness 0.559, Stability 0.821.

**Score files:** [Zenodo/Drive link — TODO].
