#!/usr/bin/env bash
# Final recipe: ESMC-600M (second-to-last layer, mean-pooled) + approximate pseudo-LL
# + masked wt-marginal deciles  ->  TabICLv2 regression, on Kermut's ProteinGym
# supervised substitution benchmark (217 single-mutant assays, 3 CV splits).
#
# Features (per variant, concatenated -> 1164 dims):
#   emb/                (1152) ESMC-600M hidden_states[-2] (layer 35), mean-pooled over residues
#   pll/                (1)    approximate pseudo-log-likelihood, 1 unmasked forward pass
#   wtmarg_masked_dec/  (11)   masked-marginal mutation-effect score, decile-aggregated
#
# Two GPUs assumed (shard 0 -> gpu0, shard 1 -> gpu1). Run from this directory.
set -e
PY=./esmc_venv/bin/python

# 1. Embeddings: ESMC-600M layer-35 (second-to-last), mean-pooled, fp32. -> emb/
$PY scripts/embed.py --gpu 0 --shard 0 --nshards 2 &
$PY scripts/embed.py --gpu 1 --shard 1 --nshards 2 &
wait

# 2. Approximate pseudo-LL (one unmasked forward pass per variant). -> pll/
$PY scripts/compute_pll.py --gpu 0 --shard 0 --nshards 2 &
$PY scripts/compute_pll.py --gpu 1 --shard 1 --nshards 2 &
wait

# 3. Masked wt-marginal mutation-effect score, decile-aggregated (11 cols). -> wtmarg_masked_dec/
$PY scripts/compute_wtmarg.py --gpu 0 --shard 0 --nshards 2 --mode masked --agg deciles &
$PY scripts/compute_wtmarg.py --gpu 1 --shard 1 --nshards 2 --mode masked --agg deciles &
wait

# 4. TabICLv2 CV regression on [emb || pll || masked-decile]. -> results/embpllmaskdec_spearman_shard*.csv
$PY scripts/regress.py --gpu 0 --shard 0 --nshards 2 \
    --pll-dir pll --wtmarg-dir wtmarg_masked_dec \
    --res-prefix embpllmaskdec_spearman_shard --pred-dir preds_embpllmaskdec &
$PY scripts/regress.py --gpu 1 --shard 1 --nshards 2 \
    --pll-dir pll --wtmarg-dir wtmarg_masked_dec \
    --res-prefix embpllmaskdec_spearman_shard --pred-dir preds_embpllmaskdec &
wait

# 5. Aggregate per split + functional category, vs Kermut.
$PY scripts/aggregate.py
