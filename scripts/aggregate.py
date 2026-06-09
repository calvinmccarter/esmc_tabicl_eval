"""Aggregate the final recipe's per-(assay,scheme) Spearman into the summary table,
matched against Kermut (per-assay, recomputed with our protocol).

Final recipe results:  results/embpllmaskdec_spearman_shard*.csv
Kermut baseline:       results/_kermut_perassay.csv  (concat-folds Spearman per assay)
Functional categories: PRIMO/data/DMS_overview.tsv (coarse_selection_type)

Writes results/_embpllmaskdec_merged_dedup.csv and results/FINAL_summary.csv.
"""
import glob, os
import numpy as np
import pandas as pd

RES_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/results"
OVERVIEW = "/home/ssm-user/sandbox/pfnworld/PRIMO/data/DMS_overview.tsv"
SCHEMES = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
LABEL = {"fold_random_5": "Random", "fold_modulo_5": "Modulo", "fold_contiguous_5": "Contiguous"}
CATS = ["Activity", "Binding", "Expression", "OrganismalFitness", "Stability"]


def main():
    parts = glob.glob(os.path.join(RES_DIR, "embpllmaskdec_spearman_shard*.csv"))
    md = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["assay", "scheme"], keep="last")
    md = md.dropna(subset=["spearman"])
    md.to_csv(os.path.join(RES_DIR, "_embpllmaskdec_merged_dedup.csv"), index=False)
    ker = pd.read_csv(os.path.join(RES_DIR, "_kermut_perassay.csv"))
    common = sorted(set(md.assay) & set(ker.assay))   # 214 (Kermut matches 214/217 by name)

    print(f"FINAL recipe: ESMC-600M L35 mean-pool + approx-pll + masked-wt-marginal deciles")
    print(f"217 assays computed; matched vs Kermut on {len(common)}.\n")
    print("=== Mean Spearman per CV split ===")
    rows = []
    for s in SCHEMES:
        a = md[(md.scheme == s) & md.assay.isin(common)].spearman.mean()
        k = ker[(ker.scheme == s) & ker.assay.isin(common)].spearman.mean()
        a217 = md[md.scheme == s].spearman.mean()
        rows.append((LABEL[s], a, k, a217))
        print(f"  {LABEL[s]:11s} ours={a:.3f}  Kermut={k:.3f}   (ours over all 217: {a217:.3f})")
    avg = (LABEL.get('avg', 'Average'), np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows]),
           np.mean([r[3] for r in rows]))
    print(f"  {'Average':11s} ours={avg[1]:.3f}  Kermut={avg[2]:.3f}   (ours over all 217: {avg[3]:.3f})")
    pd.DataFrame([dict(split=r[0], ours=r[1], kermut=r[2], ours_all217=r[3]) for r in rows] +
                 [dict(split="Average", ours=avg[1], kermut=avg[2], ours_all217=avg[3])]
                 ).to_csv(os.path.join(RES_DIR, "FINAL_summary.csv"), index=False)

    ov = pd.read_csv(OVERVIEW, sep="\t")[["DMS_id", "coarse_selection_type"]]
    mc = md.merge(ov, left_on="assay", right_on="DMS_id"); mc["cat"] = mc.coarse_selection_type.replace({"ExpressionGFP": "Expression"})
    kc = ker.merge(ov, left_on="assay", right_on="DMS_id"); kc["cat"] = kc.coarse_selection_type.replace({"ExpressionGFP": "Expression"})
    print("\n=== Per functional category (all 217 for ours; Kermut on matched) ===")
    for c in CATS:
        a = mc[mc.cat == c].groupby("assay").spearman.mean().mean()
        k = kc[(kc.cat == c) & kc.assay.isin(common)].groupby("assay").spearman.mean().mean()
        print(f"  {c:18s} ours={a:.3f}  Kermut={k:.3f}")


if __name__ == "__main__":
    main()
