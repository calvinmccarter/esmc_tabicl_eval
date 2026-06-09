"""TabICLv2 regression on ESMC embeddings, following Kermut's ProteinGym supervised CV protocol.

For each assay x cv_scheme x fold: fit TabICLRegressor on the 4 training folds, predict the
held-out fold. Targets are standardized per fold using train stats (as in Kermut's benchmark.py).
Predictions are concatenated across all 5 folds, then ONE Spearman is computed per (assay, scheme)
-- exactly matching kermut reproduce branch src/process_results/merge_score_files.py.

Writes preds/<assay>__<scheme>.csv (per-variant) and appends a row to results/tabicl_spearman.csv.
"""
import argparse, glob, os, sys, time
os.environ.setdefault("TABICL_DISABLE_PROGRESS", "1")

CV_SCHEMES = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
TRAIN_CAP = 10000  # cap training context (seeded subsample) -- a single TabICL fit at
                   # n=10k already peaks ~80GB; larger contexts OOM a 97GB GPU. ~8 assays affected.
FOLDS_DIR = "/home/ssm-user/sandbox/pfnworld/PRIMO/data/cv_folds/cv_folds_singles_substitutions"
EMB_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/emb"
PRED_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/preds"
RES_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="process this shard's assays in reverse order")
    ap.add_argument("--label", default=None, help="override output shard-file suffix (default=shard idx)")
    ap.add_argument("--emb-dir", default=EMB_DIR, help="directory of per-assay embedding .npy files")
    ap.add_argument("--res-prefix", default="tabicl_spearman_shard", help="results filename prefix")
    ap.add_argument("--pred-dir", default=PRED_DIR, help="per-variant prediction output dir")
    ap.add_argument("--pll-dir", default=None, help="if set, concat per-variant pseudo-LL (.npy) as an extra feature column")
    ap.add_argument("--wtmarg-dir", default=None, help="if set, concat per-variant wt-marginal features (.npy, (n,k)) columns")
    ap.add_argument("--wtmarg-collapse", action="store_true", help="use only the mean wt-marginal column (1 col) instead of all min/mean/max/sum")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    emb_dir = args.emb_dir
    pred_dir = args.pred_dir

    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from tabicl import TabICLRegressor

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    suffix = args.label if args.label is not None else str(args.shard)
    res_path = os.path.join(RES_DIR, f"{args.res_prefix}{suffix}.csv")
    # done-set spans ALL shard files (same prefix) so parallel shards + sweep compose safely
    done = set()
    for p in glob.glob(os.path.join(RES_DIR, f"{args.res_prefix}*.csv")):
        prev = pd.read_csv(p)
        done |= set(zip(prev.assay, prev.scheme))

    assays = sorted(os.path.basename(f)[:-4] for f in glob.glob(FOLDS_DIR + "/*.csv"))
    assays = [a for i, a in enumerate(assays) if i % args.nshards == args.shard]
    if args.reverse:
        assays = assays[::-1]

    def append_row(d):
        hdr = not os.path.exists(res_path)
        pd.DataFrame([d]).to_csv(res_path, mode="a", header=hdr, index=False)

    for ai, assay in enumerate(assays):
        emb_path = os.path.join(emb_dir, assay + ".npy")
        if not os.path.exists(emb_path):
            print(f"[s{args.shard}] MISSING EMB {assay}", flush=True)
            continue
        X = np.load(emb_path).astype(np.float32)
        if args.pll_dir is not None:
            pll_path = os.path.join(args.pll_dir, assay + ".npy")
            if not os.path.exists(pll_path):
                print(f"[s{args.shard}] MISSING PLL {assay}", flush=True)
                continue
            pll = np.load(pll_path).astype(np.float32).reshape(-1, 1)
            assert pll.shape[0] == X.shape[0], f"pll row mismatch {assay}"
            X = np.concatenate([X, pll], axis=1)            # + 1 pseudo-LL column
        if args.wtmarg_dir is not None:
            wm_path = os.path.join(args.wtmarg_dir, assay + ".npy")
            if not os.path.exists(wm_path):
                print(f"[s{args.shard}] MISSING WTMARG {assay}", flush=True)
                continue
            wm = np.load(wm_path).astype(np.float32)
            if wm.ndim == 1:
                wm = wm.reshape(-1, 1)
            if args.wtmarg_collapse:
                wm = wm[:, [1]] if wm.shape[1] >= 2 else wm[:, [0]]   # mean column only (1 col)
            assert wm.shape[0] == X.shape[0], f"wtmarg row mismatch {assay}"
            X = np.concatenate([X, wm], axis=1)             # + wt-marginal column(s)
        df = pd.read_csv(os.path.join(FOLDS_DIR, assay + ".csv"))
        y = df["DMS_score"].to_numpy(dtype=np.float64)
        assert len(df) == X.shape[0], f"row mismatch {assay}"

        for scheme in CV_SCHEMES:
            if (assay, scheme) in done:
                continue
            t0 = time.time()
            folds = df[scheme].to_numpy()
            y_true = np.full(len(df), np.nan)
            y_pred = np.full(len(df), np.nan)
            try:
                for f in np.unique(folds):
                    tr_idx = np.where(folds != f)[0]
                    te = folds == f
                    # cap training context to bound GPU memory (seeded, deterministic)
                    if len(tr_idx) > TRAIN_CAP:
                        rng = np.random.default_rng(abs(hash((assay, scheme, int(f)))) % (2**32))
                        tr_idx = np.sort(rng.choice(tr_idx, TRAIN_CAP, replace=False))
                    mu, sd = y[tr_idx].mean(), y[tr_idx].std()
                    sd = sd if sd > 0 else 1.0
                    ytr = (y[tr_idx] - mu) / sd
                    reg = TabICLRegressor(random_state=42)
                    reg.fit(X[tr_idx], ytr)
                    y_pred[te] = reg.predict(X[te])
                    y_true[te] = (y[te] - mu) / sd
                rho = spearmanr(y_true, y_pred).correlation
                outdf = df[["mutant"]].copy()
                outdf["y"] = y_true
                outdf["y_pred"] = y_pred
                outdf["fold"] = folds
                outdf.to_csv(os.path.join(pred_dir, f"{assay}__{scheme}.csv"), index=False)
                append_row(dict(assay=assay, scheme=scheme, spearman=rho, n=len(df)))
                print(f"[s{args.shard}] ({ai+1}/{len(assays)}) {assay} {scheme} "
                      f"rho={rho:.3f} n={len(df)} {time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[s{args.shard}] ERROR {assay} {scheme}: {e}", flush=True)
                append_row(dict(assay=assay, scheme=scheme, spearman=float("nan"), n=len(df)))


if __name__ == "__main__":
    main()
