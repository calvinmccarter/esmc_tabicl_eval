"""WT-marginal mutation-effect features (ESM zero-shot scoring), two modes:

  --mode wt      (DEFAULT): UNMASKED 'wt-marginals' (Meier et al.). ONE forward pass of the WT
                 sequence; score X{pos}Y = logP_wt[pos,Y] - logP_wt[pos,X] read from that pass.
  --mode masked: 'masked-marginals'. For each UNIQUE mutated position p, mask p in the WT and run;
                 score reads the masked-position logits (model predicts blind). ~L passes/assay
                 (batched), usually a stronger scorer but more expensive.

Per variant the per-mutation scores are aggregated into [min, mean, max, sum] (identical for single
mutants; generalizes to multi-mutation). Output: wtmarg/<assay>.npy (wt) or wtmarg_masked/<assay>.npy.
600M, fp32, inference_mode. Shortest-first so the long-protein tail can be deferred.
"""
import argparse, glob, os, re, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

FOLDS_DIR = "/home/ssm-user/sandbox/pfnworld/PRIMO/data/cv_folds/cv_folds_singles_substitutions"
MODEL = "biohub/ESMC-600M"
MUT_RE = re.compile(r'^([A-Za-z])(\d+)([A-Za-z])$')
STATS = ("min", "mean", "max", "sum")
TOKEN_BUDGET = 40000   # for masked mode batching (B*L)


def parse_muts(mut):
    out = []
    for part in str(mut).split(":"):
        m = MUT_RE.match(part)
        if not m:
            raise ValueError(f"bad mutant {mut}")
        out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def derive_wt(df):
    def revert(row):
        seq = list(row["mutated_sequence"])
        for wt_aa, pos, mut_aa in parse_muts(row["mutant"]):
            assert seq[pos - 1] == mut_aa, "mutant/seq mismatch"
            seq[pos - 1] = wt_aa
        return "".join(seq)
    wt = revert(df.iloc[0])
    for k in range(1, min(4, len(df))):
        assert revert(df.iloc[k]) == wt, "inconsistent WT reconstruction"
    return wt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mode", choices=["wt", "masked"], default="wt")
    ap.add_argument("--agg", choices=["stats", "deciles"], default="stats",
                    help="stats=[min,mean,max,sum] (4); deciles=percentiles 0,10,..,100 (11)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--out-tag", default="", help="suffix appended to output dir (e.g. _6b)")
    args = ap.parse_args()
    MODEL_NAME = args.model
    DECILES = list(range(0, 101, 10))
    base = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/wtmarg"
    out_dir = base + ("" if args.mode == "wt" else "_masked") + ("" if args.agg == "stats" else "_dec") + args.out_tag
    os.makedirs(out_dir, exist_ok=True)
    dev = f"cuda:{args.gpu}"; torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, dtype=dt).to(dev).eval()
    mask_id = tok.mask_token_id
    aa_id = {}

    all_assays = sorted(os.path.basename(f)[:-4] for f in glob.glob(FOLDS_DIR + "/*.csv"))
    def seqlen(a):
        with open(os.path.join(FOLDS_DIR, a + ".csv")) as fh:
            next(fh); return len(next(fh).split(",")[1])
    all_assays.sort(key=seqlen)
    assays = [a for i, a in enumerate(all_assays) if i % args.nshards == args.shard]
    print(f"[gpu{args.gpu}] mode={args.mode} {len(assays)} assays", flush=True)

    for ai, assay in enumerate(assays):
        out_path = os.path.join(out_dir, assay + ".npy")
        if os.path.exists(out_path):
            print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) skip {assay}", flush=True); continue
        df = pd.read_csv(os.path.join(FOLDS_DIR, assay + ".csv"))
        wt = derive_wt(df)
        base_ids = tok([wt], return_tensors="pt")["input_ids"][0]      # (Ltok,), token idx p = residue p

        t0 = time.time()
        # logp_at[p] -> (vocab,) log-probs for the distribution used to score position p
        logp_at = {}
        if args.mode == "wt":
            enc = tok([wt], return_tensors="pt").to(dev)
            with torch.inference_mode():
                lp = F.log_softmax(model(**enc).logits[0].float(), dim=-1).cpu().numpy()  # (Ltok,vocab)
            logp_at = lp                                                # array indexable by position
        else:
            positions = sorted({p for mut in df["mutant"] for (_, p, _) in parse_muts(mut)})
            Ltok = base_ids.shape[0]
            bs = max(1, TOKEN_BUDGET // Ltok)
            for s in range(0, len(positions), bs):
                chunk = positions[s:s + bs]
                batch = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
                for r, p in enumerate(chunk):
                    batch[r, p] = mask_id                              # mask the residue at token idx p
                with torch.inference_mode():
                    out = model(input_ids=batch.to(dev),
                                attention_mask=torch.ones_like(batch).to(dev)).logits
                    for r, p in enumerate(chunk):
                        logp_at[p] = F.log_softmax(out[r, p].float(), dim=-1).cpu().numpy()

        ncol = len(STATS) if args.agg == "stats" else len(DECILES)
        feats = np.zeros((len(df), ncol), dtype=np.float32)
        for r, mut in enumerate(df["mutant"].tolist()):
            scores = []
            for wt_aa, pos, mut_aa in parse_muts(mut):
                wid = aa_id.setdefault(wt_aa, tok.convert_tokens_to_ids(wt_aa))
                mid = aa_id.setdefault(mut_aa, tok.convert_tokens_to_ids(mut_aa))
                lp = logp_at[pos]
                scores.append(lp[mid] - lp[wid])
            s = np.array(scores)
            if args.agg == "stats":
                feats[r] = [s.min(), s.mean(), s.max(), s.sum()]
            else:
                feats[r] = np.percentile(s, DECILES)        # 11 decile columns
        np.save(out_path, feats)
        print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) {assay} n={len(df)} wtlen={len(wt)} "
              f"{time.time()-t0:.1f}s mean-LLR med={np.median(feats[:,1]):.2f}", flush=True)


if __name__ == "__main__":
    main()
