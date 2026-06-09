"""Approximate pseudo-log-likelihood per variant: ONE forward pass, NO masking.

pll(seq) = sum_i log P(s_i | s)  over residue positions i, where P comes from the ESMCForMaskedLM
output logits at position i given the FULL (unmasked) sequence -- i.e. the model sees s_i itself, so
this is the cheap one-pass approximation to the true (masked) pseudo-LL. One scalar per variant,
saved to pll/<assay>.npy aligned to CSV row order. (Within an assay, absolute pll vs WT-relative LLR
differ only by a per-assay constant, which TabICL's per-feature normalization removes -- so absolute
is sufficient.) Reuses the embedder's length-sorted batching. fp32.
"""
import argparse, glob, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

FOLDS_DIR = "/home/ssm-user/sandbox/pfnworld/PRIMO/data/cv_folds/cv_folds_singles_substitutions"
OUT_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/pll"
MODEL = "biohub/ESMC-600M"
TOKEN_BUDGET = 60000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dev = f"cuda:{args.gpu}"; torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()
    cls_id, eos_id, pad_id = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id

    all_assays = sorted(os.path.basename(f)[:-4] for f in glob.glob(FOLDS_DIR + "/*.csv"))
    def seqlen(a):
        with open(os.path.join(FOLDS_DIR, a + ".csv")) as fh:
            next(fh); return len(next(fh).split(",")[1])
    all_assays.sort(key=seqlen)
    assays = [a for i, a in enumerate(all_assays) if i % args.nshards == args.shard]
    print(f"[gpu{args.gpu}] {len(assays)} assays", flush=True)

    for ai, assay in enumerate(assays):
        out_path = os.path.join(OUT_DIR, assay + ".npy")
        if os.path.exists(out_path):
            print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) skip {assay}", flush=True); continue
        df = pd.read_csv(os.path.join(FOLDS_DIR, assay + ".csv"))
        seqs = df["mutated_sequence"].tolist(); n = len(seqs)
        order = sorted(range(n), key=lambda i: len(seqs[i]))
        pll = np.zeros(n, dtype=np.float32)

        t0 = time.time(); i = 0
        while i < n:
            j = i; Lmax = len(seqs[order[i]]) + 2
            while j < n:
                Lcur = max(Lmax, len(seqs[order[j]]) + 2)
                if (j - i + 1) * Lcur > TOKEN_BUDGET and j > i:
                    break
                Lmax = Lcur; j += 1
            idx = order[i:j]
            enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            ids = enc["input_ids"]
            with torch.inference_mode():
                logits = model(**enc).logits                         # (B, L, vocab)
                logp = F.log_softmax(logits, dim=-1)
                tok_logp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)  # (B,L) logP of true token
            keep = enc["attention_mask"].bool() & (ids != cls_id) & (ids != eos_id) & (ids != pad_id)
            seq_pll = (tok_logp * keep).sum(1)                        # sum over residue positions
            seq_pll = seq_pll.float().cpu().numpy()
            for r, k in enumerate(idx):
                pll[k] = seq_pll[r]
            i = j
        np.save(out_path, pll)
        print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) {assay} n={n} {time.time()-t0:.1f}s "
              f"pll[min/med/max]={pll.min():.1f}/{np.median(pll):.1f}/{pll.max():.1f}", flush=True)


if __name__ == "__main__":
    main()
