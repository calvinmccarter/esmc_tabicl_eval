"""Embed every variant sequence in the ProteinGym single-mutant assays with ESMC-600M.

Uses the SECOND-TO-LAST transformer layer (hidden_states[-2]), mean-pooled over residue
positions (excluding <cls>/<eos>/<pad>). One .npy per assay, aligned to CSV row order.

Model: biohub/ESMC-600M  (transformers-native, model_type=esmc, 36 layers, d_model=1152).
Run one process per GPU with --shard/--nshards to split the 217 assays.
"""
import argparse, glob, os, sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

FOLDS_DIR = "/home/ssm-user/sandbox/pfnworld/PRIMO/data/cv_folds/cv_folds_singles_substitutions"
OUT_DIR = "/home/ssm-user/sandbox/pfnworld/esmc_tabicl_eval/emb"
MODEL = "biohub/ESMC-600M"
TOKEN_BUDGET = 60000  # tokens per batch (B*L); fp32 uses ~36GB here, fits 97GB GPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--layers", default="-2", help="comma-sep hidden_states indices to AVERAGE before pooling (default -2)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(dev)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to(dev).eval()
    cls_id, eos_id, pad_id = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id

    # order shortest-sequence first so the bulk finishes fast and the few very long
    # assays land last; round-robin shard on that order to balance both GPUs.
    all_assays = sorted(os.path.basename(f)[:-4] for f in glob.glob(FOLDS_DIR + "/*.csv"))
    def seqlen(a):
        with open(os.path.join(FOLDS_DIR, a + ".csv")) as fh:
            next(fh)
            return len(next(fh).split(",")[1])
    all_assays.sort(key=seqlen)
    assays = [a for i, a in enumerate(all_assays) if i % args.nshards == args.shard]
    print(f"[gpu{args.gpu}] {len(assays)} assays", flush=True)

    for ai, assay in enumerate(assays):
        out_path = os.path.join(out_dir, assay + ".npy")
        if os.path.exists(out_path):
            print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) skip {assay}", flush=True)
            continue
        df = pd.read_csv(os.path.join(FOLDS_DIR, assay + ".csv"))
        seqs = df["mutated_sequence"].tolist()
        n = len(seqs)
        order = sorted(range(n), key=lambda i: len(seqs[i]))  # batch similar lengths
        emb = np.zeros((n, model.config.d_model), dtype=np.float32)

        t0 = time.time()
        i = 0
        while i < n:
            j = i
            Lmax = len(seqs[order[i]]) + 2
            # grow batch while within token budget
            while j < n:
                Lcur = max(Lmax, len(seqs[order[j]]) + 2)
                if (j - i + 1) * Lcur > TOKEN_BUDGET and j > i:
                    break
                Lmax = Lcur
                j += 1
            idx = order[i:j]
            batch = [seqs[k] for k in idx]
            enc = tok(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True)
            hs = sum(out.hidden_states[i] for i in layers) / len(layers)  # average selected layers (B, L, d)
            ids = enc["input_ids"]
            keep = enc["attention_mask"].bool() & (ids != cls_id) & (ids != eos_id) & (ids != pad_id)
            m = keep.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1.0)
            pooled = pooled.float().cpu().numpy()
            for r, k in enumerate(idx):
                emb[k] = pooled[r]
            i = j
        np.save(out_path, emb)
        dt = time.time() - t0
        print(f"[gpu{args.gpu}] ({ai+1}/{len(assays)}) {assay} n={n} {dt:.1f}s -> {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
