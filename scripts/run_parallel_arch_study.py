#!/usr/bin/env python3
"""
run_parallel_arch_study.py — Run multi-seed architecture study with parallel GPU execution.

Runs multiple seeds simultaneously on the GPU for faster completion.
Each seed trains independently, so they can run in parallel.

Usage:
  python run_parallel_arch_study.py --arch spatiotemporal --seeds 30 --parallel 4
  python run_parallel_arch_study.py --arch vit1d --seeds 30 --parallel 6
  python run_parallel_arch_study.py --arch convlstm --seeds 30 --parallel 4
  python run_parallel_arch_study.py --arch unet1d --seeds 30 --parallel 2
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qMR_Robust.models.resnet1d import ResNet1D
from qMR_Robust.models.vit1d import ViT1D
from qMR_Robust.models.spatiotemporal_transformer import SpatioTemporalTransformer
from qMR_Robust.models.convlstm1d import ConvLSTM1D
from qMR_Robust.models.unet1d import UNet1D
from qMR_Robust.models.losses import evidential_regression_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = ROOT / "results" / "figures"
CKPT_DIR = ROOT / "results" / "checkpoints"


class MRFMetaDataset(Dataset):
    def __init__(self, h5_path, split="train", train_ratio=0.8):
        hf = h5py.File(h5_path, "r")
        n = hf.attrs["n_signals"]
        n_train = int(n * train_ratio)
        s, e = (0, n_train) if split == "train" else (n_train, n)
        self.signals = np.stack(
            [hf["corrupted_signals"][s:e].real, hf["corrupted_signals"][s:e].imag], axis=1
        ).astype(np.float32)
        self.params = hf["parameters"][s:e, :2].astype(np.float32)
        hf.close()
        if split == "train":
            self.mean = self.params.mean(0)
            self.std = self.params.std(0) + 1e-8
        else:
            self.mean = np.zeros(2, dtype=np.float32)
            self.std = np.ones(2, dtype=np.float32)

    def set_norm(self, mean, std):
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.signals[i]),
            (torch.from_numpy(self.params[i]) - self.mean) / self.std,
        )



def load_real_data():
    """Unit-safe qMRLab VFA loader (T1 in ms, zero-pad cross-sequence protocol)."""
    from qMR_Robust.data.loaders import load_qmrlab_vfa
    from pathlib import Path as _P
    qmrlab_dir = _P(__file__).resolve().parent.parent / "data" / "real" / "qmrlab" / "vfa_t1_data"
    data = load_qmrlab_vfa(qmrlab_dir, pad_mode="zeropad")
    return data.signals, data.t1_ms



def build_model(arch):
    if arch == "resnet1d":
        return ResNet1D(in_channels=2, hidden_dim=128, output_dim=2, dropout=0.1, evidential=True)
    elif arch == "vit1d":
        return ViT1D(in_channels=2, hidden_dim=128, output_dim=2, patch_size=50, n_heads=4, n_layers=4, dropout=0.1, evidential=True)
    elif arch == "spatiotemporal":
        return SpatioTemporalTransformer(in_channels=2, seq_len=1000, hidden_dim=128, output_dim=2, n_heads=4, n_temporal_layers_1=3, n_temporal_layers_2=2, dropout=0.1, evidential=True)
    elif arch == "convlstm":
        return ConvLSTM1D(in_channels=2, hidden_dim=128, n_lstm_layers=2, output_dim=2, dropout=0.1, evidential=True)
    elif arch == "unet1d":
        return UNet1D(in_channels=2, hidden_dim=64, output_dim=2, dropout=0.1, evidential=True)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def evaluate_real(model, X_real, real_t1, t_mean, t_std):
    model.eval()
    all_ep, all_gamma = [], []
    with torch.no_grad():
        for i in range(0, len(X_real), 256):
            batch = torch.from_numpy(X_real[i:i+256]).to(DEVICE)
            out = model(batch).view(-1, 2, 4)
            ep = (out[..., 3] / (out[..., 1] * (out[..., 2] - 1.0))).max(dim=-1).values
            all_ep.append(ep.cpu().numpy())
            all_gamma.append(out[..., 0].cpu().numpy())
    ep = np.concatenate(all_ep)
    gamma = np.concatenate(all_gamma) * t_std + t_mean
    error = np.abs(real_t1 - gamma[:, 0])
    ep_clean = np.where(np.isfinite(ep), ep, 1e6)
    rho, _ = spearmanr(ep_clean, error) if np.std(ep_clean) > 1e-12 else (0.0, 1.0)
    return float(rho)


def train_single_seed(seed, arch, mrf_path, X_real, real_t1, t_mean, t_std, n_epochs=30):
    """Train one seed and return result. Designed to run in parallel."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = CKPT_DIR / f"arch30_{arch}_seed{seed}.pt"
    if ckpt_path.exists():
        model = build_model(arch).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        rho = evaluate_real(model, X_real, real_t1, t_mean, t_std)
        return {"seed": seed, "rho": rho, "time_s": 0, "cached": True}

    train_ds = MRFMetaDataset(mrf_path, split="train")
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=2)
    model = build_model(arch).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            out = model(x)
            nig = out.view(out.shape[0], 2, 4)
            result = evidential_regression_loss(y, nig, coeff=1.0, epoch=epoch, annealing_epochs=10)
            loss = result["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    rho = evaluate_real(model, X_real, real_t1, t_mean, t_std)
    elapsed = time.time() - t0
    return {"seed": seed, "rho": rho, "time_s": elapsed, "cached": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, required=True, choices=["resnet1d", "vit1d", "spatiotemporal", "convlstm", "unet1d"])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--parallel", type=int, default=4, help="Number of seeds to run in parallel")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    mrf_path = str(ROOT / cfg["paths"]["failure_forecast_mrf"])

    hf = h5py.File(mrf_path, "r")
    n = hf.attrs["n_signals"]
    n_train = int(n * 0.8)
    t_mean = hf["parameters"][:n_train, :2].astype(np.float32).mean(0)
    t_std = hf["parameters"][:n_train, :2].astype(np.float32).std(0) + 1e-8
    hf.close()

    print(f"Loading real data...")
    X_real, real_t1 = load_real_data()

    # Generate seeds
    np.random.seed(12345)
    all_seeds = sorted(np.random.randint(1, 1000, max(args.seeds, 50)).tolist())
    seeds = all_seeds[:args.seeds]

    # Check for existing checkpoints
    existing = set()
    for seed in seeds:
        ckpt = CKPT_DIR / f"arch30_{args.arch}_seed{seed}.pt"
        if ckpt.exists():
            existing.add(seed)
    new_seeds = [s for s in seeds if s not in existing]

    print(f"\n{'='*60}")
    print(f"  {args.arch} — {args.seeds}-seed study (parallel={args.parallel})")
    print(f"  Existing checkpoints: {len(existing)}")
    print(f"  New seeds to train: {len(new_seeds)}")
    print(f"{'='*60}")

    results = []

    # Process existing checkpoints first (fast)
    for seed in sorted(existing):
        t0 = time.time()
        rho = train_single_seed(seed, args.arch, mrf_path, X_real, real_t1, t_mean, t_std)
        rho["time_s"] = time.time() - t0
        results.append(rho)
        label = "GOOD" if rho["rho"] > 0.8 else "MOD" if rho["rho"] > 0.3 else "NZ" if rho["rho"] > -0.1 else "DEG"
        print(f"  Seed {seed:4d}: ρ={rho['rho']:+.3f} [{label}] (cached)")

    # Process new seeds in parallel
    if new_seeds:
        print(f"\n  Training {len(new_seeds)} new seeds with {args.parallel} parallel workers...")

        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for seed in new_seeds:
                future = executor.submit(train_single_seed, seed, args.arch, mrf_path, X_real, real_t1, t_mean, t_std)
                futures[future] = seed

            for future in as_completed(futures):
                seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    label = "GOOD" if result["rho"] > 0.8 else "MOD" if result["rho"] > 0.3 else "NZ" if result["rho"] > -0.1 else "DEG"
                    print(f"  Seed {seed:4d}: ρ={result['rho']:+.3f} [{label}] ({result['time_s']:.0f}s)")
                except Exception as e:
                    print(f"  Seed {seed:4d}: ERROR {e}")
                    results.append({"seed": seed, "rho": float("nan"), "time_s": 0, "error": str(e)})

    # Sort results by seed
    results.sort(key=lambda x: x["seed"])

    # Summary
    rhos = [r["rho"] for r in results if not np.isnan(r["rho"])]
    good = sum(1 for r in rhos if r > 0.8)
    degenerate = sum(1 for r in rhos if r < -0.1)
    n = len(rhos)

    from scipy.stats import norm
    p_hat = degenerate / n if n > 0 else 0
    z = 1.96
    denom = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
    lo = max(0, center-margin)
    hi = min(1, center+margin)

    print(f"\n{'='*60}")
    print(f"  {args.arch} SUMMARY ({n} seeds)")
    print(f"{'='*60}")
    print(f"  Good: {good}/{n} ({good/n*100:.0f}%)")
    print(f"  Degenerate: {degenerate}/{n} ({degenerate/n*100:.0f}%)")
    print(f"  Failure rate: {p_hat*100:.0f}% (95% Wilson CI: [{lo*100:.1f}%, {hi*100:.1f}%])")
    good_rhos = [r for r in rhos if r > 0.8]
    if good_rhos:
        print(f"  Good seeds: mean ρ = {np.mean(good_rhos):.3f} ± {np.std(good_rhos):.3f}")

    # Save
    output = {
        "architecture": args.arch,
        "n_seeds": n,
        "seeds": seeds,
        "results": results,
        "summary": {
            "good": good,
            "degenerate": degenerate,
            "failure_rate": float(p_hat),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "mean_good_rho": float(np.mean(good_rhos)) if good_rhos else None,
        }
    }
    out_path = FIG / f"arch30_{args.arch}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
