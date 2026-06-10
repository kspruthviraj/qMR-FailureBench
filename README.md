# qMR-FailureBench

**The Sim-to-Real Uncertainty Gap in Quantitative MRI: Characterization, Benchmark, and Counterfactual Correction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20632105.svg)](https://doi.org/10.5281/zenodo.20632105)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## What is this?

Machine learning models trained on **synthetic MRI data** can extract quantitative tissue measurements ($T_1$, $T_2$, metabolite concentrations) from real brain scans. But **can they tell you when they're wrong?**

We show they usually cannot. When tested on real scanner data, the models still produce reasonable measurements, but their confidence estimates break down. We call this the **sim-to-real uncertainty gap**.

This repository provides:
1. **qMR-FailureBench** — A standardized benchmark (60K signals, 5 evaluation tasks)
2. **PhysicsCorruptor** — A tool for injecting entangled MRI artifacts ($B_0$, $B_1^+$, motion)
3. **Evidential models** — ResNet-1D, ViT-1D, and Spatio-Temporal Transformer with NIG outputs
4. **Forecaster** — Failure detection, corruption attribution, and counterfactual correction
5. **All paper results** — 12 trained models, 13 figures, complete evaluation scripts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Generate synthetic data
python scripts/run_experiment.py

# Step 2: Run the full experiment suite (baselines, ablations, severity curves)
python scripts/run_v3_final.py

# Step 3: Run the sim-to-real gap experiment (requires qMRLab data)
python scripts/run_qmrlab_validation.py

# Step 4: Run the novel dual-head + attribution experiments
python scripts/run_novel_experiments.py

# Step 5: Run remaining analyses (ensemble diagnostic, counterfactual stats)
python scripts/run_final_analysis.py
```

## Project Structure

```
├── .gitignore                              # Excludes data, results, checkpoints
├── LICENSE                                 # MIT License
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── configs/
│   └── config.yaml                         # Full configuration
├── qMR_Robust/
│   ├── simulators/
│   │   ├── manager.py                      # MRF Bloch + MRS Lorentzian simulation
│   │   └── corruptor.py                    # PhysicsCorruptor (entangled B0/B1/motion)
│   ├── models/
│   │   ├── resnet1d.py                     # ResNet-1D with evidential NIG head
│   │   ├── vit1d.py                        # ViT-1D with evidential NIG head
│   │   ├── spatiotemporal_transformer.py   # Spatio-Temporal Transformer
│   │   ├── losses.py                       # NIG NLL + Evidential Regularizer
│   │   ├── baselines.py                    # MC-Dropout, Ensemble, Quantile, Heteroscedastic
│   │   ├── corruption_attribution.py       # Dual-head attribution model
│   │   ├── severity_regression.py          # Severity regression + counterfactual correction
│   │   ├── physics_aware_loss.py           # SNR-anchored evidential loss
│   │   └── registry.py                     # Model factory
│   ├── eval/
│   │   ├── forecaster.py                   # Failure detection + correction pipeline
│   │   └── metrics.py                      # Calibration, AUROC, reliability diagrams
│   └── benchmark/
│       └── __init__.py                     # qMR-FailureBench packaging
├── scripts/
│   ├── run_experiment.py                   # Step 1: Data generation + baseline training
│   ├── run_v3_final.py                     # Step 2: Full experiment suite
│   ├── run_novel_experiments.py            # Step 3: Dual-head + attribution
│   ├── run_final_analysis.py               # Step 4: Ensemble diagnostic + counterfactual stats
│   ├── run_sim_to_real_gap.py              # Sim-to-real gap characterization
│   ├── run_adaptation_curve.py             # Calibration repair scaling law
│   ├── run_qmrlab_validation.py            # Real in-vivo MRF validation (qMRLab)
│   └── run_biggaba_mrs.py                  # Real MRS validation (Big GABA)
├── tests/
│   └── test_failure_forecast.py            # Unit tests
└── paper/
    ├── main.tex                            # LaTeX source
    └── main.pdf                            # Compiled PDF
```

**Data and checkpoints** are gitignored. Generate with:
```bash
python scripts/run_experiment.py
```

## Key Results

| Method | MAE (ms) | AUROC | Attr. F1 | Can Correct? |
|--------|----------|-------|----------|--------------|
| Deterministic | 229.1 | --- | --- | No |
| Heteroscedastic | 249.1 | 0.710 | --- | No |
| Deep Ensemble (5) | 221.0 | 0.476 | --- | No |
| **Ours (NLL+ER)** | **223.9±14.5** | **0.642±0.020** | **0.859** | **Yes (39.6% improvement)** |

## The Sim-to-Real Gap

| Metric | Synthetic | Real (zero-shot) | Real (repaired) |
|--------|-----------|------------------|-----------------|
| MAE (ms) | 223.9 | 64.7 | --- |
| Spearman ρ | 0.149 | 0.358 | 0.604 |

5% of real data (233 voxels) is enough to repair the calibration gap.

## Citation

```bibtex
@dataset{qmrfailurebench2026,
  title={qMR-FailureBench: A Benchmark for Explainable Failure Forecasting and Counterfactual Correction in Quantitative MRI},
  author={Kyathanahally, Sreenath P},
  year={2026},
  doi={10.5281/zenodo.20632105},
  url={https://doi.org/10.5281/zenodo.20632105}
}
```

## License

MIT License. See [LICENSE](LICENSE).
