#!/usr/bin/env python3
"""Write PRUNED paper — focused on sim-to-real gap as the star."""
import json, subprocess
from pathlib import Path

FIG = Path("results/figures")
PAPER = Path("paper")

# Load results
gap = json.loads((FIG / "sim_to_real_gap.json").read_text())
ms = json.loads((FIG / "v3_multi_seed.json").read_text())
sev = json.loads((FIG / "v2_severity_results.json").read_text())
abl = json.loads((FIG / "v2_loss_ablation.json").read_text())
mc = json.loads((FIG / "main_comparison_metrics.json").read_text())
attr = json.loads((FIG / "attribution_metrics.json").read_text())
ens_diag = json.loads((FIG / "ensemble_diagnostic.json").read_text())
sig = json.loads((FIG / "significance_tests.json").read_text())
cf_strat = json.loads((FIG / "counterfactual_stratified.json").read_text())
ensemble = json.loads((FIG / "v3_deep_ensemble.json").read_text())
conformal = json.loads((FIG / "v3_conformal.json").read_text())
p123 = json.loads((FIG / "phase123_results.json").read_text())

# Build substitution dict
S = {}
S["gap_mae"] = f"{gap['zero_shot']['mae_ms']:.1f}"
S["synth_rho"] = f"{ms['correlation_mean']:.3f}"
S["zero_rho"] = f"{gap['zero_shot']['spearman_rho']:.3f}"
S["repair_rho"] = f"{gap['calibration_repair']['spearman_rho']:.3f}"
S["auroc"] = f"{ms['auroc_cal_mean']:.3f}"
S["auroc_ci_lo"] = f"{sig['auroc_95ci'][0]:.3f}"
S["auroc_ci_hi"] = f"{sig['auroc_95ci'][1]:.3f}"
S["cf_pct"] = f"{sev['counterfactual']['improvement_pct']:.1f}"
S["b0_pct"] = f"{cf_strat['by_corruption']['B0-dominant']['pct_improved']:.0f}"
S["mot_pct"] = f"{cf_strat['by_corruption']['Motion-dominant']['pct_improved']:.0f}"
S["b1_pct"] = f"{cf_strat['by_corruption']['B1-dominant']['pct_improved']:.0f}"
S["mae_mean"] = f"{ms['mae_ms_mean']:.1f}"
S["mae_std"] = f"{ms['mae_ms_std']:.1f}"
S["rmse_mean"] = f"{ms['rmse_ms_mean']:.1f}"
S["rmse_std"] = f"{ms['rmse_ms_std']:.1f}"
S["auroc_std"] = f"{ms['auroc_cal_std']:.3f}"
S["rho_mean"] = f"{ms['correlation_mean']:.3f}"
S["rho_std"] = f"{ms['correlation_std']:.3f}"
S["mae_ci_lo"] = f"{sig['mae_95ci'][0]:.0f}"
S["mae_ci_hi"] = f"{sig['mae_95ci'][1]:.0f}"
S["det_mae"] = f"{mc['Deterministic']['mae_ms']:.1f}"
S["det_rmse"] = f"{mc['Deterministic']['rmse_ms']:.1f}"
S["het_mae"] = f"{mc['Heteroscedastic']['mae_ms']:.1f}"
S["het_rmse"] = f"{mc['Heteroscedastic']['rmse_ms']:.1f}"
S["het_auroc"] = f"{mc['Heteroscedastic']['auroc']:.3f}"
S["qtl_mae"] = f"{mc['Quantile']['mae_ms']:.1f}"
S["qtl_rmse"] = f"{mc['Quantile']['rmse_ms']:.1f}"
S["qtl_auroc"] = f"{mc['Quantile']['auroc']:.3f}"
S["ens_mae"] = f"{ensemble['mae_ms']:.1f}"
S["ens_rmse"] = f"{ensemble['rmse_ms']:.1f}"
S["ens_auroc"] = f"{ensemble['auroc']:.3f}"
S["ens_rho"] = f"{ensemble['correlation']:.3f}"
S["conf_mae"] = f"{conformal['mae_ms']:.1f}"
S["er_only_auroc"] = f"{abl['NLL only']['auroc']:.2f}"
S["er_plus_auroc"] = f"{abl['NLL+ER']['auroc']:.2f}"
S["ens_r"] = f"{ens_diag['pearson_r']:.3f}"
S["b0_f1"] = f"{attr['B0']['f1']:.3f}"
S["b1_f1"] = f"{attr['B1']['f1']:.3f}"
S["mot_f1"] = f"{attr['Motion']['f1']:.3f}"
S["b0_prec"] = f"{attr['B0']['precision']:.3f}"
S["b0_rec"] = f"{attr['B0']['recall']:.3f}"
S["b1_prec"] = f"{attr['B1']['precision']:.3f}"
S["b1_rec"] = f"{attr['B1']['recall']:.3f}"
S["mot_prec"] = f"{attr['Motion']['precision']:.3f}"
S["mot_rec"] = f"{attr['Motion']['recall']:.3f}"

# Ablation rows
abl_rows = ""
for name, vals in abl.items():
    delta = vals["auroc"] - abl["NLL only"]["auroc"]
    abl_rows += f"    {name} & {vals['mae_ms']:.0f} & {vals['auroc']:.3f} & {delta:+.3f} \\\\\\\\\n"
S["ABL_ROWS"] = abl_rows

# Counterfactual rows
cf_rows = ""
for name, vals in cf_strat["by_corruption"].items():
    cf_rows += f"    {name} & {vals['n_samples']} & {vals['pct_improved']:.1f}\\\\% & {vals['mean_improvement_ms']:.0f} \\\\\\\\\n"
S["CF_ROWS"] = cf_rows

# Domain shift rows (combined cross-vendor + cross-field)
cv = p123["cross_vendor"]
cf = p123["cross_field"]
domain_rows = ""
for name, vals in cv.items():
    vendor = name.split("_")[-1].upper()
    if vendor == "GE": vendor = "GE"
    domain_rows += f"    Vendor ({vendor}) & {vals['mae_ms']:.1f} & {vals['spearman_rho']:.3f} \\\\\\\\\n"
for name, vals in cf.items():
    field = name.split("_")[-1]
    domain_rows += f"    Field ({field}) & {vals['mae_ms']:.1f} & {vals['spearman_rho']:.3f} \\\\\\\\\n"
S["DOMAIN_ROWS"] = domain_rows

# Read template and substitute
template = (PAPER / "template_pruned.tex").read_text()
for key, val in S.items():
    template = template.replace("{{" + key + "}}", val)

(PAPER / "main.tex").write_text(template)
print("Paper written.")

for i in range(3):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=str(PAPER), capture_output=True, text=True, timeout=120)
    for line in r.stdout.split("\n"):
        if "Output written" in line:
            print(f"Pass {i+1}: {line.strip()}")

pdf = PAPER / "main.pdf"
if pdf.exists():
    print(f"PDF: {pdf.stat().st_size / 1024 / 1024:.1f} MB")
print("DONE")
