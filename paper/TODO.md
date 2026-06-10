# TODO.md — Path to Phenomenal Paper

## Core Reframing (All Reviewers Agree)

**Current framing:** "Here's our uncertainty model + benchmark"
**Target framing:** "We discovered a fundamental phenomenon: synthetic data transfers for regression but not uncertainty calibration. Here's the benchmark to measure it and tools to fix it."

**This is the difference between a method paper (forgettable) and a phenomenon paper (highly cited).**

---

## ✅ DONE (26 items) — See previous TODO.md versions

---

## ✅ JUST COMPLETED (Phase 1/2/3 + Pruning)

### Phase 1: Pruned Narrative
- [x] Moved selective prediction, cross-vendor, cross-field to supplementary (1 paragraph in main text)
- [x] Condensed Deep Ensemble to single paragraph in Discussion
- [x] Paper now 8 pages — tight, focused on sim-to-real gap

### Phase 2: New Figures
- [x] Corruption severity histogram (synthetic wide/heavy-tailed vs real narrow/moderate)
- [x] Reliability diagram for real zero-shot data
- [x] Canonical leaderboard table (all baselines × all tasks)

### Phase 3: Citations & Benchmark
- [x] 22 references with full titles
- [x] Benchmark README with 5 tasks, challenge splits
- [x] Fixed typos (% formatting, GE capitalization)

### Paper
- [x] 8 pages, 1.6 MB, 22 references, 12 figures, 6 tables — FOCUSED

---

## 🔴 DO NOW (Today — Highest Impact, Can Execute Immediately)

### 1. Reframe Core Narrative
- [x] Rewrite abstract to lead with sim-to-real gap phenomenon
- [x] Rewrite introduction to position as "first systematic characterization"
- [x] Make Sim-to-Real the centerpiece
- [x] Demote physics loss to brief mention
- [x] Make ER the hero in 1 sentence

### 2. Stratify Counterfactual Success Rate
- [x] Break by corruption type: B0=68.4%, B1=53.6%, Motion=70.0%
- [x] B0/Motion most correctable (structured phase errors)
- [x] B1 harder (amplitude scaling ≈ tissue variation)

### 3. Deepen Deep Ensemble Insight
- [x] Added: "Future ensemble methods must force diverse physical reasoning"
- [x] Confirmed: variance-error r=0.094 (near zero)

### 4. Flip AUROC Narrative
- [x] "Baseline achieves 0.642. We challenge community to reach >0.80"

### 5. Reference Clinical Workflow Figure
- [x] Figure 8 referenced in Discussion

### 6. Add Benchmark Tasks Table
- [x] 5 tasks: Failure detection, Attribution, Severity, Counterfactual, Sim-to-real

### 7. Recompile Paper
- [x] Paper v3 compiled: 7 pages, 1.5 MB

---

## 🟡 DO THIS WEEK (High Impact)

### 8. Benchmark README + Zenodo Release
- [x] README with quickstart, 5 tasks, challenge splits (DONE)
- [ ] PhysicsCorruptor standalone documented module
- [ ] Release on Zenodo for citable DOI (needs user: go to zenodo.org, upload, get DOI)
- [ ] GitHub release with tags

### 9. Example Notebook
- [ ] Colab-friendly notebook reproducing the sim-to-real experiment
- [ ] Shows: load data → run model → plot uncertainty → demonstrate gap

---

## 🟢 DO IF TIME (Pushes to 9.5+/10)

### 10. Big GABA Real MRS Validation
- [ ] Apply for access
- [ ] Run sim-to-real analysis on MRS
- [ ] If same pattern appears (regression transfers, uncertainty doesn't): PHENOMENON CONFIRMED
- [ ] This is the single highest-value experiment remaining

### 11. Cross-Dataset Pattern
- [ ] Show sim-to-real gap exists in BOTH MRF and MRS
- [ ] This elevates from "observation" to "phenomenon"
- [ ] "Phenomena get cited"

### 12. AUROC > 0.75
- [ ] Longer training (100+ epochs)
- [ ] Better hyperparameter tuning
- [ ] This is secondary to the scientific story

---

## THE STORY WE WANT REVIEWERS TO TELL

> "This paper discovers that synthetic qMRI data transfers for regression but fundamentally fails for uncertainty calibration — a sim-to-real gap. They release qMR-FailureBench to standardize measuring this gap, provide a practical calibration repair method, and demonstrate actionable counterfactual correction. The Evidential Regularizer is the key innovation that makes corruption-aware uncertainty possible. This will be the reference paper for anyone working on uncertainty in quantitative MRI."

---

## PRIORITY ORDER (What to Do Right Now)

1. **Stratify counterfactual success rate** (10 min, high impact)
2. **Rewrite abstract** around sim-to-real phenomenon (15 min, highest impact)
3. **Rewrite introduction** to lead with the gap (15 min, highest impact)
4. **Demote physics loss** to 1-2 sentences (5 min)
5. **Flip AUROC narrative** in conclusion (5 min)
6. **Add benchmark tasks table** (10 min)
7. **Recompile** (2 min)
8. **Then: Benchmark README + Zenodo** (2-3 hours)

### Textual Polish (Just Applied)
- [x] Softened ensemble language: 'consistent with deterministic biases'
- [x] Fixed 'similar gaps likely exist' → 'consistent with reports of'
- [x] Clarified Spearman ρ comparison (not apples-to-apples)
- [x] Renamed reliability diagram caption
- [x] Added adaptation curve (5% real data saturates repair)
- [x] Paper now 9 pages, 1.6 MB

## REMAINING (User Action)

### Big GABA MRS Validation
- [ ] Register at nitrc.org/account/register.php
- [ ] Login and download P8_P.zip, S1_P.zip, etc.
- [ ] Upload to data/real/biggaba/
- [ ] Run MRS sim-to-real validation

### Zenodo Release
- [ ] Go to zenodo.org, upload qMR-FailureBench/, get DOI
- [ ] Add DOI to paper

### arXiv
- [ ] Upload paper + supplementary to arXiv

