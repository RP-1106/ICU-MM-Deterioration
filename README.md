# ICU Respiratory Deterioration Prediction

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/ClinicalBERT-HuggingFace-FFD21E">
  <img src="https://img.shields.io/badge/data-MIMIC--IV%20v3.1-58a6ff">
</p>

This project predicts **which ICU patients will need advanced breathing support in the next hours**, using only information available at the time of prediction. It is built on MIMIC-IV (91K ICU stays, 35.5M lab records) and designed around one rule: *the model may only see what a clinician would know at that moment.*

| Predict after | At-risk stays | Events | **Test AUROC** | Test AUPRC | AUPRC at chance |
|---|---|---|---|---|---|
| 3 h after ICU admission | 60,561 | 13,513 | **0.830** | 0.714 | 0.223 |
| 6 h | 53,518 | 6,837 | **0.768** | 0.456 | 0.126 |
| 12 h | 48,706 | 3,835 | **0.716** | 0.181 | 0.074 |

The strongest predictor is the patient's **recent oxygen flow rate**. A fine-tuned ClinicalBERT on the same information reaches 0.678 AUROC at 12 h, below the gradient-boosted model.

---

## The prediction task

**Landmark design.** At a fixed time *L* after ICU admission (3, 6 or 12 h):

1. **At-risk cohort.** Include patients who are still in the ICU and have **not yet** needed advanced respiratory support.
2. **Features.** Use only data recorded before *L*. Every patient gets exactly the same observation window.
3. **Outcome.** Predict whether the patient **starts advanced respiratory support between *L* and 48 h**.

**Advanced respiratory support** means any of:
- invasive ventilation;
- non-invasive ventilation (BiPAP/CPAP);
- high-flow nasal cannula, charted as the device or as an oxygen flow above 15 L/min;
- FiO₂ ≥ 60%.

Support that is already running at ICU admission counts at hour 0, so those patients are never in the at-risk group.

### Why this design
- **A fixed window for everyone.** If the amount of data a patient has depends on their outcome, a model can learn from the amount instead of the content. I measured this directly: in a pipeline where the data cutoff followed the outcome, window length alone scored an AUROC of 1.00. Giving every patient an identical window rules this out by construction.
- **An outcome built from raw charting.** The outcome is derived and checked against MIMIC's respiratory tables. The checks include unit-aware FiO₂, pre-admission intubations and device-name matching, and they found and corrected silent rule failures in an earlier label definition. See `03_label_check`.
- **Safe respiratory features.** Because at-risk patients have had no advanced support before *L*, their earlier oxygen data (nasal cannula, flow ≤ 15 L/min, FiO₂ < 60%) cannot already satisfy the outcome. That makes it safe to use as features.

## Features
| Group | Count | Details |
|---|---|---|
| Labs | 160 | 20 common labs × mean, min, max, count, first, last, change, and mean over hours 0–6 (from −6 h to *L*) |
| Medications | 13 | order counts, IV/drip counts, and 8 drug-category flags (0 to *L*) |
| Oxygen therapy | 12 | device type, number of device charts, O₂ flow (max, last, count), FiO₂ (max, last, count), any supplemental O₂ |
| Demographics | 2 | age, sex |

## Results in detail

**What each part contributes.** Test AUROC / AUPRC for the better of logistic regression and gradient boosting, using the same splits and settings throughout:

| Predict after | Labs + medications | + oxygen-therapy features |
|---|---|---|
| 3 h | 0.789 / 0.671 | **0.830 / 0.714** |
| 6 h | 0.697 / 0.373 | **0.768 / 0.456** |
| 12 h | 0.658 / 0.136 | **0.716 / 0.181** |

**Oxygen flow and risk.** The effect climbs steadily. At the 12 h landmark, the rate of new advanced support is:
- 4.2% with no flow charted;
- 8.6% at up to 6 L/min;
- 21.2% at 7–15 L/min;
- 25.9% above 15 L/min.

I checked this feature against documentation lag, where high flow is recorded before the device is charted. Failure rates stay well below 100%, and events are not bunched right after the landmark.

**ClinicalBERT (12 h).**
- **Setup:** each patient's features written out as a text summary, then Bio_ClinicalBERT fine-tuned on every positive plus 3 negatives per positive, with a class-weighted loss, the epoch chosen by validation AUPRC, and no truncation (maximum length 384 tokens).
- **Result:** AUROC **0.678** / AUPRC 0.149, compared with 0.716 / 0.177 for gradient boosting on the same patients. Averaging the two models did not help. When the input is essentially a table, tree models read the numbers more effectively than a language model reads them as text.

## Repository
```
notebooks/
├── 01_leakage_audit.ipynb          # single-feature and counterfactual leakage diagnostics
├── 02_landmark_redesign.ipynb      # vectorised landmark feature pipeline (35.5M labs, ~2 min)
├── 03_label_check.ipynb            # outcome-definition checks against raw respiratory charting
├── 04_landmark_v2.ipynb            # final outcome, oxygen features, model comparison, feature importance
└── 05_clinicalbert_landmark.ipynb  # ClinicalBERT fine-tuning and comparison
results/                            # aggregate metrics only (JSON / PNG)
scripts/                            # MIMIC-IV extraction scripts (cohort, labs, prescriptions, respiratory tables)
```

**Engineering notes**
- Lab features are built with chunked, vectorised pandas and cached as Parquet. I verified the output against a reference per-patient implementation, and it matches exactly.
- Splits are 70/15/15 **by patient** (no patient appears in two splits), with a fixed seed. Models are selected on validation, and each test set is scored once.
- Every notebook prints summary statistics only. No patient-level data is written to the repository.

## Reproducing
1. Get credentialed access to [MIMIC-IV](https://physionet.org/content/mimiciv/) and build the tables in `data/comb/` with `scripts/`.
2. Run notebooks `02 → 05` in Google Colab. `05` needs a T4 GPU; the others run on CPU.
3. Metrics are written as JSON files, matching those in `results/`.

## Limitations and next steps
- **Vital signs are not included yet.** SpO₂, respiratory rate and heart rate come from MIMIC-IV `chartevents` and are the most likely source of further improvement.
- Calibration and alert-threshold selection are not yet evaluated.
- The outcome uses documented respiratory support as a stand-in for respiratory failure, and the data comes from a single hospital (MIMIC-IV). External validation, for example on eICU, would be needed before any real use.

## Acknowledgements
This project builds on the codebase and MIMIC-IV extraction scripts of **[ICU-MM](https://github.com/ayushdeo/ICU-MM)**, a USC GRIDS team project led by Ayush Deo (MIT License). The landmark framework, outcome definition, oxygen-therapy features, diagnostics and all results above are this project's own work.

*MIMIC-IV is credentialed PhysioNet data. This repository contains code and aggregate metrics only.*

