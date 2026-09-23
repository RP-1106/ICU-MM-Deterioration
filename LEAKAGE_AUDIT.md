# Audit and redesign of ICU respiratory-deterioration prediction

*Rhea Pandita. Fork of [ayushdeo/ICU-MM](https://github.com/ayushdeo/ICU-MM) (USC GRIDS team project).*

## Summary

The original pipeline reported a test **AUROC of 0.9998** for predicting respiratory failure from the first 12 hours of an ICU stay. This audit found two independent problems behind that number and rebuilt the task so the reported performance is honest.

1. **Label leakage in the features.** The window of data collected for each patient ended when that patient's outcome happened. As a result, the window length alone identified the label perfectly on held-out data (AUROC **1.000**).
2. **Errors in the outcome label.**
   - One of its three rules never fired.
   - Patients who arrived already intubated were counted as "did not fail."
   - A humidified mask was counted as high-flow oxygen.

**Final, leak-free results.** Among patients not yet on advanced breathing support at hour *L*, the model predicts who starts it by hour 48. Test set: 7–10k ICU stays per landmark.

| Predict after | At-risk stays | Events | AUROC | AUPRC | AUPRC at chance |
|---|---|---|---|---|---|
| 3 h | 60,561 | 13,513 | **0.830** | 0.714 | 0.223 |
| 6 h | 53,518 | 6,837 | **0.768** | 0.456 | 0.126 |
| 12 h | 48,706 | 3,835 | **0.716** | 0.181 | 0.074 |

*Gradient boosting on labs, medications and oxygen data recorded before the landmark (logistic regression at 12 h, where it performed slightly better). A fine-tuned ClinicalBERT trails this model at 12 h: 0.678 vs 0.716 AUROC.*

The notebooks are in `notebooks/`, numbered in the order they were run. The results files are in `results/` and contain summary numbers only.

---

## 1. The leak (`01_leakage_audit.ipynb`)

`GRIDS_FeatureEng_BERT_train.ipynb` set each patient's feature cutoff like this:

```python
feature_cutoff = rf_time if respiratory_failure == 1 else icu_intime + 12h
```

Stopping at `rf_time` was meant to keep treatment given *after* the failure out of the features. The problem is that the cutoff now depends on the label. On the training set:

| `feature_window_hours` | Stable patients | Patients who failed |
|---|---|---|
| exactly 12.0 h | **100%** | **0%** |
| under 12 h | 0% | 89.4% (median 1.92 h) |
| over 12 h | 0% | 10.6% (failures between 12 and 48 h, whose features ran up to the failure) |

The window length reached the model in three ways:
- as feature columns;
- as a sentence in the ClinicalBERT text: *"Clinical data available for 23 minutes…"*;
- indirectly, through anything that grows with the window, such as lab counts, the number of missing labs, and how long the text is.

`stay_duration_hours`, the length of the whole ICU stay, was also used as a feature, even though it's only known at discharge.

**Evidence** (test set: 13,769 stays):

| Check | Result |
|---|---|
| Rule "window ≠ 12 h", no model | AUROC **1.000** |
| Gradient boosting on the structured features as built, no ClinicalBERT | AUROC **0.999**, matching the 0.9998 headline |
| Same model with the window and count columns removed | AUROC 0.915, still inflated because the missing-data pattern depends on the window |
| Stable patients with only the window changed from 12 h to 2 h | mean predicted risk **0.1% → 100.0%** |

**Also affected:** the multimodal fusion notebook (`Fusion_Code_Final.ipynb`) and the Streamlit app both use `feature_window_hours` and the original label. Their reported results (fused AUROC 0.855, structured + CXR 0.870) therefore come from the same leaky set-up and would need to be rerun.

## 2. Landmark redesign (`02_landmark_redesign.ipynb`)

A *landmark* design gives every patient the same observation window:

1. Pick a fixed time *L*.
2. Keep only patients still in the ICU and not yet failed at *L*.
3. Use data from before *L* only, and predict failure in (*L*, 48 h].

The window-length columns, `stay_duration_hours` and the window sentence are all removed. I checked the vectorized feature code against the team's per-patient functions on the same inputs, and the outputs match exactly. With the team's label, results fell to AUROC **0.775 / 0.701 / 0.655** at 3 / 6 / 12 h.

## 3. Errors in the label (`03_label_check.ipynb`)

`build_respiratory_failure_labels.py` marks failure at the first of three events: ventilation, oxygen escalation, or high FiO₂. Recomputing it reproduces the label file for 99.99% of stays. It has these problems:

| Problem | Evidence |
|---|---|
| **The high-FiO₂ rule never fires.** It looks for a label called `"fio2"`, but MIMIC names this measurement *"Inspired O2 Fraction"*. The values are also percentages (median 40; 99.9% are above 1), so the `≥ 0.6` cutoff was wrong too. | 0 matching rows. A working rule would cover 29,239 stays. |
| **Already-intubated patients count as "not failed."** Ventilation that started before ICU admission is dropped, and a charted endotracheal tube isn't one of the rules. | 12.5% of the 12 h "at-risk" group had an endotracheal or tracheostomy tube charted before hour 12, and their failure rate matched everyone else's (6.4% vs 6.5%). |
| **Most "failures" are patients arriving on support.** | 38% of events happen within 1 h of admission, and 32% of positives are ventilation within the first hour. |
| **"High flow neb" counts as high-flow oxygen.** It's a humidified mask. | Matched by the keyword `"high flow"`. |
| **Column labels differ from what the script assumes.** Item 223834 is the oxygen *flow rate*, not the oxygen device. | This one is harmless to the label, but matters for building features. |

## 4. Corrected label and oxygen features (`04_landmark_v2.ipynb`)

**New label: first start of advanced breathing support.** That means invasive or non-invasive ventilation, high-flow nasal cannula (the charted device, or an oxygen flow above 15 L/min), or FiO₂ ≥ 60%. Support already running at admission counts at hour 0.
- The positive rate goes from 38.2% to 48.2%.
- 9,248 stays become positive, mostly patients already on support. 86 stop being positive.

Because "at risk" now means *no advanced support before L*, the oxygen data recorded before *L* can't already meet the label. That makes it safe to use as features: the oxygen device, the flow rate (15 L/min or less), and FiO₂ below 60%.

**Test-set results.** Each cell shows AUROC / AUPRC for the better model. The model settings and the split method are the same in every column.

| Predict after | Team label | Corrected label | Corrected label + oxygen features |
|---|---|---|---|
| 3 h | 0.775 / 0.602 | 0.789 / 0.671 | **0.830 / 0.714** |
| 6 h | 0.701 / 0.310 | 0.697 / 0.373 | **0.768 / 0.456** |
| 12 h | 0.655 / 0.119 | 0.658 / 0.136 | **0.716 / 0.181** |

**What drives the predictions.** The most recent oxygen flow rate before the landmark is the top feature at 6 h and 12 h. At 3 h the top feature is how often the oxygen device was charted. The effect climbs steadily with flow. At 12 h, the failure rate is:
- 4.2% with no flow charted;
- 8.6% at up to 6 L/min;
- 21.2% at 7–15 L/min;
- 25.9% above 15 L/min.

**Leak check on that feature.** Could a high flow before *L* just mean high-flow oxygen that hadn't been charted yet? The >15 L/min group is small (123–170 stays). Its failure rate is not close to 100%, and its events aren't bunched right after *L* (about 10% within 1 h, the same as the other groups). I still count flows above 15 L/min as support in the final label, to be conservative. Without that rule, the results are 0.837 / 0.760 / 0.734 AUROC.

## 5. ClinicalBERT on the corrected data (`05_clinicalbert_landmark.ipynb`)

This uses the team's model (`Bio_ClinicalBERT`) and learning rate, trained on the 12 h landmark data:
- training sample of every positive plus 3 negatives per positive, with a class-weighted loss;
- epoch chosen by validation AUPRC;
- maximum length of 384 tokens, so 0% of texts are cut off.

| Model (12 h, test set) | AUROC | AUPRC |
|---|---|---|
| ClinicalBERT on the patient text | 0.678 | 0.149 |
| Gradient boosting on the same data | **0.716** | **0.177** |
| Average of the two | 0.708 | 0.173 |
| *(chance)* | 0.5 | 0.074 |

The text model sees the same information as a list of numbers written out as sentences, and it does worse than the tabular model. Averaging the two doesn't help. A first run cut the texts off at 256 tokens, which dropped the oxygen sentence at the end for 63% of patients and scored 0.627. I added a check that stops the run if more than 2% of texts are truncated.

## 6. Limitations and next steps

- **Vital signs.** SpO₂, respiratory rate and heart rate aren't in the project's extracted tables. They need MIMIC-IV `chartevents` (PhysioNet access) and are the most likely source of further improvement.
- **Multimodal fusion and the Streamlit app** still use the leaky features and the original label. They should be rebuilt on the landmark cohort, including re-matching the chest X-ray embeddings.
- **Calibration** (whether the predicted risk matches the actual rate) and decision thresholds haven't been evaluated yet.
- The label uses documented respiratory support as a stand-in for respiratory failure. Clinical practice, such as when to start high-flow oxygen, varies between units.

## Data use

MIMIC-IV is credentialed PhysioNet data. This repository contains code and summary metrics only. It contains no patient-level rows, and notebook outputs are limited to summary statistics.
