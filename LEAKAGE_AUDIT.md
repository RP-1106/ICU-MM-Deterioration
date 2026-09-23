# Label leakage audit: feature window length

*Rhea Pandita. Fork of [ayushdeo/ICU-MM](https://github.com/ayushdeo/ICU-MM).*

## TL;DR

The original ClinicalBERT model reported a test AUROC of **0.9998** for predicting respiratory failure. That score came almost entirely from **one number the pipeline wrote into every patient's text: how many hours of data had been collected.**

- For patients who had respiratory failure, data collection stopped at the moment of failure.
- For everyone else, it ran for exactly 12 hours.

As a result, **every stable patient had a window of exactly 12.00 h and no failing patient did.** Whether a patient's window equals 12 h predicts the label perfectly on held-out data (**AUROC 1.000**).

I rebuilt the dataset with a landmark design, so every patient gets the same fixed data window. On that data, the honest structured-model performance is:

| Predict failure after | Test AUROC | Test AUPRC | Chance AUPRC |
|---|---|---|---|
| 3 h | **0.77** | 0.60 | 0.19 |
| 6 h | **0.70** | 0.31 | 0.11 |
| 12 h | **0.65** | 0.12 | 0.07 |

## 1. What the original pipeline did

`GRIDS_FeatureEng_BERT_train.ipynb`, cell 16:

```python
cohort_clean["feature_cutoff"] = cohort_clean.apply(
    lambda row: row["rf_time"]                                  # positives: stop at the failure
        if row["respiratory_failure"] == 1 and pd.notna(row["rf_time"])
        else row["icu_intime"] + pd.Timedelta(hours=12),        # negatives: full 12 h
    axis=1)
```

There was a good reason for stopping at `rf_time`: it keeps labs and medications ordered *in response to* the failure out of the features. The problem is that the stopping point now depends on the label, which creates a new leak.

| `feature_window_hours` (train set) | Stable | Failed |
|---|---|---|
| exactly 12.0 h | **100%** | **0%** |
| less than 12 h | 0% | 89.4% (median 1.92 h) |
| more than 12 h | 0% | 10.6% |

The last row was a surprise. Patients who failed between hour 12 and hour 48 were given data **up to their failure time**, which is more than the 12 hours the design intended.

The window length reached the model in three ways:

1. As feature columns: `feature_window_hours`, `obs_window_hours` and `is_short_stay`.
2. As a sentence in the ClinicalBERT text: *"Clinical data available for 23 minutes following ICU admission."*
3. Through anything that grows with the window. Examples are the `*_count` lab columns, `total_presc`, how many labs were never measured, and how long the text is.

`stay_duration_hours` (the whole ICU stay, known only at discharge) was a separate source of future information.

## 2. Evidence (`notebooks/01_leakage_audit.ipynb`)

Train: 63,852 stays. Test: 13,769 stays. 38% positive.

**Test AUROC from a single number, with no model:**

| Single feature | Test AUROC |
|---|---|
| \|window − 12 h\| | **1.000** |
| `feature_window_hours` (shorter = failure) | 0.891 |
| clinical text word count | 0.772 |
| number of labs never measured | 0.747 |
| total lab draws | 0.714 |
| `total_presc` | 0.701 |
| `stay_duration_hours` (future info) | 0.675 |

**Structured models on the original features:**

| Feature set | LogReg AUROC | GradBoost AUROC |
|---|---|---|
| A. as built | 0.916 | **0.999** |
| B. minus window columns | 0.891 | 0.937 |
| C. minus window columns and counts | 0.874 | 0.915 |

Gradient boosting on the structured features alone reproduces the 0.9998 headline, with no language model. Removing columns (B and C) doesn't fix the problem, because the missing-data pattern still depends on the window.

**Timer test.** I took the stable test patients and changed only the window from 12 h to 2 h, leaving every lab and medication value the same. Their mean predicted risk went from **0.1% to 100.0%**. The model was measuring elapsed time, not patient state.

## 3. Why it matters in deployment

At prediction time you know how long the patient has been in the ICU. You don't know how long until they fail. In training, though, the window meant "time until failure" for the positive cases, and "exactly 12 h" meant "will not fail." Neither can be known when the prediction is actually made.

There's also a timing problem. The median time to failure is **1.9 h**, so most failures happen before a 12-hour prediction window has even closed.

## 4. The fix: landmark design (`notebooks/02_landmark_redesign.ipynb`)

1. Pick a fixed landmark time *L*.
2. Keep only patients who are still in the ICU and haven't had respiratory failure by *L*.
3. Give every patient the same window: labs from −6 h to *L*, medications from 0 h to *L*. The new label is **respiratory failure between *L* and 48 h**.

Removed: `feature_window_hours`, `obs_window_hours`, `is_short_stay`, `stay_duration_hours`, and the "Clinical data available…" sentence.

Everything else is unchanged: the same 20 labs × 8 statistics, 13 medication features, text template, and patient-level 70/15/15 split (`random_state=42`). I checked the vectorized feature code against the original per-stay functions and the outputs match exactly.

| Landmark | Stays (train/val/test) | Positives | Positive rate | LogReg AUROC / AUPRC | GradBoost AUROC / AUPRC |
|---|---|---|---|---|---|
| 3 h | 48,570 / 10,264 / 10,450 | 13,114 | 18.9% | 0.742 / 0.539 | **0.775 / 0.602** |
| 6 h | 43,553 / 9,302 / 9,481 | 6,623 | 10.6% | 0.676 / 0.268 | **0.701 / 0.310** |
| 12 h | 40,180 / 8,552 / 8,573 | 3,704 | 6.4% | **0.655 / 0.119** | 0.654 / 0.117 |

*(Test set shown. Validation-set numbers are within ±0.02.)*

**How to read this:**

- **The real problem is much harder.** The 12 h landmark keeps only **3,704** of the original 34,828 positive cases (11%), because most failures happen before hour 12.
- **Accuracy drops as the prediction reaches further ahead.** AUPRC is 3.2× the chance level at 3 h, 2.9× at 6 h and 1.8× at 12 h. Predicting closer to the event is easier.
- **The features are missing the obvious respiratory signals.** They include no SpO₂, respiratory rate, FiO₂, oxygen device or blood gases, even though the outcome is defined by respiratory support. This is the clearest next improvement.

## 5. Not yet redone

- **Respiratory vitals before *L***, from `respiratory_chartevents.csv`. These are legitimate features, because at-risk patients haven't met any failure criterion by *L*.
- **ClinicalBERT on landmark data.** The new CSVs work directly as inputs to `ClinicalBERT_Train.ipynb`; only the three paths in cell 2 need changing.
- **Multimodal fusion.** It needs the CXR embeddings to be re-matched to the new cohort.
- **Streamlit app.** Its "ICU data window" slider feeds the leaky feature and should be removed once the models are retrained.

## Data use

MIMIC-IV is credentialed PhysioNet data. This repository contains code and aggregate metrics only. It contains no patient-level rows, and notebook outputs are kept to summary statistics.
