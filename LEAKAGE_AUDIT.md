# Label leakage audit: feature window length

*Rhea Pandita. Fork of [ayushdeo/ICU-MM](https://github.com/ayushdeo/ICU-MM).*

> ✏️ **Before you publish:** replace every `[__]` with the number from your own run (`results/leakage_audit_results.json` and `results/landmark_results.json`). Don't publish a number you didn't measure yourself.

## TL;DR

The ClinicalBERT model got a test AUROC of 0.9998. Most of that came from **one number the pipeline wrote into every patient's text: how many hours of data were collected.** For patients who had respiratory failure, collection stopped when the failure happened. For everyone else it ran the full 12 hours. So a window shorter than 12 hours meant "this patient failed."

- That window length **alone**, with no model, gets test AUROC **[__]**.
- Taking out the window-dependent features drops the structured model from **[__]** to **[__]**.
- The fix is a **landmark design**. On the corrected data, the structured model gets AUROC **[__]** and AUPRC **[__]** (chance level: **[__]**).

## 1. What the original pipeline did

`GRIDS_FeatureEng_BERT_train.ipynb`, cell 16:

```python
cohort_clean["feature_cutoff"] = cohort_clean.apply(
    lambda row: row["rf_time"]                                  # positives: stop at the failure
        if row["respiratory_failure"] == 1 and pd.notna(row["rf_time"])
        else row["icu_intime"] + pd.Timedelta(hours=12),        # negatives: full 12 h
    axis=1)
cohort_clean["feature_window_hours"] = (cohort_clean["feature_cutoff"] - cohort_clean["icu_intime"]).dt.total_seconds() / 3600
```

There was a good reason for stopping at `rf_time`: it keeps labs and medications ordered *in response to* the failure out of the features. The problem is that the stopping point now depends on the label, which creates a new leak.

| | Stable patients | Patients who failed |
|---|---|---|
| `feature_window_hours` | always exactly 12.0 | median [__] h; [__]% below 12 h |

That number then reached the model in three ways:

1. As a feature column: `feature_window_hours`, plus `obs_window_hours` and `is_short_stay`.
2. As a sentence in the ClinicalBERT text: *"Clinical data available for 23 minutes following ICU admission."*
3. **Indirectly, through anything that grows with the window.** Examples are every `*_count` lab column, `total_presc`, `unique_drugs`, `iv_count`, how many labs are missing, and how long the text is. A patient cut off at hour 1 has had one creatinine draw. A patient observed for 12 hours has had several.

A separate issue: **`stay_duration_hours` is the length of the whole ICU stay**, which isn't known until discharge. It's also future information and shouldn't be a feature.

## 2. Evidence (`notebooks/01_leakage_audit.ipynb`)

**Test AUROC from one number, no model:**

| Single feature | Test AUROC |
|---|---|
| `feature_window_hours` | [__] |
| total lab draws (sum of `*_count`) | [__] |
| number of labs never measured | [__] |
| clinical text word count | [__] |

**Structured model, removing leaky features step by step:**

| Feature set | LogReg AUROC | GradBoost AUROC |
|---|---|---|
| A. as built | [__] | [__] |
| B. minus window columns | [__] | [__] |
| C. minus window columns and counts | [__] | [__] |

Even C is an upper bound, because which labs are missing still depends on how long the window was.

**Timer test.** I took the stable test patients and changed only the window length from 12 h to 2 h, leaving every lab and medication the same. Their mean predicted risk went from **[__]%** to **[__]%**. A model that reacts this way is measuring elapsed time, not patient state.

## 3. Why it matters in deployment

At prediction time you know how long the patient has been in the ICU. You don't know how long until they fail. In training, though, `feature_window_hours` meant "time until failure" for the positive cases. The training data contains no stable patient with a 2-hour window, so at hour 2 the model flags nearly everyone.

There's also a timing problem. Median time to failure is [__] h, and [__]% of failures happen inside the 12-hour window used to predict them. A model can't really predict an event that has usually already happened before the prediction is made.

## 4. The fix: landmark design (`notebooks/02_landmark_redesign.ipynb`)

1. Pick a fixed landmark time *L* (12 h).
2. Keep only patients still in the ICU who haven't had respiratory failure by *L*.
3. Give every patient the same window: labs from −6 h to *L*, medications from 0 h to *L*. The new label is **respiratory failure between *L* and 48 h**.

Also removed: `feature_window_hours`, `obs_window_hours`, `is_short_stay`, `stay_duration_hours`, and the "Clinical data available…" sentence. The count features stay in. Once every patient has the same window, "eight creatinine draws in 12 hours" really does mean the patient is being watched more closely.

Everything else is unchanged: the same 20 labs × 8 statistics, the same 13 medication features, the same text template, and the same patient-level 70/15/15 split with `random_state=42`. I checked the vectorized feature code against the original per-stay functions and the outputs match exactly.

| Landmark | Stays | Positives | Positive rate | LogReg AUROC / AUPRC | GradBoost AUROC / AUPRC |
|---|---|---|---|---|---|
| 12 h | [__] | [__] | [__]% | [__] / [__] | [__] / [__] |
| 6 h *(optional)* | [__] | [__] | [__]% | [__] / [__] | [__] / [__] |

The fix costs a lot of data: roughly 90% of positive cases are gone, because most failures happen before hour 12. That is simply how hard the real problem is. Published ICU deterioration models usually land around 0.70–0.85 AUROC.

## 5. Not yet redone

- **ClinicalBERT on landmark data.** The new CSVs work directly as inputs to `ClinicalBERT_Train.ipynb`; only the three paths in cell 2 need changing. Result: [__ / not yet run].
- **Multimodal fusion.** It needs the CXR embeddings to be re-matched to the new cohort.
- **Streamlit app.** Its "ICU data window" slider feeds the leaky feature and should be removed once the models are retrained.

## Data use

MIMIC-IV is credentialed PhysioNet data. This repository contains code and aggregate metrics only. It contains no patient-level rows, and notebook outputs are kept to summary statistics.
