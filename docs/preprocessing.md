# 🧼 Preprocessing Pipeline for Fraud Detection

This document describes the preprocessing logic implemented in [`02_preprocessing.py`](./src/preprocessing.py), including how the data is cleaned, transformed, encoded, and split for modeling.

---

## 🎯 Objective

Prepare raw credit card transaction data for supervised fraud detection using a machine learning pipeline. This includes:

* Feature engineering
* Handling categorical variables
* Stratified train/test splitting
* Saving processed data for downstream training

---

## 📁 Input & Output

**Input:**

* Raw CSV file with 23 columns, including personal info, transaction metadata, and a binary fraud label.

**Output:**

* 4 CSV files stored in `data/processed/`:

  * `X_train.csv`
  * `X_test.csv`
  * `y_train.csv`
  * `y_test.csv`

---

## 🧱 Steps in Preprocessing

### 1. Drop Irrelevant Columns

Columns that do not help prediction or introduce privacy risks:

* `first`, `last`, `street`, `trans_num`, `unix_time`

---

### 2. Feature Engineering

#### 📅 Datetime parsing:

* `trans_date_trans_time` → extract `hour`, `dayofweek`, `month`
* `dob` → compute `age`

#### 📍 Distance computation:

* Compute Haversine distance between `(lat, long)` and `(merch_lat, merch_long)`
* Output: `distance_km`

#### 💰 Amount transformation:

* Optional log-scaling using `log1p()` on `amt` to reduce skew

---

### 3. Target Encoding (Mean Encoding)

Categorical columns:

* `category`, `merchant`, `job`, `state`, `gender`

These are encoded using **TargetEncoder**, based on training data only.

---

### 4. Stratified Train/Test Split

* Ensures the same fraud ratio in both train and test sets.
* Default split is 80% train / 20% test.

---

## 🧪 Unit Testing with `pytest`

Test coverage includes:

| Test Name                                     | Description                                                             |
| --------------------------------------------- | ----------------------------------------------------------------------- |
| `test_preprocess_structure`                   | Validates output columns, datatypes, and dropped fields                 |
| `test_haversine_output`                       | Checks distance calculation is functional and non-zero                  |
| `test_encoding_output`                        | Runs full pipeline with mocked dataset, confirms all 4 CSVs are created |
| `test_stratified_split_preserves_class_ratio` | Verifies class balance in train/test is preserved (±10%)                |

Tests are located in:

```
tests/test_preprocessing.py
```

To run the tests:

```bash
pytest tests/test_preprocessing.py
```

You should see:

```bash
tests/test_preprocessing.py ....    # All tests pass
```

---

Voici une **section additionnelle** à intégrer dans ton `preprocessing.md` pour documenter le script `check_preproc_data.py`.

---

## 🔍 Post-Preprocessing Sanity Check

After generating your preprocessed files, you can run a standalone check using the utility script [`check_preproc_data.py`](./src/check_preproc_data.py). This helps ensure data quality before moving to training.

### ✅ What It Does

The script performs key diagnostics on the preprocessed datasets:

* Confirms dataset presence and shapes
* Inspects column types
* Detects non-numeric columns (CatBoost will fail on these)
* Flags high-cardinality categorical variables
* Checks class imbalance in the target `is_fraud`

---

### 🚀 How to Run

```bash
python check_preproc_data.py
```

Optional: target a specific version by timestamp:

```bash
python check_preproc_data.py --timestamp 20240612_1120
```

---

### 🧪 What You'll See

* ✅ Data shape and positive class count
* 🧬 Column types and any unexpected `object` types
* ⚠️ Warnings for high-cardinality categorical fields (e.g., `city`)
* 📊 Distribution of fraud vs. non-fraud across train/test splits

---

### 💡 When to Use It

* Before training, especially after a schema or encoding change
* To debug issues with non-numeric features (e.g., CatBoost errors)
* To validate reproducibility between runs (via `--timestamp`)

---

## 📌 Notes

* This script is agnostic to the downstream ML framework (e.g. CatBoost, XGBoost).
* Designed to be modular and easily plugged into a larger ML pipeline.