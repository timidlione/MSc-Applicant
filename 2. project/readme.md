# Crab Age Regression (CatBoost + Optuna)

This repository contains **my individual contribution** to a team competition project.  
I was responsible for **model training, evaluation metric design, and iterative performance improvement**.  
Accordingly, this repo includes **only my part (training script)**; other team assets (domain EDA, deployment code, etc.) are **not included**.
 
The script predicts **Age** (continuous target) from crab biometric features using **CatBoostRegressor** with **Optuna** hyperparameter optimization.  
It loads CSV files, runs a reproducible train/validation split, searches hyperparameters to minimize **MAE**, trains the final model with early stopping, evaluates on the validation set, and creates a **submission CSV** for the test set.

**Reproducibility:** the train/validation split uses `random_state=42`. If you need fully reproducible HPO, see the Tips section (Optuna sampler seed).

---

## 1. Project Structure

> Dataset files are referenced as external inputs and are **not committed**.  
> We provide a sample submission and a cropped leaderboard screenshot for transparency.

```

.
├─ main.py
├─ requirements.txt
├─ README.md
├─ submissions/
│  └─ submission_example.csv
├─ assets/
│  └─ leaderboard/
│     └─ final.png
└─ data/ (not committed)
├─ train.csv
├─ test.csv
└─ sample_submission.csv

````

> **Note on paths:** The code by default reads `kaggle/train.csv` and `kaggle/test.csv`.  
> You can either place files in `kaggle/`, or adjust paths in `main.py` to use `data/`.

---

## 2. Environment

- **OS/Hardware:** macOS (Apple Silicon **M3 Pro**)
- **IDE:** Work was done in **PyCharm** (open the project → run `main.py`)
- **Frameworks:** CatBoost, Optuna, scikit-learn, pandas, NumPy, Matplotlib/Seaborn

---

## 3. How to Run

1) Place the CSV files:
   - Default in code: `kaggle/train.csv`, `kaggle/test.csv`  
   - Or modify paths in `main.py` to `data/train.csv`, `data/test.csv`

2) Install dependencies:
```bash
pip install -r requirements.txt
````

3. Run:

```bash
python main.py
```

**Pipeline overview**

* Load `train.csv`/`test.csv`, preserve `id` for submission, and remove it from features.
* Declare `Sex` as categorical and pass via `cat_features` to CatBoost.
* Split training data into **train/validation (80/20, random_state=42)**.
* Run **Optuna** (50 trials) to minimize **validation MAE** with early stopping during search.
* Train a final **CatBoostRegressor** using `study.best_params` (with early stopping).
* Evaluate on validation set (prints **MAE**).
* Predict on `test.csv` and save `submission(최종).csv`.

---

## 4. Data

* `train.csv` — training set, target column: **Age**
* `test.csv` — test set; predict **Age** (int or float acceptable)
* `sample_submission.csv` — required submission format reference

> Data files are **not** included in this repository per competition rules. Only paths and format are documented.

---

## 5. Hyperparameters

**Optuna search space**

* `iterations ∈ [300, 1500]`
* `depth ∈ [4, 12]`
* `learning_rate ∈ [0.001, 0.1]`
* `l2_leaf_reg ∈ [1e-4, 10.0]`
* `bagging_temperature ∈ [0.0, 1.0]`

**Objective:** minimize **MAE** on the validation set (lower is better).
**Early stopping (search):** `early_stopping_rounds=300`
**Final training:** `study.best_params` + `loss_function='MAE'`, `random_seed=42`, `early_stopping_rounds=100`

---

## 6. Tips

* **Fully reproducible HPO (optional)**

  ```python
  from optuna.samplers import TPESampler
  study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
  ```
* **Prediction range (optional)**
  If the task defines a hard Age range (e.g., `[1, 20]`), you may clip predictions:

  ```python
  test_predictions = np.clip(test_predictions, 1, 20)
  ```

  Apply the **same rule on validation** when reporting MAE to keep evaluation consistent.

---

## 7. Competition Results

**Leaderboard (Final / Private): 4th place — MAE 1.34911**
Date: 2025-02-08
Competition: 코드잇 부스트 데모데이

<p align="center">
  <img src="assets/leaderboard/final.png" alt="Final Leaderboard (4th place, Private MAE 1.34911)" width="680">
</p>

### 7.1 Validation Metric (local)

| Model Variant                   | Validation MAE |
| ------------------------------- | -------------- |
| CatBoost + Optuna (best params) | 1.34911        |

### 7.2 Submission

* Example submission file (competition format):
  [`submissions/submission_example.csv`](submissions/submission_example.csv)

> **Compliance note**
>
> * This repository was prepared **after the competition ended**.
> * We **do not** redistribute competition data or reveal test labels.
> * The leaderboard image is **cropped** to show only our team row; other teams’ identifiers are not exposed.
> * The submission file is provided **only** as a format example.

---

## 8. Citation

If you use this baseline or ideas from it, please cite this repository.

```bibtex
@software{crab_age_regression_catboost_optuna,
  title   = {Crab Age Regression: CatBoost + Optuna Baseline},
  author  = {Sangmin Woo and Team Members},
  year    = {2025},
  url     = {https://github.com/timidlione/MSc-Applicant/tree/main }
}
```

```
```
