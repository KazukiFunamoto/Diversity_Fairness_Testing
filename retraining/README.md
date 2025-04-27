# Retraining Experiments

This folder is for running retraining experiments and saving results.

## Structure

- `diversity_retrain.py`: Investigates how changes in diversity levels affect fairness improvement and accuracy degradation.
- `retrain_methods.py`: Investigates how different retraining methods affect fairness improvement and accuracy degradation.
- `diversity_retrain_results/`: Stores the results of `diversity_retrain.py`.
- `retrain_methods_results/`: Stores the results of `retrain_methods.py`.
- `classifier/datasets/`: Contains the datasets used in all experiments when trainig classifiers.

## How to Run

### diversity_retrain.py

You can run it by:

```
python diversity_retrain.py CENSUS_age SVM
```
Arguments:
- Dataset and protected attribute (e.g., CENSUS_age)
- Model (e.g., SVM)

The results will be stored under `diversity_retrain_results/`.

The tasks are combinations of:

- **Datasets and protected attributes**:
  - `CENSUS_age`
  - `CENSUS_race`
  - `CENSUS_sex`
  - `BANK_age`
  - `GERMAN_age`
  - `GERMAN_sex`

- **Models**:
  - `SVM`
  - `MLPC`
  - `RF`

Each combination of dataset/protected attribute and model is tested, leading to 6 × 3 = 18 tasks.

You can run all tasks automatically using `run_all_diversity_retrain.py`, which runs each task 30 times.  
The corresponding results are stored in `all_results/RQ_RQ1/`.

---

### retrain_methods.py

You can run it by:

```
python retrain_methods.py const_const CENSUS_age SVM
```
Arguments:
- Retraining method (e.g., const_const, random_majority, etc.)
- Dataset and protected attribute
- Model

Alternatively, you can specify `all` as the retraining method to compare all 9 retraining methods:

```
python retrain_methods.py all CENSUS_age SVM
```

The results will be stored under `retrain_methods_results/`.

The tasks are the same combinations as above:

- **Datasets and protected attributes**:
  - `CENSUS_age`
  - `CENSUS_race`
  - `CENSUS_sex`
  - `BANK_age`
  - `GERMAN_age`
  - `GERMAN_sex`

- **Models**:
  - `SVM`
  - `MLPC`
  - `RF`

Each combination of dataset/protected attribute and model is tested, leading to 6 × 3 = 18 tasks.

There are nine retraining methods, listed as follows:
- `const_const`
- `const_random`
- `const_majority`
- `random_const`
- `random_random`
- `random_majority`
- `pair_const`
- `pair_random`
- `pair_majority`

You can run all tasks automatically using `run_all_retrain_methods.py`, which runs each retraining method for each task 30 times.  
The corresponding results are stored in `all_results/RQ3/`.

**Note:**  
Both `diversity_retrain.py` and `retrain_methods.py` use RSUTT as the fairness testing algorithm to detect discriminatory data. RSUTT includes a hyperparameter \(N\) that controls the number of global search iterations.  To ensure that the number of detected IDIs exceeds the size of the training dataset for every task, we automatically configure \(N\) based on the dataset, protected attribute, and model. The assigned values of \(N\) for each of the 18 scenarios are as follows:

| Dataset-Attribute | Model | N |
|:-------------------|:------|:--|
| CENSUS_age         | SVM   | 1000 |
| CENSUS_age         | MLPC  | 1000 |
| CENSUS_age         | RF    | 1000 |
| CENSUS_race        | SVM   | 10000 |
| CENSUS_race        | MLPC  | 10000 |
| CENSUS_race        | RF    | 1000 |
| CENSUS_sex         | SVM   | 100000 |
| CENSUS_sex         | MLPC  | 10000 |
| CENSUS_sex         | RF    | 1000 |
| BANK_age           | SVM   | 10000 |
| BANK_age           | MLPC  | 10000 |
| BANK_age           | RF    | 1000 |
| GERMAN_age         | SVM   | 1000 |
| GERMAN_age         | MLPC  | 1000 |
| GERMAN_age         | RF    | 1000 |
| GERMAN_sex         | SVM   | 10000 |
| GERMAN_sex         | MLPC  | 1000 |
| GERMAN_sex         | RF    | 1000 |


