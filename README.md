# Breast Cancer Treatment Response Prediction

## Project Overview

This project applies machine learning to predict breast cancer treatment response using clinical biomarkers and MRI-derived radiomics features. The project contains two related prediction tasks:

1. **pCR Classification** — predicting whether a patient achieves pathological complete response after neoadjuvant chemotherapy.
2. **RFS Regression** — predicting recurrence-free survival time.

The aim was to explore whether combining clinical information with MRI-derived radiomic features can support better pre-treatment patient stratification.

---

## Problem Context

Neoadjuvant chemotherapy is commonly used for locally advanced breast cancer, but not every patient responds equally well. Being able to predict treatment response before chemotherapy may help improve patient stratification and support more personalised treatment planning.

This project uses multimodal machine learning with:

* Clinical biomarkers
* MRI-derived radiomics features
* Classification models for pCR prediction
* Regression models for RFS estimation

---

## Dataset

The assignment dataset is based on a simplified version of the public I-SPY 2 breast cancer trial data.

The training dataset contained:

* 400 patients
* 11 clinical features
* 107 MRI-derived radiomics features
* Two target variables:

  * `pCR (outcome)` for classification
  * `RelapseFreeSurvival (outcome)` for regression

Missing values were represented using `999` and were handled during preprocessing.

---

## Methodology

### 1. Data Preprocessing

The preprocessing pipeline included:

* Replacing `999` values with missing values
* Median imputation for continuous clinical variables
* Mode imputation for categorical variables
* Gene missingness indicator creation
* Label encoding for categorical variables
* Log transformation and standardisation for skewed radiomics features
* Train-validation splitting with fixed random seed

---

### 2. Feature Selection

Feature selection was performed separately for the two tasks.

For **pCR classification**, several feature selection methods were compared:

* ANOVA F-test
* Chi-squared test
* Random Forest feature importance
* Recursive Feature Elimination

The final pCR model used a compact 5-feature subset, retaining the mandatory clinical biomarkers ER, HER2, and Gene.

For **RFS regression**, Sequential Feature Selection and permutation importance were used. The final RFS model used 23 features, including selected radiomic features and the mandatory clinical biomarkers.

---

### 3. Model Development

For **pCR classification**, models explored included:

* Logistic Regression
* Random Forest
* XGBoost
* Support Vector Machine

Class imbalance was handled using SMOTE-Tomek resampling during model development.

For **RFS regression**, models explored included:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Support Vector Regression
* Random Forest Regression
* Gradient Boosting Regression
* Ensemble Regression

---

## Results

### pCR Classification

The final selected pCR model was an **XGBoost classifier** trained using the selected 5-feature subset.

**Final pCR performance:**

* Balanced Accuracy: **0.825**
* Precision: **0.770**
* Recall: **0.919**
* F1-score: **0.838**

---

### RFS Regression

The final selected RFS model was an ensemble combining Random Forest, Gradient Boosting, and Ridge Regression.

**Final RFS performance:**

* Mean Absolute Error: **18.36**

---

## Key Takeaways

* Combining clinical biomarkers with MRI-derived radiomics features improved treatment response modelling.
* A compact feature subset was effective for pCR classification, supporting interpretability and reducing dimensionality.
* Tree-based and ensemble models performed better than simpler linear baselines.
* Feature selection was important due to the high dimensionality of radiomics data compared with the number of patients.
* The project demonstrates an end-to-end machine learning workflow for healthcare prediction tasks.

---

## Repository Structure

```text
breast-cancer-treatment-response-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── pcr_classification_model_development.ipynb
│   ├── rfs_regression_model_development.ipynb
│   └── FinalTestRFS.ipynb
│
├── src/
│   └── FinalTestPCR.py
│
├── reports/
│   └── breast_cancer_ml_report.pdf
│
├── images/
│   ├── pipeline_overview.png
│   ├── feature_selection_comparison.png
│   ├── sfs_mae_curve.png
│   └── final_results_table.png
│
├── models/
│   └── README.md
│
└── sample_outputs/
    ├── PCRPrediction_sample.csv
    └── RFSPrediction_sample.csv
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/umrahzargar/breast-cancer-treatment-response-prediction.git
```

2. Navigate to the project folder:

```bash
cd breast-cancer-treatment-response-prediction
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebooks:

```bash
jupyter notebook
```

---

## Notes

The original assignment dataset and trained model artifacts may not be included in this repository due to data access and coursework restrictions. The repository focuses on documenting the methodology, modelling workflow, evaluation approach, and prediction pipeline structure.

---

## Author

**Umrah**

* GitHub: [github.com/umrahzargar](https://github.com/umrahzargar)
* LinkedIn: [linkedin.com/in/umrah-zargar](https://www.linkedin.com/in/umrah-zargar)
