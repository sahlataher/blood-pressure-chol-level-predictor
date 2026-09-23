# Blood Pressure and Cholesterol Level Predictor

A machine learning project that predicts blood pressure and cholesterol classification from health and demographic data in the **NHANES** (National Health and Nutrition Examination Survey) dataset, built by Team Purple as part of a class competition against 9 teams.

## Overview

This project builds automated preprocessing pipelines and trains/tunes a wide range of classification models to predict:
- **Blood pressure status**: `normal` vs. `abnormal` (derived from systolic/diastolic readings)
- **Cholesterol status**: `normal`, `borderline`, or `high` (derived from total cholesterol level)

The final models ranked **1st in blood pressure prediction accuracy and F1 macro score**, and **1st in cholesterol accuracy / 2nd in F1 macro score**, among 9 competing teams.

## Approach

1. **Data loading & labeling** — Loaded the NHANES dataset and derived target labels for BP (`normal`/`abnormal`) and cholesterol (`normal`/`borderline`/`high`) from raw clinical measurements.
2. **Train/test split** — 80/20 split for both the BP and cholesterol prediction tasks.
3. **Preprocessing pipeline** — Built with `ColumnTransformer`, including one-hot encoding for categorical features, standard scaling for numeric features, and imputation for missing values.
4. **Hyperparameter tuning** — Used `GridSearchCV` and `RandomizedSearchCV` (10-fold CV) to tune each model, separately optimizing for accuracy and for F1 macro score.
5. **Model comparison** — Evaluated 11 base models plus 3 ensemble methods via 10-fold cross-validation, scoring accuracy, precision (macro), recall (macro), and F1 (macro).
6. **Final model selection & deployment** — Best-performing model serialized with `cloudpickle` and uploaded to the HuggingFace Hub.

## Models Evaluated

14 models were trained, tuned, and compared via cross-validation:
- Decision Tree (stump, depth-limited)
- Decision Tree (deeper)
- Random Forest
- AdaBoost
- SVM
- Gaussian Naive Bayes
- Logistic Regression
- K-Nearest Neighbors
- Multi-Layer Perceptron (MLP)
- Gradient Boosting
- Histogram-Based Gradient Boosting
- Voting Classifier (hard voting)
- Voting Classifier (soft voting)
- Stacking Classifier (logistic regression meta-learner)

## Tech Stack

- **Language**: Python
- **Libraries**: scikit-learn, pandas, NumPy
- **Model serialization**: cloudpickle
- **Model hosting**: HuggingFace Hub

## Results

| Metric | Result |
|---|---|
| Blood Pressure Accuracy | 🥇 1st of 9 teams |
| Blood Pressure F1 Macro | 🥇 1st of 9 teams |
| Cholesterol Accuracy | 🥇 1st of 9 teams |
| Cholesterol F1 Macro | 🥈 2nd of 9 teams |

## Getting Started

### Prerequisites

```bash
pip install scikit-learn pandas numpy cloudpickle huggingface_hub
```

### Usage

```bash
git clone https://github.com/sahlataher/blood-pressure-chol-level-predictor.git
cd blood-pressure-chol-level-predictor
```

Open the notebook and run cells in order:
1. Load `nhanes.csv` into the working directory
2. Run data loading, labeling, and splitting cells
3. Run preprocessing pipeline and model training cells
4. Run grid/random search cells to reproduce tuning
5. Final model is saved as `model.pkl` and can be uploaded to HuggingFace Hub

## Team

Built by Team Purple- Aisha Malik (aisha1021) and Sahla Taher (sahlataher) for CSCI 35300 Machine Learning.
