## Authors

- Marie Lormant
- Verter Stoilov
- Viktor Petrov

EPFL — CS-433 Machine Learning

# CS-433 — Project II  
## Multiclass Failure Prediction of Carbon Fibre Composites using Machine Learning

This repository contains the code developed for **CS-433 – Machine Learning Project II**, focusing on the prediction of failure modes in carbon-fiber composite laminates using supervised machine learning models.

The objective is to identify the **first failing ply–mode pair** in a composite laminate, or to predict a **safe state**, based solely on global strain measurements.

---

## Problem Description

A multidirectional composite laminate composed of four unidirectional plies with fiber orientations  
**0°, 45°, 90°, and −45°** is considered.  
For each ply, four failure modes are evaluated:

- Fiber tension (ft)
- Fiber compression (fc)
- Matrix tension (mt)
- Matrix compression (mc)

Failure indices (FI) are computed analytically for each ply–mode combination. A sample is considered failed if at least one failure index exceeds a threshold of 1. If more than 1 FIs are above this threshold, the largest value is considered.

This results in a **17-class classification problem**:
- **16 failure classes** (4 plies × 4 modes)
- **1 safe class**

---

## Dataset

- **1,000,000 synthetic samples**
- Each sample contains:
  - 6 global strain components  
    `ε = {ε₁, ε₂, ε₃, ε₄, ε₅, ε₆}`
  - Failure indices for all ply–mode combinations

Labels are assigned by selecting the ply–mode pair with the highest failure index if `FI ≥ 1`, otherwise the sample is labeled as safe.

---

## Models Implemented

The following models were implemented and compared:

- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Support Vector Machine (SVM)**
- **Neural Network (MLP) — PyTorch**
- **Neural Network — TensorFlow/Keras**

All models were evaluated using:
- Accuracy
- Macro-averaged F1 score
- Weighted F1 score
- False Negatives (FN), with particular emphasis on safety-critical errors(ethical aspect)

---

## Training and Evaluation Protocol

- The dataset is split using an **80/20 stratified train–test split**
- The test set is kept strictly unseen until final evaluation
- Hyperparameter tuning is performed **only on the training subset**
- Class imbalance is handled via class weighting or metric-based optimization (macro-F1)

False negatives are explicitly monitored, as misclassifying a failing laminate as safe is considered a **catastrophic error**.

---

## Hyperparameter Optimization

### XGBoost
- Hyperparameters are optimized using **Optuna** with a **Tree-structured Parzen Estimator (TPE)** sampler
- Optimization objective: **maximize macro-averaged F1 score**
- Tuned parameters include:
  - Tree depth
  - Learning rate
  - Number of estimators
  - Subsampling ratios
  - Column sampling strategies
  - Regularization terms (L1, L2)
  - Minimum child weight
  - Minimum loss reduction (`gamma`)

### Neural Networks
- Tuned parameters include:
  - Number of hidden layers
  - Number of units per layer
  - Learning rate
  - Dropout rates
  - Batch size
  - Weight decay / regularization

---

## Ethical Considerations

A dedicated ethical risk analysis is conducted to address **false negatives**, defined as unsafe samples incorrectly classified as safe.  
For the Random Forest model, an adjustable decision threshold is introduced to trade overall performance for **maximum safety recall**, demonstrating that catastrophic errors can be eliminated at the cost of increased false positives.

---

## Results Summary

- All models achieve excellent overall F1-score and Accuracy
- All models show reduced performance on underrepresented failure modes (notably ±45° plies)
- Performance strongly correlates with class sample size

Detailed results and per-class metrics are available in the report appendix.

---

## Repository Structure
- dataset folder with the dataset.skl data
- xgb folder:
    - XGBoostBaselineModel.py - baseline xgboost model
    - xgboost_optuna.py - optimization code for the xbgoost model
    - optuna_xgboost_study.pkl - optimized hyperparameters
    - XGBoostModel_Optimized.py - xgboost model that trains and tests with the optimized hyperparameters
    - 
- nnTensorFlow folder:
    - nnTensorFlowBaselineModel.py - baseline Neural Networks (TensorFlow) model
    - neural_network_optimization.py - optimization code for the Neural Networsk (TensorFlow) model
    - load_best_tuner_model.py - neural networks (tensorflow) that trains and tests with the optimized hyperparameters
    - trial_0071 - trial with the smallest validation loss with the optimized hyperparameters

---

## Report

The full project report, including methodology, results, ethical analysis, and discussion, is available as:

**`ML-Project_2.pdf`**

---


