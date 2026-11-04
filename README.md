# 🫘 Dry Bean Variety Classification – ML Assignment 2

Five classical machine-learning classifiers trained on the UCI Dry Bean dataset, compared on six evaluation metrics, and served through an interactive Streamlit app.

- **Live app:** `<https://mlassignment2-eq2hfkz4w3f8s5s8ugyhov.streamlit.app/>`
- **GitHub repository:** <https://github.com/SaskahmNishantMalge/ML_Assignment_2>

---

## a. Problem statement

Dry beans are the most widely produced pulse crop, and seed quality largely decides crop yield. Separating beans of different varieties by hand is slow and error-prone. Given **16 shape measurements** extracted from an image of a single bean (area, perimeter, axis lengths, roundness, shape factors …), the goal is to **predict which of 7 registered varieties the bean belongs to**.

This is a **multi-class classification** problem. Five models – Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and an RBF Support Vector Machine – are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC).

## b. Dataset description

| Property | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository (dataset id 602) – <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> |
| Reference | Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507. |
| Instances | **13,611** (13,543 after removing 68 exact duplicate rows) – requirement ≥ 500 ✅ |
| Features | **16** numeric features – requirement ≥ 12 ✅ |
| Target | `Class` – 7 varieties |
| Missing values | None |

**Features**

- *12 dimensional features:* Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness
- *4 shape factors:* ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

**Class distribution (imbalanced)**

| Class | Count | Share |
|---|---|---|
| DERMASON | 3,546 | 26.1 % |
| SIRA | 2,636 | 19.4 % |
| SEKER | 2,027 | 14.9 % |
| HOROZ | 1,928 | 14.2 % |
| CALI | 1,630 | 12.0 % |
| BARBUNYA | 1,322 | 9.7 % |
| BOMBAY | 522 | 3.8 % |

**Key characteristics that shaped the modelling**

- Features are on very different scales (Area ≈ 20,000–250,000 vs ShapeFactor2 ≈ 0.0006–0.004), so Logistic Regression, KNN and SVM are trained on standardised features.
- 15 feature pairs have |correlation| > 0.95 (e.g. Area ↔ ConvexArea = 1.00), which violates the independence assumption of Naive Bayes.
- Because the classes are imbalanced, the split is **stratified** and Precision / Recall / F1 are **macro-averaged** (every variety counts equally). AUC is one-vs-rest, macro-averaged.

## c. GitHub repository link

<https://github.com/SaskahmNishantMalge/ML_Assignment_2>

```
ML_Assignment_2/
├── app.py                        # Streamlit app (loads saved models, never retrains)
├── requirements.txt              # pinned dependencies
├── README.md
├── test_data.csv                 # held-out 20 % test split (2,709 rows, 16 features + Class)
├── data/
│   └── Dry_Bean_Dataset.csv      # full raw dataset (CSV copy of the UCI .xlsx)
├── notebooks/
│   ├── 01_data_exploration.ipynb # Step 1 – loading, data quality, class balance, correlations
│   └── ModelTraining.ipynb       # Step 2 – training, tuning, evaluation and saving of all 5 models
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.json              # test-set metrics + tuned hyperparameters
    └── feature_info.json         # feature order and class labels used by the app
```

## d. Models used

**Training setup**

- Duplicates removed, then a stratified 80 / 20 train–test split (`random_state = 42`) → 10,834 training rows, 2,709 test rows.
- Each model is a scikit-learn `Pipeline` (scaler + classifier where scaling is needed), tuned with 5-fold stratified `GridSearchCV` on the training split only, using macro F1 as the selection score.
- All metrics below are computed on the untouched 20 % test split.

| Model | Pre-processing | Tuned hyperparameters |
|---|---|---|
| Logistic Regression | StandardScaler | C = 500 |
| Decision Tree | – | criterion = entropy, max_depth = 16, min_samples_leaf = 10 |
| KNN | StandardScaler | n_neighbors = 21, weights = distance |
| Naive Bayes (Gaussian) | – | var_smoothing = 1e-9 |
| Support Vector Machine (RBF) | StandardScaler | C = 10, gamma = scale |

### Comparison table (test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9177 | **0.9948** | 0.9296 | 0.9296 | 0.9295 | 0.9005 |
| Decision Tree | 0.9011 | 0.9740 | 0.9138 | 0.9124 | 0.9130 | 0.8803 |
| KNN | 0.9192 | 0.9912 | 0.9334 | 0.9294 | 0.9309 | 0.9023 |
| Naive Bayes | 0.7630 | 0.9670 | 0.7736 | 0.7685 | 0.7669 | 0.7143 |
| Support Vector Machine | **0.9236** | 0.9944 | **0.9356** | **0.9340** | **0.9347** | **0.9076** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong for a linear model: 91.8% accuracy, MCC 0.90, and the highest AUC (0.9948), so its probability rankings are the best of the five. It chose very weak regularisation (C = 500), meaning the scaled shape features are almost linearly separable. Train accuracy is only about 1% above test accuracy, so no overfitting. Most errors are DERMASON and SIRA being mistaken for each other. |
| Decision Tree | Lowest of the four strong models: 90.1% accuracy, F1 0.913, MCC 0.880. AUC is lower (0.974) because a tree's leaves give coarse probabilities. It has the largest train/test gap (94.7% vs 90.1%), so it overfits slightly even with max_depth = 16 and min_samples_leaf = 10. Splitting on one feature at a time handles curved class boundaries less well. Weakest on SIRA (recall 0.843). |
| KNN | Close to the best: 91.9% accuracy, F1 0.931, and the best CALI recall (0.963). Best setting: 21 neighbours, with closer neighbours counting more. Training accuracy shows 100% only because, with distance weighting, each training point is its own nearest neighbour; the test score is the honest one. Works well only because features were standardised; otherwise Area would dominate the distances. |
| Naive Bayes | Clearly the worst: 76.3% accuracy, MCC 0.714, and about 3× more errors than the other models (642). It assumes features are independent, but 15 feature pairs are correlated above 0.95, so the same information is counted several times. BARBUNYA recall is only 0.46 (108 BARBUNYA beans were called CALI). AUC is still 0.967, so the right class is often its second-highest guess. Train and test accuracy are equal, so the problem is the model's assumptions, not overfitting. |
| Support Vector Machine | Best on 5 of the 6 metrics: Accuracy 0.9236, Precision 0.9356, Recall 0.9340, F1 0.9347, MCC 0.9076. Its AUC (0.9944) is a close second. It also had the best cross-validation F1 (0.9438) and the fewest test errors (207). The RBF kernel draws curved boundaries, which helps on the hard DERMASON/SIRA pair. Small train/test gap (94.0% vs 92.4%). |
| **Overall winner** | **SVM**, by a small margin over Logistic Regression and KNN (all within 0.6% accuracy). The main difficulty in this dataset is telling DERMASON from SIRA, which every model confuses. BOMBAY is identified almost perfectly by every model because those beans are much larger than the rest. If training speed mattered, Logistic Regression is a close and cheaper alternative with the best AUC. |

---

## Streamlit app

The app loads the five saved pipelines from `model/` and **only calls `predict` / `predict_proba`** – nothing is retrained on the uploaded data.

- **CSV upload** – upload a test file with the 16 feature columns and `Class` (the bundled `test_data.csv` is used until a file is uploaded, and can be downloaded from the sidebar).
- **Model selection dropdown** – choose any of the five classifiers.
- **Evaluation metrics** – Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model.
- **Confusion matrix and classification report** – per-variety precision, recall and F1.
- **Compare all models** – all five models scored on the same file, with a grouped bar chart.
- **Predictions** – row-level predictions with confidence, a misclassified-only filter and CSV download.
- Uploaded files are validated (missing columns, missing target, non-numeric or empty cells).

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models, open `notebooks/ModelTraining.ipynb` and run all cells – it rewrites `model/*.joblib`, `model/metrics.json` and `test_data.csv`.

> The saved models were trained on the BITS Virtual Lab with **Python 3.12 and scikit-learn 1.6.1**. On Streamlit Community Cloud, choose **Python 3.12** under *Advanced settings* (the pinned numpy version has no Python 3.13 wheels).
