# Cost-Sensitive-Income-Classification
Klasifikasi income menggunakan XGBoost dengan optimasi threshold dan analisis expected loss untuk mendukung pengambilan keputusan kredit berbasis risiko.


# Cost-Sensitive Income Classification

### Risk-Based Lending Use Case

## 1. Problem Framing

Objective:
Membangun model klasifikasi untuk memprediksi **income >50K** sebagai proxy kemampuan bayar dalam konteks underwriting.

Business constraint:

* False Positive (approve high-risk borrower) lebih mahal dibanding False Negative.
* Evaluation metric harus memprioritaskan **precision**.

Formulasi:
Binary classification dengan cost asymmetry.

---

## 2. Dataset

* Source: Adult Income Dataset (Kaggle)
* Observations: 48,841 → 48,808 (setelah drop duplicate)
* Features: 13
* Target distribution:

  * > 50K: ~24%
  * ≤50K: ~76%
* Imbalanced classification problem

Target encoding:

```
income >50K  → 1
income ≤50K  → 0
```

---

## 3. Exploratory Data Analysis (EDA)

Findings utama:

* Income >50K berkorelasi positif dengan:

  * Age
  * Hours-per-week
  * Education level
* Strong categorical predictors:

  * Relationship
  * Marital Status
  * Occupation
* Gender menunjukkan gap signifikan (perlu awareness terkait fairness)

EDA menunjukkan bahwa problem memiliki separability yang cukup baik.

---

## 4. Preprocessing Pipeline

Pipeline berbasis `ColumnTransformer`:

### Numerical Features

* RobustScaler
  (lebih robust terhadap outlier dibanding StandardScaler)

### Low Cardinality Categorical

* OneHotEncoding

### High Cardinality Categorical

* Binary Encoding
  (mengurangi dimensionality explosion)

Split:

* Train/Test = 80/20
* Stratified split

---

## 5. Model Selection Strategy

### Evaluation Metric

Menggunakan **F0.5-score**

Alasan:

* Fβ dengan β=0.5 memberikan bobot lebih tinggi pada precision.
* Selaras dengan business objective (minimize False Positive).

### Cross Validation

* Stratified K-Fold
* Metric: mean F0.5

### Model Candidates

* Logistic Regression
* Random Forest
* AdaBoost
* Stacking
* XGBoost
* Resampling strategies:

  * SMOTE
  * RandomUnderSampler
  * NearMiss

---

## 6. Model Performance

Best Model:

> **XGBoost tanpa resampling**

Cross-validation results:

* Mean F0.5 = 0.693
* Std Dev = 0.0038

Insight:

* Resampling tidak meningkatkan F0.5.
* UnderSampling menurunkan performa signifikan.
* XGBoost mampu menangani class imbalance secara internal (tree-based gradient boosting + weighted splits).

---

## 7. Hyperparameter Tuning

Method:

* RandomizedSearchCV

Best parameters:

```
max_depth = 5
min_child_weight = 5
learning_rate = 0.0595
n_estimators = 185
gamma = 2.098
subsample = 0.606
colsample_bytree = 0.845
```

Best CV F0.5 = 0.696

Model complexity dikontrol untuk menghindari overfitting.

---

## 8. Threshold Optimization

Default threshold: 0.5
Optimized threshold: 0.5283

Results:

| Metric | Default | Optimized |
| ------ | ------- | --------- |
| F0.5   | 0.6985  | 0.7017    |

Dampak:

* False Positive ↓
* False Negative ↑
* Model menjadi lebih konservatif

Ini menunjukkan bahwa threshold tuning lebih berdampak daripada resampling dalam konteks cost-sensitive problem.

---

## 9. Discriminative Power

* ROC-AUC = 0.903
* PR-AUC = 0.753
* Baseline PR-AUC = 0.24

Interpretation:

* Strong class separability
* PR-AUC jauh di atas baseline → bukan sekadar bias kelas mayoritas

---

## 10. Cost-Based Evaluation

Confusion Matrix (Optimal Threshold):

* FP = 427
* FN = 1,043

Assumed Cost:

* Cost FP = Rp36M
* Cost FN = Rp14.4M

Expected Loss:

* FP Loss = Rp15.37B
* FN Loss = Rp15.02B
* Total = Rp30.39B

Key insight:
Model evaluation diterjemahkan langsung ke expected financial impact, bukan hanya metric teknikal.

---

## 11. Feature Importance

Top Features:

* Relationship
* Marital Status
* Education
* Hours-per-week
* Occupation

Model cenderung menangkap sinyal stabilitas ekonomi rumah tangga sebagai proxy income.

---

## 12. Technical Takeaways

* F0.5 lebih relevan dibanding accuracy/ROC untuk problem ini.
* Threshold tuning > resampling untuk cost-sensitive scenario.
* Gradient boosting efektif pada imbalanced tabular data.
* Translasi confusion matrix ke expected loss meningkatkan business interpretability.
* Perlu eksplorasi fairness (gender & race sebagai sensitive features).

---

## 13. Possible Improvements

* Cost-sensitive training (custom loss function)
* Class weight tuning
* SHAP for model interpretability
* Fairness analysis (Equal Opportunity / Demographic Parity)
* Probability calibration (Platt / Isotonic)
* Bayesian optimization untuk hyperparameter search

