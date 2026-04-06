# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions on a highly imbalanced real-world dataset.

---

## 📌 Problem Statement

Credit card fraud is a critical issue in the financial industry. This project addresses the challenge of detecting fraudulent transactions in a dataset where fraud cases represent less than **0.2%** of total records — making it a classic **extreme class imbalance problem**.

The dataset is sourced from Kaggle (ULB Machine Learning Group) and contains transactions made by European cardholders in September 2013.  

To ensure privacy, all sensitive features have been transformed using **Principal Component Analysis (PCA)**. Only **Time**, **Amount**, and the target variable **Class** remain in their original form.

---

## 📂 Dataset

- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
- **Total transactions:** 284,807  
- **Fraud cases:** 492 (~0.172%)  
- **Features:** 30 (V1–V28 via PCA + Time + Amount)  
- **Target:** `Class`  
  - 0 → Legitimate  
  - 1 → Fraud  

> ⚠️ All sensitive data is anonymized using PCA. No personal data is exposed.

---

##  Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed class imbalance (99.8% vs 0.2%)  
- Visualized transaction distributions (Time & Amount)  
- Checked correlations and outliers  

---

### 2. Data Preprocessing
- Scaled **Time** and **Amount** using `StandardScaler`  
- Performed train-test split with `stratify=y`  
- Applied **SMOTE** only on training data to avoid data leakage  

---

### 3. Models Used
- Logistic Regression (`class_weight='balanced'`)  
- Decision Tree (with depth tuning)  
- XGBoost (`scale_pos_weight` for imbalance handling)  

---

### 4. Evaluation Metrics

Since accuracy is misleading for imbalanced datasets, we used:

- **Precision** → Correct fraud predictions  
- **Recall** → Fraud detection rate  
- **F1-score** → Balance of precision & recall  
- **PR-AUC** → Best metric for imbalanced problems  

---

##  Results (Final Model: XGBoost)

| Metric    | Score |
|----------|------|
| Precision | 0.72 |
| Recall    | 0.85 |
| F1-score  | 0.78 |
| PR-AUC    | 0.85 |

>  High recall (0.85) is prioritized, as missing fraud is more costly than false alarms.

---

##  Conclusion

XGBoost combined with **SMOTE** and **class imbalance handling** delivered the best results.

The model:
- Detects **85% of fraud cases**
- Maintains a strong balance between precision and recall  

### Key Takeaways:
- Accuracy is misleading for imbalanced datasets  
- Always use **F1-score and PR-AUC**  
- Apply SMOTE only on training data  
- XGBoost handles imbalance effectively  

---

##  Tech Stack

- Python 3.x  
- Scikit-learn  
- XGBoost  
- Imbalanced-learn (SMOTE)  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## 📁 Project Structure
credit-card-fraud-detection/
│
├── data/
├── credit_card_fraud_detection.ipynb
├── requirements.txt
└── README.md


---

## ⚠️ Dataset Note

The dataset is not included due to Kaggle’s terms of use.

### Download manually:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  

### Or via Colab:

```python
!kaggle datasets download -d mlg-ulb/creditcardfraud
!unzip creditcardfraud.zip

🙏 Acknowledgements
Special thanks to Tek-nr whose project AI-Based-Fraud-Detection served as a valuable reference and source of inspiration for this work.
you can check out his work also:
👉 https://github.com/Tek-nr/AI-Based-Fraud-Detection/blob/main/creditcard/creditcard%20-%20with%20feature%20selection%20(corr).ipynb
