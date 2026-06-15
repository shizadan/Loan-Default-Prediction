# 🏦 Loan Default Prediction

A machine learning project that predicts whether a loan applicant is likely to default, based on financial and demographic features.

Built as part of the **Week 3 Assignment – AI & Data Science Bootcamp**.

---

## 📌 Project Overview

Loan default is a major risk for financial institutions. This project uses supervised machine learning to classify applicants as likely to default or not, based on features like credit score, income, debt-to-income ratio, and employment history.

The dataset is highly imbalanced (~12% default rate), so the project emphasizes proper evaluation metrics (F1 Score) over raw accuracy.

---

## 📂 Repository Structure

```
loan-default-prediction/
│
├── Loan_Default_Prediction.ipynb   # Full ML notebook (EDA → Modeling → Evaluation)
├── Loan_default.csv                # Dataset (255,347 rows, 18 columns)
└── README.md                       # Project documentation
```

---

## 📊 Dataset

- **Source:** [Kaggle – Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- **Size:** 255,347 rows × 18 columns
- **Target Variable:** `Default` (0 = No Default, 1 = Default)

**Key Features:**

| Feature | Description |
|---|---|
| CreditScore | Applicant's credit score |
| Income | Annual income ($) |
| LoanAmount | Requested loan amount ($) |
| InterestRate | Loan interest rate (%) |
| DTIRatio | Debt-to-income ratio |
| MonthsEmployed | Duration of current employment |
| LoanPurpose | Purpose of the loan |
| Education | Highest education level |

---

## 🔧 Workflow

1. **Data Loading & Exploration** – Shape, types, null checks, default rate
2. **Exploratory Data Analysis (EDA)** – Distribution plots, correlation heatmap, class imbalance check
3. **Preprocessing** – One-hot encoding of categorical features, stratified train/test split (80/20)
4. **Model Training** – Random Forest, Logistic Regression, XGBoost (with class imbalance handling)
5. **Evaluation** – Accuracy, F1 Score, Classification Report

---

## 📈 Results

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 0.679 | 0.335 |
| Random Forest | 0.885 | 0.026 |
| XGBoost | 0.714 | 0.337 |

> XGBoost achieved the best F1 score, making it the most effective model for identifying loan defaulters in this imbalanced dataset.

> *Note: Random Forest shows high accuracy but low F1, indicating it favours the majority class (non-defaulters). XGBoost and Logistic Regression with class balancing are better suited for this problem.*

---

## ⚙️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/shizadan/loan-default-prediction.git
cd loan-default-prediction
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Launch the notebook:
```bash
jupyter notebook Loan_Default_Prediction.ipynb
```

---

## 🧠 Key Learnings

- **Imbalanced classification** requires F1 Score over accuracy as the primary metric
- **Stratified splitting** ensures both train and test sets reflect the real default rate
- **class_weight='balanced'** and **scale_pos_weight** help models pay attention to minority class (defaulters)
- One-hot encoding is essential for converting categorical variables before feeding into sklearn models

---

## 👤 Author

**Dan Shina**  
Senior Manager → Transitioning into AI & Data Science  
📍 Kaduna, Nigeria  
🔗 [GitHub](https://github.com/shizadan)

---

*Built with Python · scikit-learn · XGBoost · Pandas · Matplotlib*
