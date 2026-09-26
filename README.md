# Heart Failure Prediction – Machine Learning Analysis

End-to-end data analysis and machine learning project that predicts the presence of heart disease from clinical patient data. Developed as the final project of the **AI Powered Data Analyst** certification (Gatehouse Awards, UK – Ofqual regulated).

## Overview

The project follows a complete data science workflow:

1. **Exploratory Data Analysis (EDA)** – distributions of all features and a correlation matrix to identify the variables most related to heart disease.
2. **Hypothesis formulation** – based on the correlation analysis, `Oldpeak`, `ST_Slope`, `Age` and `MaxHR` were identified as the most informative features.
3. **Unsupervised learning** – K-Means clustering (k = 3) on standardized `Age`, `MaxHR` and `Oldpeak` to segment patients into risk groups, followed by an analysis of the heart disease rate in each cluster.
4. **Preprocessing** – one-hot encoding of categorical variables and a 70/30 train/test split.
5. **Supervised learning** – training and comparison of three classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. **Evaluation** – accuracy, confusion matrices, classification reports (precision, recall, F1) and ROC curves with AUC.

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) by fedesoriano (Kaggle).

It contains clinical records of patients with 11 features, including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression (`Oldpeak`) and ST slope. The target variable is `HeartDisease` (1 = disease, 0 = no disease).

The dataset is downloaded automatically in the notebook via `kagglehub`.

## Results

| Model               | AUC  | Accuracy |
|---------------------|------|----------|
| Logistic Regression | 0.94 | 87.7%    |
| Random Forest       | 0.94 | 87.3%    |
| Decision Tree       | 0.79 | 77.5%    |

**Key findings**

- Logistic Regression and Random Forest achieved the best performance, with equal AUC (0.94).
- Logistic Regression offers strong performance together with interpretability, making it a good baseline.
- Random Forest provides stable predictions with reduced overfitting.
- The Decision Tree is the easiest to visualize and explain, but tends to overfit.
- The K-Means clusters show clearly different heart disease rates, supporting the hypothesis that `Age`, `MaxHR` and `Oldpeak` help separate patients into risk levels:

| Cluster | Patients | With heart disease | Disease rate |
|---------|----------|--------------------|--------------|
| 1       | 248      | 207                | 83.5% (high risk)   |
| 0       | 325      | 205                | 63.1% (medium risk) |
| 2       | 345      | 96                 | 27.8% (low risk)    |

## Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Data handling:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn (StandardScaler, KMeans, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, evaluation metrics)
- **Data source:** Kaggle via kagglehub

## How to Run

```bash
git clone https://github.com/MyronFaryna/heart-failure-prediction-ml.git
cd heart-failure-prediction-ml
pip install -r requirements.txt
jupyter notebook heart_failure_prediction_analysis.ipynb
```

Then run all cells. The dataset is downloaded automatically on the first run.

## Project Structure

```
heart-failure-prediction-ml/
├── heart_failure_prediction_analysis.ipynb   # Full analysis and models
├── requirements.txt                          # Python dependencies
└── README.md
```

## Note

Code comments and interpretations inside the notebook are written in Greek.

This project is for educational purposes only and is not intended for medical diagnosis.

## Author

**Myron Faryna** – [LinkedIn](https://www.linkedin.com/in/%CE%BC%CF%85%CF%81%CF%89%CE%BD-%CF%86%CE%B1%CF%81%CE%B9%CE%BD%CE%B1-385171339) · [GitHub](https://github.com/MyronFaryna)
