# Customer Churn Prediction Evolve Alvaro Martinez

Machine Learning project focused on predicting customer churn in a banking environment using classification models and business-oriented analysis.

The project includes:

* Exploratory Data Analysis (EDA)
* Data preprocessing and feature engineering
* Model training and evaluation
* Power BI dashboard for business insights

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook
* Power BI

---

# Project Structure

```bash
├── dashboard/          # Power BI dashboard files
├── data/
│   ├── raw/            # Original dataset
│   └── processed/      # Cleaned and processed datasets
├── models/             # Trained models (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_models.ipynb
├── README.md
└── requirements.txt
```

---

# Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

The best overall performance was achieved with XGBoost and Random Forest.

Main evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# Main Objectives

* Analyze customer behavior patterns
* Identify factors related to churn
* Compare multiple classification models
* Create business-oriented visualizations and dashboards

---

# Dashboard

The project includes a Power BI dashboard focused on:

* Customer churn analysis
* Business KPIs
* Customer segmentation
* Model performance comparison
* Feature importance visualization

---

# How to Run

1. Clone the repository
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open the notebooks in order:

* 01_eda.ipynb
* 02_preprocessing.ipynb
* 03_models.ipynb

---

# Results

The project achieved strong classification results using ensemble models, especially XGBoost, with high ROC-AUC performance and balanced churn detection.

---

# Author

Álvaro Martínez Flores

Academic project developed during the Master in Artificial Intelligence and Big Data at Evolve.
