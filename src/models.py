import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import joblib

# #################################################
# Creación de funciones para entrenar modelos, evaluar resultados y guardarlos
# #################################################

# lista global de resultados
results = []

# Funcion para la evaluación
def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

# LOGISTIC REGRESSION
def train_logistic_regression(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
):

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    evaluate_model(
        lr,
        "Logistic Regression",
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    return lr

# DECISION TREE
def train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
):

    dt = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    evaluate_model(
        dt,
        "Decision Tree",
        X_train,
        X_test,
        y_train,
        y_test
    )

    return dt

# RANDOM FOREST
def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
):

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    evaluate_model(
        rf,
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test
    )

    return rf


# XGBOOST
def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
):

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    evaluate_model(
        xgb_model,
        "XGBoost",
        X_train,
        X_test,
        y_train,
        y_test
    )

    return xgb_model

# Guardado de resultados
def get_results_dataframe():
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )
    return results_df


def save_model(model, path):
    joblib.dump(model, path)


def save_feature_importance(
    model,
    X_train,
    path
):

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        path,
        index=False
    )