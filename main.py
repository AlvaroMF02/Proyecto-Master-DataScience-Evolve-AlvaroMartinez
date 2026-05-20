from src.viz import (plot_churn_distribution, plot_active_customers_vs_churn, plot_correlation_matrix)
from src.io import load_data, save_dataframe
from src.features import create_features
from src.cleaning import (
    clean_columns,
    drop_unused_columns,
    encode_categorical_variables,
    convert_booleans_to_int
)
from src.utils import (
    split_data,
    scale_data
)
from src.models import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    get_results_dataframe,
    save_model,
    save_feature_importance
)

# Cargar datos
df = load_data(
    "data/raw/Bank-Customer-Churn.csv"
)

# Visualizaciones
plot_churn_distribution(df)
plot_active_customers_vs_churn(df)
plot_correlation_matrix(df)

# Añadir nuevas características
df = create_features(df)

# Limpieza
df = clean_columns(df)
df = drop_unused_columns(df)
df = encode_categorical_variables(df)
df = convert_booleans_to_int(df)

# Guardado
save_dataframe(
    df,
    "data/processed/churn_processed.csv"
)

# Division de datos
X_train, X_test, y_train, y_test = split_data(df)

# Escalado de datos
X_train_scaled, X_test_scaled = scale_data(
    X_train,
    X_test
)

# Guardar train y test
save_dataframe(
    X_train,
    "data/processed/X_train.csv"
)

save_dataframe(
    X_test,
    "data/processed/X_test.csv"
)

save_dataframe(
    X_train_scaled,
    "data/processed/X_train_scaled.csv"
)

save_dataframe(
    X_test_scaled,
    "data/processed/X_test_scaled.csv"
)

save_dataframe(
    y_train,
    "data/processed/y_train.csv"
)

save_dataframe(
    y_test,
    "data/processed/y_test.csv"
)

print("Procesado de datos completado sin problemas.")

# entrenar modelos
lr = train_logistic_regression(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

dt = train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test
)

rf = train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)

xgb_model = train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
)

# dataframe de resultados
results_df = get_results_dataframe()

results_df.to_csv(
    "reports/metrics/resultados_modelos.csv",
    index=False
)

# guardar modelos
save_model(
    rf,
    "models/random_forest.pkl"
)

save_model(
    xgb_model,
    "models/xgboost.pkl"
)

# guardar importancia de características
save_feature_importance(
    xgb_model,
    X_train,
    "reports/metrics/importancia_caracteristicas.csv"
)

print("Modelos entrenados con éxito.")