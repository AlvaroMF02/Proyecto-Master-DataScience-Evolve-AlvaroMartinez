# Customer Churn Prediction — Proyecto Máster Evolve

Proyecto de Machine Learning y Análisis de Datos enfocado en la predicción de abandono de clientes (*customer churn*) en un entorno bancario mediante modelos de clasificación, feature engineering y análisis orientado a negocio.

El proyecto combina análisis exploratorio de datos, preprocesamiento, modelado predictivo y visualización de información para identificar patrones de comportamiento asociados al abandono de clientes.

---

# Objetivos del Proyecto

* Analizar patrones de comportamiento de clientes
* Identificar factores relacionados con el churn
* Comparar distintos modelos de clasificación
* Crear visualizaciones y métricas orientadas a negocio
* Construir un pipeline modular y reproducible

---

# Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook
* Power BI
* Joblib

---

# Estructura del Proyecto

```bash
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Datos procesados
│
├── models/                     # Modelos entrenados (.pkl)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_models.ipynb
│
├── reports/
│   └── metrics/                # Métricas e importancia de variables
│
├── src/
│   ├── io.py
│   ├── cleaning.py
│   ├── features.py
│   ├── viz.py
│   ├── utils.py
│   └── models.py
│
├── dashboard/                  # Dashboard Power BI
│
├── main.py                     # Pipeline reproducible
├── requirements.txt
└── README.md
```

---

# Análisis Exploratorio de Datos (EDA)

La fase de EDA se centró en comprender el comportamiento de los clientes e identificar patrones relacionados con el abandono bancario.

Principales análisis realizados:

* Distribución de churn
* Churn según edad y grupos de edad
* Clientes activos vs inactivos
* Correlación entre variables numéricas y categóricas

Las visualizaciones generadas se exportan automáticamente dentro de `reports/`.

---

# Preprocesamiento y Feature Engineering

El pipeline de preprocesamiento incluye:

* Verificación de valores nulos
* Codificación de variables categóricas
* Conversión de variables booleanas
* Escalado de variables numéricas
* División train/test

Nuevas características creadas:

* `loyal_customer`
* `age_group`

Toda la lógica de limpieza y transformación fue modularizada dentro de la carpeta `src/`.

---

# Modelos de Machine Learning

Los siguientes modelos fueron entrenados y evaluados:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

Métricas utilizadas:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Los mejores resultados se obtuvieron utilizando modelos ensemble, especialmente XGBoost y Random Forest.

---

# Resultados del Proyecto

El proyecto obtuvo un buen rendimiento predictivo manteniendo interpretabilidad desde el punto de vista de negocio.

Principales hallazgos:

* Los clientes de mayor edad presentan mayor probabilidad de abandono
* Los clientes inactivos tienen más tendencia al churn
* Alemania presenta mayor tasa de abandono respecto a otros países
* Los clientes considerados leales muestran menor churn

Las métricas y reportes generados se exportan automáticamente en `reports/metrics/`.

---

# Dashboard

El proyecto incluye un dashboard desarrollado en Power BI centrado en:

* Análisis de churn
* KPIs de negocio
* Segmentación de clientes
* Comparativa entre grupos de clientes
* Importancia de variables

---

# Pipeline Reproducible

El proyecto incluye un pipeline modular y reproducible ejecutable mediante:

```bash
python main.py
```

El pipeline realiza automáticamente:

1. Carga del dataset
2. Limpieza y transformación de datos
3. Creación de nuevas características
4. Generación de visualizaciones
5. Entrenamiento de modelos
6. Exportación de métricas y datasets procesados
7. Guardado de modelos entrenados

---

# Cómo Ejecutar el Proyecto

## Clonar repositorio

```bash
git clone https://github.com/AlvaroMF02/Proyecto-Master-DataScience-Evolve-AlvaroMartinez
```

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar pipeline completo

```bash
python main.py
```

## Abrir notebooks

Orden recomendado:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_models.ipynb`

---

# Posibles Mejoras Futuras

* Optimización de hiperparámetros
* Validación cruzada avanzada
* API de predicción en tiempo real con Flask o FastAPI
* Mejoras en la interactividad del dashboard

---

# Dataset

Bank Customer Churn Dataset:

https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

---

# Autor

Álvaro Martínez Flores

Proyecto académico desarrollado durante el Máster en Inteligencia Artificial y Big Data en Evolve Academy.