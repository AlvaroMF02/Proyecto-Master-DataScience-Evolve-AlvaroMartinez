
# 🏦 Bank Customer Churn Prediction

### Análisis de datos y Machine Learning para la predicción del abandono de clientes bancarios

Proyecto de Data Analytics y Machine Learning desarrollado durante el Máster en Data Analysis e IA de Evolve Academy.

El objetivo es analizar el comportamiento de los clientes de una entidad bancaria, identificar patrones asociados al abandono (*Customer Churn*) y desarrollar modelos predictivos que permitan estimar qué clientes presentan una mayor probabilidad de abandonar el banco.

El proyecto integra análisis exploratorio de datos (EDA), preprocesamiento, Feature Engineering, entrenamiento y evaluación de modelos de clasificación y visualización de resultados mediante Power BI.

Desde una perspectiva de negocio, este tipo de análisis puede ayudar a identificar segmentos de clientes sobre los que diseñar estrategias de fidelización y retención.

---

## 🛠️ Tecnologías utilizadas

- **Python:** desarrollo del pipeline de análisis y modelado.
- **Pandas y NumPy:** manipulación, limpieza y transformación de datos.
- **Scikit-learn:** preprocesamiento, entrenamiento y evaluación de modelos.
- **XGBoost:** implementación de un modelo de Gradient Boosting.
- **Matplotlib y Seaborn:** análisis exploratorio y visualización.
- **Joblib:** almacenamiento de modelos entrenados.
- **Jupyter Notebook:** experimentación y documentación del análisis.
- **Power BI:** visualización de indicadores de negocio y resultados predictivos.

---

## 🔄 Arquitectura y flujo del proyecto

El proyecto sigue un pipeline modular que permite ejecutar las distintas etapas del proceso de análisis y Machine Learning.

```text
           Dataset bancario
                  |
                  v
       Análisis exploratorio (EDA)
                  |
                  v
       Limpieza y preprocesamiento
                  |
                  v
          Feature Engineering
                  |
                  v
          División Train / Test
                  |
                  v
         Entrenamiento de modelos
       LR | DT | Random Forest | XGB
                  |
                  v
        Evaluación y comparación
       Accuracy | Recall | ROC-AUC
                  |
                  v
          Exportación de datos
          Modelos y métricas
                  |
                  v
               Power BI
     Business Insights | ML Performance
```

---

## 📊 Dataset y análisis exploratorio

Se utiliza el Bank Customer Churn Dataset, compuesto por 10.000 registros y 18 variables originales relacionadas con las características y el comportamiento de los clientes.

Entre las variables analizadas destacan:

- Edad y género.
- País de residencia.
- Antigüedad como cliente.
- Saldo bancario.
- Número de productos contratados.
- Actividad del cliente.
- Puntuación crediticia.
- Salario estimado.
- Abandono de la entidad (`Exited`).

### Distribución del abandono

El análisis inicial muestra un conjunto de datos moderadamente desbalanceado:

| Estado del cliente | Registros | Porcentaje |
|---|---:|---:|
| Permanece en el banco | 7.962 | 79,62 % |
| Abandona el banco | 2.038 | 20,38 % |
| **Total** | **10.000** | **100 %** |

Esta distribución hace especialmente importante utilizar métricas adicionales al Accuracy para evaluar correctamente los modelos predictivos.

### Principales hallazgos del EDA

Durante el análisis exploratorio se identificaron diferentes patrones asociados al abandono:

- Los clientes que abandonan presentan una edad mediana superior.
- Los clientes inactivos muestran una mayor tasa de abandono.
- Alemania presenta un porcentaje de churn superior al de Francia y España.
- Se observaron diferencias en el abandono según características demográficas y productos contratados.

Estos resultados describen asociaciones dentro del dataset analizado, no relaciones causales.

### Detección de posible Data Leakage

Durante el análisis de correlaciones se detectó que la variable `Complain` presentaba una correlación perfecta con la variable objetivo (`Exited`).

Esto podía representar una posible fuga de información (*Data Leakage*), introduciendo información que revelaría indirectamente el resultado que se pretende predecir.

Por este motivo, se decidió eliminar dicha variable durante el preprocesamiento para evitar una evaluación artificialmente optimista de los modelos.

---

## ⚙️ Preprocesamiento y Feature Engineering

El procesamiento de los datos se organiza mediante funciones independientes dentro de la carpeta `src/`.

Las principales transformaciones realizadas fueron:

- Normalización de los nombres de las columnas.
- Eliminación de variables identificativas y de aquellas que podían introducir Data Leakage.
- Codificación de variables categóricas.
- Conversión de variables booleanas a valores numéricos.
- Creación de nuevas características a partir de variables existentes.
- División estratificada de los datos en entrenamiento (80 %) y prueba (20 %).
- Estandarización de variables numéricas mediante `StandardScaler`.

### Nuevas características

Se implementaron dos variables adicionales:

| Variable | Descripción |
|---|---|
| `loyal_customer` | Identifica clientes con una antigüedad igual o superior a cinco años. |
| `age_group` | Segmentación de clientes por intervalos de edad. |

El escalado se aplica a Logistic Regression, mientras que los modelos basados en árboles utilizan las características sin estandarizar.

Para evitar introducir información del conjunto de prueba durante el escalado, `StandardScaler` se ajusta únicamente con los datos de entrenamiento.

---

## 🤖 Modelos de Machine Learning

Se entrenaron y compararon cuatro algoritmos de clasificación:

1. Logistic Regression.
2. Decision Tree.
3. Random Forest.
4. XGBoost.

Se aplicó `class_weight="balanced"` en Logistic Regression, Decision Tree y Random Forest para considerar el desequilibrio entre las clases durante el entrenamiento.

### Métricas de evaluación

Los modelos se evaluaron utilizando cinco métricas:

- **Accuracy:** proporción total de predicciones correctas.
- **Precision:** proporción de abandonos correctamente identificados entre los clientes clasificados como abandono.
- **Recall:** capacidad para detectar clientes que realmente abandonan.
- **F1 Score:** equilibrio entre Precision y Recall.
- **ROC-AUC:** capacidad del modelo para discriminar entre ambas clases a través de distintos umbrales.

La comparación de estas métricas permite estudiar el comportamiento de los modelos desde diferentes perspectivas, evitando depender exclusivamente del Accuracy.

---

## 📈 Resultados de los modelos

Los resultados obtenidos sobre el conjunto de prueba fueron:

| Modelo | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72,35 % | 40,16 % | 72,55 % | 51,70 % | 0,800 |
| Decision Tree | 80,00 % | 51,02 % | 49,26 % | 50,12 % | 0,686 |
| Random Forest | 86,75 % | 83,57 % | 43,63 % | 57,33 % | 0,863 |
| XGBoost | 86,40 % | 75,56 % | 49,26 % | 59,64 % | 0,873 |

### Interpretación de los resultados

Los modelos Random Forest y XGBoost presentan un rendimiento destacado en Accuracy y ROC-AUC.

Sin embargo, Logistic Regression alcanza el mayor Recall, identificando aproximadamente el 72,55 % de los clientes que realmente abandonan la entidad.

Esta diferencia demuestra la importancia de seleccionar las métricas de evaluación en función del objetivo de negocio.

Por ejemplo, en una estrategia de retención orientada a detectar el mayor número posible de clientes en riesgo, el Recall puede resultar especialmente relevante, aunque implique identificar erróneamente a algunos clientes que finalmente permanecerán.

Por tanto, la selección final del modelo dependería del equilibrio deseado entre la detección de abandonos y el coste de intervenir sobre clientes que no presentan un riesgo real.

### Importancia de las variables

Se analizó la importancia de las características mediante los modelos Random Forest y XGBoost.

Entre las variables con mayor importancia en XGBoost se encuentran:

- Número de productos contratados.
- Actividad del cliente.
- Edad.
- País de residencia, especialmente Alemania.
- Saldo bancario.

La importancia de las variables representa su contribución al modelo, pero no debe interpretarse como evidencia de causalidad.

Los resultados y modelos generados se exportan para facilitar su posterior análisis y reutilización.

---

## 📊 Dashboard en Power BI

Se desarrolló un dashboard interactivo para complementar el análisis predictivo con una perspectiva orientada a Business Intelligence.

El informe está dividido en dos páginas principales:

### 1. Business Insights

Permite analizar el comportamiento de los clientes y explorar los principales indicadores relacionados con el abandono.

![Business Insights](dashboard/Bussiness%20Insights.png)

### 2. ML Performance

Presenta los resultados del proceso de modelado, facilitando la comparación del rendimiento predictivo y la interpretación de las variables utilizadas.

![ML Performance](dashboard/ML%20Performance.png)

El archivo original se encuentra disponible en:

`dashboard/Dashboard.pbix`

---

## 📂 Estructura del repositorio

```text
Proyecto-Master-DataScience-Evolve-AlvaroMartinez/
│
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Datos generados durante el procesamiento
│
├── models/                     # Modelos entrenados (.pkl)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Análisis exploratorio
│   ├── 02_preprocessing.ipynb  # Limpieza y Feature Engineering
│   └── 03_models.ipynb         # Entrenamiento y evaluación
│
├── reports/
│   ├── figures/                # Visualizaciones del EDA
│   └── metrics/                # Resultados e importancia de variables
│
├── src/
│   ├── io.py                   # Carga y exportación de datos
│   ├── cleaning.py             # Limpieza y codificación
│   ├── features.py             # Creación de características
│   ├── utils.py                # División y escalado
│   ├── models.py               # Entrenamiento y evaluación
│   └── viz.py                  # Generación de visualizaciones
│
├── dashboard/
│   ├── Dashboard.pbix
│   ├── Bussiness Insights.png
│   └── ML Performance.png
│
├── main.py                     # Pipeline completo
├── requirements.txt            # Dependencias
└── README.md                   # Documentación
```

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/AlvaroMF02/Proyecto-Master-DataScience-Evolve-AlvaroMartinez.git

cd Proyecto-Master-DataScience-Evolve-AlvaroMartinez
```

### 2. Crear un entorno virtual

Se recomienda utilizar un entorno virtual de Python para aislar las dependencias del proyecto.

```bash
python -m venv .venv
```

Activarlo en Windows:

```bash
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline completo

```bash
python main.py
```

Este script automatiza las siguientes operaciones:

1. Carga del dataset original.
2. Generación de visualizaciones exploratorias.
3. Limpieza y transformación de datos.
4. Creación de nuevas características.
5. División y escalado de los datos.
6. Entrenamiento y evaluación de los cuatro modelos.
7. Exportación de métricas y datasets procesados.
8. Almacenamiento de Random Forest y XGBoost mediante Joblib.
9. Exportación de la importancia de variables de XGBoost.

### 5. Consultar los notebooks

Para explorar el desarrollo paso a paso, se recomienda seguir este orden:

```text
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_models.ipynb
```

### 6. Visualizar los resultados

Los archivos generados por el pipeline se almacenan principalmente en:

- `data/processed/`
- `reports/figures/`
- `reports/metrics/`
- `models/`

El dashboard puede consultarse mediante Power BI Desktop, utilizando el archivo `dashboard/Dashboard.pbix`.

---

## 📌 Conclusiones y posibles mejoras

El proyecto permite desarrollar un flujo completo de análisis y Machine Learning aplicado a un problema de abandono de clientes.

Durante su desarrollo se trabajó tanto en la preparación y calidad de los datos como en la evaluación de modelos predictivos, prestando especial atención a la interpretación de los resultados y su aplicación a un contexto de negocio.

La comparación entre modelos demuestra que una mayor exactitud global no implica necesariamente una mayor capacidad para detectar clientes en riesgo de abandono.

Como posibles líneas de mejora se plantean:

- Optimización de hiperparámetros mediante herramientas como Optuna.
- Incorporación de validación cruzada avanzada.
- Análisis de interpretabilidad mediante SHAP.
- Evaluación de diferentes umbrales de clasificación según el coste de falsos positivos y falsos negativos.
- Desarrollo de una API de predicción mediante FastAPI o Flask.
- Ampliación de las visualizaciones del dashboard.

---

## 📚 Fuente de datos

[Bank Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

---

**Autor:** Álvaro Martínez Flores  
**Formación:** Máster en Data Analysis e IA — Evolve Academy
