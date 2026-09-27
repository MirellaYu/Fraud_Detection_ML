# 🤖 Detección de fraude con aprendizaje automático

## Descripción del proyecto

Este proyecto desarrolla un sistema de detección de transacciones fraudulentas mediante Machine Learning utilizando el Fraud Detection Dataset de Kaggle.

El objetivo es construir un modelo de clasificación binaria capaz de identificar transacciones potencialmente fraudulentas a partir de sus características transaccionales.

El proyecto sigue un flujo end-to-end de Data Science:

**EDA → Feature Engineering → Preprocesamiento → Modelado → Evaluación → Deployment**

Se entrenaron y compararon dos modelos:

- Regresión Logística
- Random Forest

Finalmente, el modelo seleccionado se integró en una aplicación web desarrollada con Flask para realizar predicciones sobre nuevas transacciones.

## 🎯 Objetivos del proyecto

- Analizar las características de las transacciones y su relación con el fraude.
- Identificar y evaluar el desbalance de clases presente en el dataset.
- Preparar las variables para su utilización en modelos de Machine Learning.
- Entrenar modelos de clasificación para detectar transacciones fraudulentas.
- Comparar el desempeño de los modelos utilizando métricas adecuadas para un problema de clasificación desbalanceada.
- Seleccionar el modelo con mejor desempeño general.
- Integrar el modelo seleccionado en una aplicación web para realizar predicciones.

## 🔍 Proceso de Data Science

### 1. Análisis exploratorio de datos

Se realizó un análisis exploratorio para comprender:

- Distribución de las variables.
- Distribución de las transacciones fraudulentas y no fraudulentas.
- Desbalance de clases.
- Comportamiento de los montos de las transacciones.
- Distribución del fraude según tipo de transacción.
- Valores nulos y posibles inconsistencias.

### 2. Feature Engineering

Se realizó la selección de variables utilizadas para el modelado.

Variables predictoras:

- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

Variable objetivo:

- `isFraud`

Se excluyeron variables como `step`, `nameOrig`, `nameDest` e `isFlaggedFraud` para evitar utilizar identificadores o una variable que representa una alerta previa de fraude.

### 3. Preprocesamiento

Se utilizó un pipeline de Scikit-learn para mantener el procesamiento integrado al entrenamiento del modelo.

- Variables numéricas: estandarización para Regresión Logística.
- Variable categórica `type`: One-Hot Encoding.
- División train/test estratificada.
- `class_weight="balanced"` para considerar el fuerte desbalance entre las clases.

## 🤖 Modelado

Se entrenaron dos modelos de clasificación:

| Modelo | Propósito |
|---|---|
| Logistic Regression | Modelo base lineal |
| Random Forest | Modelo no lineal |

Ambos modelos fueron entrenados utilizando una división estratificada de los datos y `class_weight="balanced"`.

La evaluación se realizó utilizando principalmente F1-Macro, Recall-Macro y Balanced Accuracy debido al fuerte desbalance de clases.

## 📊 Resultados

Los modelos fueron evaluados sobre un conjunto de prueba de 1,272,524 transacciones.

| Modelo | Accuracy | Precisión Macro | Recall Macro | F1-Macro | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9468 | 0.5110 | 0.9411 | 0.5080 | 0.9411 |
| Random Forest | 0.9996 | 0.9163 | 0.9454 | 0.9304 | 0.9454 |

Debido al fuerte desbalance de clases, Accuracy no se utiliza como único criterio de evaluación.

Random Forest obtuvo un F1-Macro de 0.9304 y una Balanced Accuracy de 0.9454 en el conjunto de prueba, superando a la Regresión Logística en las principales métricas utilizadas para este problema.

Por este motivo, Random Forest fue seleccionado para la integración con la aplicación web.

## 🌐 Deployment

El modelo Random Forest seleccionado fue integrado en una aplicación web desarrollada con Flask.

La aplicación permite ingresar las características de una transacción y obtener:

- Predicción de fraude/no fraude.
- Probabilidad estimada asociada a la predicción.

### Flujo de la aplicación

Usuario
↓
Formulario web
↓
Flask API
↓
Modelo Random Forest
↓
Predicción
↓
Probabilidad estimada
↓
Resultado en la interfaz

## 🧰 Herramientas utilizadas

- **Python:** lenguaje principal para análisis y Machine Learning.
- **Pandas / NumPy:** manipulación y preparación de datos.
- **Matplotlib:** visualización y análisis exploratorio.
- **Scikit-learn:** preprocesamiento, entrenamiento y evaluación de modelos.
- **Flask:** integración del modelo en una aplicación web.
- **Jupyter Notebook:** análisis exploratorio y Feature Engineering.
- **VS Code:** desarrollo y organización del proyecto.
- **Git / GitHub:** control de versiones y documentación.
- **Dataset:** Fraud Detection Dataset (Kaggle).
