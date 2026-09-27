# 3. Modelado del modelo Regresión Logística
# 3.1. Importación de las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

# 3.2. Directorio del modelo
SEED = 42
DIR_RESULTADOS = Path("models/logistic_regression")
DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


# 3.3. Cargar el dataset final
DATA_PATH = Path("data/final/fraud_detection_final.csv")
df_model = pd.read_csv(DATA_PATH)

# 3.4. Definir las variables predictoras y la variable objetivo
# Variables categóricas
categorical = ["type"]
# Variables numéricas
numeric = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]
# Variable objetivo
target = "isFraud"

# 3.5. Separar variables predictoras y variable objetivo
X = df_model.drop(columns=[target])
y = df_model[target]

# 3.6. Definir el preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="drop"
)

# 3.7. Configuración de hiperparámetros del modelo
pipeline = Pipeline([
    ("prep", preprocessor),
    (
        "clf",
        LogisticRegression(
            C=1.0,
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=SEED
        )
    )
])

# 3.8. División de los datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# 3.9. Entrenamiento del modelo Regresion Logística
print(" Iniciando entrenamiento Regresion Logistica...")
pipeline.fit(X_train, y_train)
print(" Entrenamiento terminado")

# 3.10. Curva de aprendizaje del modelo Regresión Logística
def plot_learning_curve(model, X, y, seed=42):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    resultados = pd.DataFrame({
        "train_size": train_sizes,
        "f1_macro_train_cv_mean": train_mean,
        "f1_macro_train_cv_std": train_std,
        "f1_macro_valid_cv_mean": val_mean,
        "f1_macro_valid_cv_std": val_std,
        "loss_train_mean (1-F1)": 1 - train_mean,
        "loss_valid_mean (1-F1)": 1 - val_mean
    })

    ruta_csv = DIR_RESULTADOS / "learning_curve_logistic_regression.csv"
    resultados.to_csv(ruta_csv, index=False)


    plt.figure(figsize=(10, 6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Entrenamiento"
    )

    plt.plot(
        train_sizes,
        val_mean,
        marker="o",
        label="Validación"
    )

    plt.title("Curva de Aprendizaje de Regresión Logística")
    plt.xlabel("Tamaño del conjunto de entrenamiento (TRAIN)")
    plt.ylabel("F1-macro")
    plt.ticklabel_format(style="plain", axis="x")

    plt.legend()
    plt.grid(True)

    # Guardar imagen de la curva de aprendizaje
    ruta_img = DIR_RESULTADOS / "learning_curve_logistic_regression.png"
    plt.savefig(ruta_img, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

plot_learning_curve(
    pipeline,
    X_train,
    y_train,
    seed=SEED
)

# 3.11. Evaluación del modelo Regresion Logistica (TEST)
# Predicciones sobre el conjunto TEST
y_pred = pipeline.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test,y_pred,average="macro")
recall_macro = recall_score(y_test,y_pred,average="macro")
f1_macro = f1_score(y_test,y_pred,average="macro")
balanced_acc = balanced_accuracy_score(y_test,y_pred)
# Resultados
print(f"Cantidad de datos en TEST: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision Macro: {precision_macro:.4f}")
print(f"Recall Macro: {recall_macro:.4f}")
print(f"F1-Macro: {f1_macro:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Tabla de resultados
resultados_test = pd.DataFrame({
    "modelo": ["Logistic Regression"],
    "cantidad_test": [len(X_test)],
    "accuracy": [accuracy],
    "precision_macro": [precision_macro],
    "recall_macro": [recall_macro],
    "f1_macro": [f1_macro],
    "balanced_accuracy": [balanced_acc]
})

# Guardar CSV
ruta_csv = DIR_RESULTADOS / "metricas_test_logistic_regression.csv"
resultados_test.to_csv(ruta_csv, index=False)

# 3.12. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Fraude", "Fraude"]
)

disp.plot(values_format="d")

plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicho")
plt.ylabel("Real")

ruta_img = DIR_RESULTADOS / "confusion_matrix_logistic_regression.png"
plt.savefig(ruta_img, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 3.13. Guardar el modelo entrenado
ruta_modelo = DIR_RESULTADOS / "logistic_regression.pkl"
joblib.dump(pipeline, ruta_modelo)