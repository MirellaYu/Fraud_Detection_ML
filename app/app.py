from pathlib import Path
import joblib
from flask import Flask, render_template, request
import pandas as pd

# ============================================================
# Configuración
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest"
    / "random_forest_model.pkl"
)


# ============================================================
# Aplicación Flask
# ============================================================
app = Flask(__name__)


# ============================================================
# Cargar modelo
# ============================================================
model = joblib.load(MODEL_PATH)


# ============================================================
# Ruta principal
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# Ruta para recibir datos de la transacción
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([{
        "type": data["type"],
        "amount": data["amount"],
        "oldbalanceOrg": data["oldbalanceOrg"],
        "newbalanceOrig": data["newbalanceOrig"],
        "oldbalanceDest": data["oldbalanceDest"],
        "newbalanceDest": data["newbalanceDest"]
    }])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    if prediction == 1:
        probability = probabilities[1]  
    else:
        probability = probabilities[0]  
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

# ============================================================
# Ejecución
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)