import os
import sys
import warnings
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

models = {}
scaler = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    global scaler

    try:
        import joblib
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
        scaler = joblib.load(scaler_path)
        print("[OK] Scaler loaded")
    except Exception as e:
        print(f"[!] Scaler load failed: {e}")

    try:
        import joblib
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.joblib")
        models["xgboost"] = joblib.load(xgb_path)
        print("[OK] XGBoost model loaded")
    except Exception as e:
        print(f"[!] XGBoost load failed: {e}")

    try:
        import joblib
        rf_path = os.path.join(MODEL_DIR, "random_forest_model.joblib")
        models["random_forest"] = joblib.load(rf_path)
        print("[OK] Random Forest model loaded")
    except Exception as e:
        print(f"[!] Random Forest load failed: {e}")

    try:
        from keras.models import load_model as keras_load
        ann_path = os.path.join(MODEL_DIR, "ann_model.keras")
        models["ann"] = keras_load(ann_path)
        print("[OK] ANN model loaded")
    except Exception as e:
        print(f"[!] ANN load failed: {e}")

load_models()

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/style.css")
def css():
    return send_file("style.css")

@app.route("/script.js")
def js():
    return send_file("script.js")

@app.route("/api/models")
def list_models():
    return jsonify({
        "models": list(models.keys()),
        "features": [
            {"name": "air_temp", "label": "Air Temperature [K]", "type": "number", "min": 295, "max": 310},
            {"name": "process_temp", "label": "Process Temperature [K]", "type": "number", "min": 305, "max": 330},
            {"name": "rotational_speed", "label": "Rotational Speed [rpm]", "type": "number", "min": 1000, "max": 3000},
            {"name": "torque", "label": "Torque [Nm]", "type": "number", "min": 0, "max": 80},
            {"name": "tool_wear", "label": "Tool Wear [min]", "type": "number", "min": 0, "max": 260},
            {"name": "type", "label": "Type", "type": "select", "options": [
                {"value": 0, "label": "H (High)"},
                {"value": 1, "label": "L (Low)"},
                {"value": 2, "label": "M (Medium)"}
            ]}
        ]
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data"}), 400

    try:
        raw = np.array([[
            int(data["type"]),
            float(data["air_temp"]),
            float(data["process_temp"]),
            float(data["rotational_speed"]),
            float(data["torque"]),
            float(data["tool_wear"]),
        ]])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    features = scaler.transform(raw) if scaler is not None else raw

    THRESHOLDS = {
        "xgboost": 0.8208,
        "random_forest": 0.4572,
        "ann": 0.3624,
    }

    results = {}
    for name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                prob_rusak = float(model.predict_proba(features)[0][1])
            else:
                raw_out = model.predict(features, verbose=0)[0]
                prob_rusak = float(raw_out[0]) if hasattr(raw_out, "__len__") else float(raw_out)

            threshold = THRESHOLDS.get(name, 0.5)
            pred = 1 if prob_rusak >= threshold else 0

            results[name] = {
                "prediction": pred,
                "label": "Rusak" if pred == 1 else "Normal",
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
