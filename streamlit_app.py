import os
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

THRESHOLDS = {
    "xgboost": 0.8208,
    "random_forest": 0.4572,
    "ann": 0.3624,
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "ann": "ANN (Neural Network)",
}

MODEL_ICONS = {
    "xgboost": "🔵",
    "random_forest": "🟢",
    "ann": "🔴",
}

FEATURE_ORDER = ["type", "air_temp", "process_temp", "rotational_speed", "torque", "tool_wear"]

@st.cache_resource
def load_models():
    import joblib
    models = {}
    scaler = None

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    except Exception as e:
        st.warning(f"Scaler load failed: {e}")

    try:
        models["xgboost"] = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    except Exception as e:
        st.warning(f"XGBoost load failed: {e}")

    try:
        models["random_forest"] = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib"))
    except Exception as e:
        st.warning(f"Random Forest load failed: {e}")

    try:
        from keras.models import load_model as keras_load
        models["ann"] = keras_load(os.path.join(MODEL_DIR, "ann_model.keras"))
    except Exception as e:
        st.warning(f"ANN load failed: {e}")

    return models, scaler


def predict(models, scaler, input_data):
    raw = np.array([[
        input_data["type"],
        input_data["air_temp"],
        input_data["process_temp"],
        input_data["rotational_speed"],
        input_data["torque"],
        input_data["tool_wear"],
    ]])

    features = scaler.transform(raw) if scaler is not None else raw

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

            results[name] = pred
        except Exception as e:
            results[name] = None

    return results


st.set_page_config(
    page_title="Prediksi Kerusakan Mesin",
    page_icon="⚙️",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    h1 { text-align: center; }
    .card {
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .card-normal {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
    }
    .card-rusak {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
    }
    .card-error {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
    }
    .model-name {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-text {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .text-normal { color: #22c55e; }
    .text-rusak { color: #ef4444; }
    .stApp { background: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚙️ Prediksi Kerusakan Mesin")
st.markdown("<p style='text-align:center;color:#64748b'>Masukkan parameter mesin untuk memprediksi potensi kerusakan</p>", unsafe_allow_html=True)

models, scaler = load_models()

with st.form("prediction_form"):
    st.subheader("Parameter Mesin")

    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("Air Temperature [K]", min_value=295.0, max_value=310.0, value=298.0, step=0.1,
                                   help="Rentang normal: 295 – 304 K")
        process_temp = st.number_input("Process Temperature [K]", min_value=305.0, max_value=330.0, value=308.0, step=0.1,
                                       help="Rentang normal: 305 – 313 K")
        rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=1,
                                           help="Rentang normal: 1168 – 2886 rpm")

    with col2:
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=80.0, value=30.0, step=0.1,
                                 help="Rentang normal: 3.8 – 76.6 Nm")
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=260, value=0, step=1,
                                    help="Rentang normal: 0 – 253 min")
        type_map = {"H (High)": 0, "L (Low)": 1, "M (Medium)": 2}
        type_label = st.selectbox("Type", list(type_map.keys()), index=1,
                                  help="Spesifikasi mesin: Low / Medium / High")
        type_val = type_map[type_label]

    submitted = st.form_submit_button("🔍 Prediksi Kerusakan", use_container_width=True)

if submitted:
    input_data = {
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type": type_val,
    }

    with st.spinner("Memproses..."):
        results = predict(models, scaler, input_data)

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    cols = st.columns(3)
    for i, (name, pred) in enumerate(results.items()):
        with cols[i]:
            label = MODEL_LABELS.get(name, name)
            icon = MODEL_ICONS.get(name, "📋")

            if pred is None:
                st.markdown(
                    f"""
                    <div class="card card-error">
                        <div class="model-name">{label}</div>
                        <div style="font-size:1.2rem;font-weight:600;color:#f59e0b">Error</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif pred == 1:
                st.markdown(
                    f"""
                    <div class="card card-rusak">
                        <div class="model-name">{label}</div>
                        <div class="status-text text-rusak">⚠️ Rusak</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card card-normal">
                        <div class="model-name">{label}</div>
                        <div class="status-text text-normal">✅ Normal</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.caption("⚙️ Model: XGBoost, Random Forest, ANN (Neural Network) | Dataset: AI4I 2020 Predictive Maintenance")
