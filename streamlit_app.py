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


@st.cache_resource
def load_models():
    import joblib
    models = {}
    scaler = None

    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    except Exception:
        pass

    try:
        models["xgboost"] = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    except Exception:
        pass

    try:
        models["random_forest"] = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib"))
    except Exception:
        pass

    try:
        from keras.models import load_model as keras_load
        models["ann"] = keras_load(os.path.join(MODEL_DIR, "ann_model.keras"))
    except Exception:
        pass

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
            results[name] = 1 if prob_rusak >= threshold else 0
        except Exception:
            results[name] = None
    return results


st.set_page_config(page_title="Prediksi Kerusakan Mesin", page_icon="⚙️", layout="centered")

st.markdown("""
<style>
.stApp { background: #0f172a; color: #e2e8f0; }
.block-container { padding-top: 1.5rem !important; max-width: 960px !important; }
h1 { text-align: center !important; color: #f8fafc !important; font-size: 1.75rem !important; }
.stMarkdown p { color: #94a3b8; }

.card-form {
    background: #1e293b; border-radius: 1rem; padding: 0.25rem 1.25rem 1.25rem;
    border: 1px solid #334155; margin-bottom: 1rem;
}
label { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 600 !important; }
.stSlider > div > div { color: #f8fafc !important; }
.stSlider > div > div > div { background: #334155 !important; }
.stSlider > div > div > div > div { background: #38bdf8 !important; }
.stNumberInput input { background: #0f172a !important; border: 1px solid #334155 !important; border-radius: 0.5rem !important; color: #f8fafc !important; }
.stSelectbox > div > div { background: #0f172a !important; border: 1px solid #334155 !important; border-radius: 0.5rem !important; color: #f8fafc !important; }
div[data-testid="stButton"] button {
    width: 100%; background: linear-gradient(135deg, #38bdf8, #818cf8); color: #fff;
    border: none; border-radius: 0.5rem; font-size: 1rem; font-weight: 600; padding: 0.5rem 1rem;
}
div[data-testid="stButton"] button:hover { opacity: 0.9; color: #fff !important; border: none !important; }
div[data-testid="stButton"] button:active { transform: scale(0.98); }
.badge { font-size:0.7rem; color:#64748b; background:#0f172a; padding:0.2rem 0.4rem; border-radius:0.3rem; display:inline-block; white-space:nowrap; }

.result-section { margin-top: 1.5rem; }
.result-section h2 { font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.75rem; }
.input-summary {
    background: #0f172a; border: 1px solid #334155; border-radius: 0.5rem;
    padding: 0.5rem 0.75rem; margin-bottom: 1rem; font-size: 0.75rem;
    color: #94a3b8; display: flex; flex-wrap: wrap; gap: 0.4rem 1rem;
}
.input-summary strong { color: #cbd5e1; }
.result-cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem; }
.result-card { background: #1e293b; border-radius: 0.75rem; padding: 1rem; border: 1px solid #334155; position: relative; overflow: hidden; }
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.result-card.normal::before { background: #22c55e; }
.result-card.rusak::before { background: #ef4444; }
.result-card .md-nm { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:#64748b; margin-bottom:0.5rem; }
.result-card .st-row { display:flex; align-items:center; gap:0.5rem; }
.st-icon { width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0; }
.result-card.normal .st-icon { background:rgba(34,197,94,0.15); }
.result-card.rusak .st-icon { background:rgba(239,68,68,0.15); }
.st-text { font-size:1.1rem; font-weight:700; }
.result-card.normal .st-text { color:#22c55e; }
.result-card.rusak .st-text { color:#ef4444; }
.footer-note { padding:0.5rem 0.75rem; background:#0f172a; border-radius:0.5rem; font-size:0.72rem; color:#64748b; }
.tag { display:inline-block; padding:0.1rem 0.35rem; border-radius:0.25rem; font-size:0.68rem; font-weight:600; }
.tag-n { background:rgba(34,197,94,0.15); color:#22c55e; }
.tag-r { background:rgba(239,68,68,0.15); color:#ef4444; }
@media (max-width:700px) { .result-cards { grid-template-columns:1fr; } }

.row { display: flex; gap: 0.5rem; align-items: center; }
.row .val { flex: 1; }
.row .badge { flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚙️ Prediksi <span style='color:#38bdf8'>Kerusakan Mesin</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:0.9rem'>Masukkan parameter mesin untuk memprediksi potensi kerusakan</p>", unsafe_allow_html=True)

models, scaler = load_models()

if "air_temp" not in st.session_state:
    st.session_state.air_temp = 298.0
    st.session_state.process_temp = 308.0
    st.session_state.rotational_speed = 1500
    st.session_state.torque = 30.0
    st.session_state.tool_wear = 0
    st.session_state.type = "L (Low)"

st.markdown('<div class="card-form">', unsafe_allow_html=True)

row1 = st.columns(2)
with row1[0]:
    st.markdown("<label>Air Temperature [K]</label>", unsafe_allow_html=True)
    st.session_state.air_temp = st.slider("at", 295.0, 310.0, st.session_state.air_temp, 0.1, label_visibility="collapsed")
with row1[1]:
    st.markdown("<label>Process Temperature [K]</label>", unsafe_allow_html=True)
    st.session_state.process_temp = st.slider("pt", 305.0, 330.0, st.session_state.process_temp, 0.1, label_visibility="collapsed")

row2 = st.columns(2)
with row2[0]:
    st.markdown("<label>Rotational Speed [rpm]</label>", unsafe_allow_html=True)
    st.session_state.rotational_speed = st.slider("rs", 1000, 3000, st.session_state.rotational_speed, 1, label_visibility="collapsed")
with row2[1]:
    st.markdown("<label>Torque [Nm]</label>", unsafe_allow_html=True)
    st.session_state.torque = st.slider("tq", 0.0, 80.0, st.session_state.torque, 0.1, label_visibility="collapsed")

row3 = st.columns(2)
with row3[0]:
    st.markdown("<label>Tool Wear [min]</label>", unsafe_allow_html=True)
    st.session_state.tool_wear = st.slider("tw", 0, 260, st.session_state.tool_wear, 1, label_visibility="collapsed")
with row3[1]:
    st.markdown("<label>Type</label>", unsafe_allow_html=True)
    type_map = {"H (High)": 0, "L (Low)": 1, "M (Medium)": 2}
    st.session_state.type = st.selectbox("tp", list(type_map.keys()), index=["H (High)", "L (Low)", "M (Medium)"].index(st.session_state.type), label_visibility="collapsed")

st.markdown('<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.25rem">', unsafe_allow_html=True)
st.markdown('<span class="badge">Air: 295 – 310 K</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Process: 305 – 330 K</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Speed: 1000 – 3000 rpm</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Torque: 0 – 80 Nm</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Wear: 0 – 260 min</span>', unsafe_allow_html=True)
st.markdown('<span class="badge">Type: H / L / M</span>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if st.button("🔍 Prediksi Kerusakan", use_container_width=True, type="primary"):
    type_enc = {"H (High)": 0, "L (Low)": 1, "M (Medium)": 2}
    data = {
        "air_temp": st.session_state.air_temp,
        "process_temp": st.session_state.process_temp,
        "rotational_speed": st.session_state.rotational_speed,
        "torque": st.session_state.torque,
        "tool_wear": st.session_state.tool_wear,
        "type": type_enc[st.session_state.type],
    }

    with st.spinner("Memproses..."):
        results = predict(models, scaler, data)

    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.markdown("<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="input-summary">'
        f"<span><strong>Air T:</strong> {data['air_temp']} K</span>"
        f"<span><strong>Process T:</strong> {data['process_temp']} K</span>"
        f"<span><strong>Speed:</strong> {data['rotational_speed']} rpm</span>"
        f"<span><strong>Torque:</strong> {data['torque']} Nm</span>"
        f"<span><strong>Wear:</strong> {data['tool_wear']} min</span>"
        f"<span><strong>Type:</strong> {st.session_state.type}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="result-cards">', unsafe_allow_html=True)
    for name in ["xgboost", "random_forest", "ann"]:
        pred = results.get(name)
        label = MODEL_LABELS.get(name, name)
        if pred is None:
            cls, icon, txt, col = "error", "❌", "Error", "#f59e0b"
        elif pred == 1:
            cls, icon, txt, col = "rusak", "⚠️", "Rusak", "#ef4444"
        else:
            cls, icon, txt, col = "normal", "✅", "Normal", "#22c55e"
        st.markdown(
            f'<div class="result-card {cls}">'
            f'<div class="md-nm">{label}</div>'
            f'<div class="st-row">'
            f'<div class="st-icon">{icon}</div>'
            f'<div class="st-text" style="color:{col}">{txt}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">'
        '<span class="tag tag-n">Normal</span> = mesin tidak rusak, '
        '<span class="tag tag-r">Rusak</span> = berpotensi mengalami kerusakan. '
        "Model: XGBoost, Random Forest, ANN &nbsp;|&nbsp; Dataset: AI4I 2020"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
