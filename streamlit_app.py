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

SLIDER_PARAMS = [
    ("air_temp", "Air Temperature [K]", 295.0, 310.0, 0.1, 298.0, "295 – 310 K"),
    ("process_temp", "Process Temperature [K]", 305.0, 330.0, 0.1, 308.0, "305 – 330 K"),
    ("rotational_speed", "Rotational Speed [rpm]", 1000, 3000, 1, 1500, "1000 – 3000 rpm"),
    ("torque", "Torque [Nm]", 0.0, 80.0, 0.1, 30.0, "0 – 80 Nm"),
    ("tool_wear", "Tool Wear [min]", 0, 260, 1, 0, "0 – 260 min"),
]

TYPE_OPTIONS = {"H (High)": 0, "L (Low)": 1, "M (Medium)": 2}

# ─── Init session state ───
for key, _, vmin, vmax, step, default, _ in SLIDER_PARAMS:
    for suf in ("sld", "num"):
        sk = f"{key}_{suf}"
        if sk not in st.session_state:
            st.session_state[sk] = default


def _on_slider(key):
    st.session_state[f"{key}_num"] = st.session_state[f"{key}_sld"]


def _on_num(key):
    st.session_state[f"{key}_sld"] = st.session_state[f"{key}_num"]


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
                prob = float(model.predict_proba(features)[0][1])
            else:
                out = model.predict(features, verbose=0)[0]
                prob = float(out[0]) if hasattr(out, "__len__") else float(out)
            results[name] = 1 if prob >= THRESHOLDS.get(name, 0.5) else 0
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

/* kill Streamlit default spacing */
.element-container { margin: 0 !important; padding: 0 !important; }
.row-widget { margin: 0 !important; padding: 0 !important; }
.stVerticalBlock { gap: 0 !important; }

/* Card */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 0.75rem;
    padding: 1.5rem;
}

/* Form group spacing — match original .form-group margin */
.form-group-cell {
    margin-bottom: 0.75rem;
}

/* Outer-level horizontal blocks (the 2‑col grid) use match original .grid-2 gap */
div[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
}

/* Nested horizontal blocks (number‑input + badge) use original .input-row gap */
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
}

.form-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    margin-bottom: 0.35rem;
    display: block;
}

/* Slider — emulate original <input range> */
.stSlider {
    padding: 0 !important;
    margin: 0 0 0.4rem 0 !important;
}
/* hide the floating label (original didn't have it; value lives in number input) */
.stSlider [data-testid="stSliderThumbValue"] {
    display: none !important;
}
.stSlider [data-testid="stSliderTrack"] {
    background: #334155 !important;
    border-radius: 2px !important;
    height: 4px !important;
}
.stSlider [data-testid="stSliderTrack"] > div {
    background: #38bdf8 !important;
    border-radius: 2px !important;
    height: 4px !important;
}
.stSlider input[type="range"] {
    margin: 0 !important;
}
.stSlider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    background: #38bdf8 !important;
    border: 2px solid #0f172a !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    cursor: pointer !important;
    margin-top: -5px !important;
}
.stSlider input[type="range"]::-moz-range-thumb {
    background: #38bdf8 !important;
    border: 2px solid #0f172a !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    cursor: pointer !important;
}

/* Number Input — fill remaining space */
.stNumberInput {
    min-width: 0 !important;
}
.stNumberInput [data-testid="baseInput"] {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 0.5rem !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.75rem !important;
    box-shadow: none !important;
    outline: none !important;
    height: auto !important;
    box-sizing: border-box !important;
    width: 100% !important;
    min-width: 0 !important;
    transition: border-color 0.2s;
}
.stNumberInput [data-testid="baseInput"]:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}
.stNumberInput button {
    display: none !important;
}

/* Range badge */
.range-badge {
    display: inline-flex;
    align-items: center;
    height: 38px;
    padding: 0 0.5rem;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 0.35rem;
    color: #64748b;
    font-size: 0.7rem;
    white-space: nowrap;
    box-sizing: border-box;
    flex-shrink: 0;
}

/* Selectbox */
.stSelectbox [data-testid="baseInput"] {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 0.5rem !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.75rem !important;
    box-shadow: none !important;
    outline: none !important;
    height: auto !important;
    box-sizing: border-box !important;
    width: 100% !important;
    transition: border-color 0.2s;
}
.stSelectbox [data-testid="baseInput"]:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}
.stSelectbox [data-testid="stSelectboxDropdown"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 0.5rem !important;
    padding: 0.7rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    margin-top: 0.75rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* Input summary */
.input-summary {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 0.5rem;
    padding: 0.6rem 0.85rem;
    margin-bottom: 1rem;
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.2rem;
}
.input-summary span { white-space: nowrap; }
.input-summary strong { color: #cbd5e1; }

/* Results */
.results-section { margin-top: 2rem; }
.results-section h2 { font-size: 1.1rem; color: #94a3b8; margin-bottom: 0.75rem; }
.rci {
    background: #1e293b; border-radius: 0.75rem; padding: 1.25rem;
    border: 1px solid #334155; position: relative; overflow: hidden;
}
.rci::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.rci-normal::before { background: #22c55e; }
.rci-rusak::before { background: #ef4444; }
.rci-name {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #64748b; margin-bottom: 0.6rem;
}
.rci-row { display: flex; align-items: center; gap: 0.6rem; }
.rci-icon {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.rci-normal .rci-icon { background: rgba(34, 197, 94, 0.15); }
.rci-rusak .rci-icon { background: rgba(239, 68, 68, 0.15); }
.rci-status { font-size: 1.2rem; font-weight: 700; }
.rci-normal .rci-status { color: #22c55e; }
.rci-rusak .rci-status { color: #ef4444; }

.result-note {
    margin-top: 1rem; padding: 0.6rem 0.85rem; background: #0f172a;
    border-radius: 0.5rem; font-size: 0.78rem; color: #64748b;
}
.tag {
    display: inline-block; padding: 0.1rem 0.4rem;
    border-radius: 0.25rem; font-size: 0.72rem; font-weight: 600;
}
.tag-normal { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
.tag-rusak { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚙️ Prediksi <span style='color:#38bdf8'>Kerusakan Mesin</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:0.9rem'>Masukkan parameter mesin untuk memprediksi potensi kerusakan</p>", unsafe_allow_html=True)

models, scaler = load_models()

# ─── Form Card ───
st.markdown('<div class="card">', unsafe_allow_html=True)

cols = st.columns(2)

for i, (key, label, vmin, vmax, step, default, badge) in enumerate(SLIDER_PARAMS):
    with cols[i % 2]:
        st.markdown("<div class='form-group-cell'>", unsafe_allow_html=True)
        st.markdown(f"<div class='form-label'>{label}</div>", unsafe_allow_html=True)
        st.slider("", min_value=vmin, max_value=vmax, step=step,
                  key=f"{key}_sld", on_change=_on_slider, args=(key,),
                  label_visibility="collapsed")
        sub = st.columns([3, 1])
        with sub[0]:
            st.number_input("", min_value=vmin, max_value=vmax, step=step,
                           key=f"{key}_num", on_change=_on_num, args=(key,),
                           label_visibility="collapsed")
        with sub[1]:
            st.markdown(f"<div class='range-badge'>{badge}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with cols[1]:
    st.markdown("<div class='form-group-cell'>", unsafe_allow_html=True)
    st.markdown("<div class='form-label'>Type</div>", unsafe_allow_html=True)
    type_label = st.selectbox("", options=list(TYPE_OPTIONS.keys()), index=1,
                              label_visibility="collapsed")
    st.markdown("<div class='range-badge'>Tingkat spesifikasi mesin</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

predict_clicked = st.button("Prediksi Kerusakan")

st.markdown('</div>', unsafe_allow_html=True)

# ─── Prediction ───
if predict_clicked:
    data = {
        "air_temp": st.session_state.air_temp_sld,
        "process_temp": st.session_state.process_temp_sld,
        "rotational_speed": st.session_state.rotational_speed_sld,
        "torque": st.session_state.torque_sld,
        "tool_wear": st.session_state.tool_wear_sld,
        "type": TYPE_OPTIONS[type_label],
    }
    with st.spinner("Memproses..."):
        results = predict(models, scaler, data)
    st.session_state.last_data = data
    st.session_state.last_results = results
    st.session_state.last_type_label = type_label

# ─── Results ───
if st.session_state.get("last_results"):
    data = st.session_state.last_data
    results = st.session_state.last_results
    type_label = st.session_state.last_type_label

    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown("<h2>Hasil Prediksi</h2>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="input-summary">'
        f"<span><strong>Air T:</strong> {data['air_temp']} K</span>"
        f"<span><strong>Process T:</strong> {data['process_temp']} K</span>"
        f"<span><strong>Speed:</strong> {data['rotational_speed']} rpm</span>"
        f"<span><strong>Torque:</strong> {data['torque']} Nm</span>"
        f"<span><strong>Wear:</strong> {data['tool_wear']} min</span>"
        f"<span><strong>Type:</strong> {type_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    rcols = st.columns(3)
    for i, name in enumerate(["xgboost", "random_forest", "ann"]):
        pred = results.get(name)
        label = MODEL_LABELS.get(name, name)
        if pred is None:
            cls, icon, txt = "rci-error", "&#10060;", "Error"
        elif pred == 1:
            cls, icon, txt = "rci-rusak", "&#9888;", "Rusak"
        else:
            cls, icon, txt = "rci-normal", "&#10003;", "Normal"

        with rcols[i]:
            st.markdown(
                f'<div class="rci {cls}">'
                f'<div class="rci-name">{label}</div>'
                f'<div class="rci-row">'
                f'<div class="rci-icon">{icon}</div>'
                f'<div class="rci-status">{txt}</div>'
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="result-note">'
        '<span class="tag tag-normal">Normal</span> = mesin tidak rusak, '
        '<span class="tag tag-rusak">Rusak</span> = berpotensi mengalami kerusakan.'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
