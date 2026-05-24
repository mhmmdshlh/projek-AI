const modelLabels = {
    xgboost: "XGBoost",
    random_forest: "Random Forest",
    ann: "ANN (Neural Network)",
};

const sliderMap = [
    { slider: "air_temp_slider", input: "air_temp" },
    { slider: "process_temp_slider", input: "process_temp" },
    { slider: "rotational_speed_slider", input: "rotational_speed" },
    { slider: "torque_slider", input: "torque" },
    { slider: "tool_wear_slider", input: "tool_wear" },
];

document.addEventListener("DOMContentLoaded", () => {
    for (const { slider: sId, input: iId } of sliderMap) {
        const slider = document.getElementById(sId);
        const input = document.getElementById(iId);
        slider.addEventListener("input", () => { input.value = slider.value; });
        input.addEventListener("input", () => { slider.value = input.value; });
    }

    const form = document.getElementById("predictionForm");
    const resultsEl = document.getElementById("results");
    const resultsContainer = document.getElementById("resultCards");
    const inputSummary = document.getElementById("inputSummary");
    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const errorEl = document.getElementById("errorMsg");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        errorEl.classList.add("hidden");
        resultsEl.classList.add("hidden");

        const data = {
            air_temp: parseFloat(document.getElementById("air_temp").value),
            process_temp: parseFloat(document.getElementById("process_temp").value),
            rotational_speed: parseFloat(document.getElementById("rotational_speed").value),
            torque: parseFloat(document.getElementById("torque").value),
            tool_wear: parseInt(document.getElementById("tool_wear").value, 10),
            type: parseInt(document.getElementById("type").value, 10),
        };

        const typeLabel = { 0: "H (High)", 1: "L (Low)", 2: "M (Medium)" }[data.type] || "Unknown";

        submitBtn.disabled = true;
        btnText.innerHTML = '<span class="spinner"></span>Memproses...';

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Server error");
            }

            const results = await res.json();

            inputSummary.innerHTML = `
                <span><strong>Air T:</strong> ${data.air_temp} K</span>
                <span><strong>Process T:</strong> ${data.process_temp} K</span>
                <span><strong>Speed:</strong> ${data.rotational_speed} rpm</span>
                <span><strong>Torque:</strong> ${data.torque} Nm</span>
                <span><strong>Wear:</strong> ${data.tool_wear} min</span>
                <span><strong>Type:</strong> ${typeLabel}</span>
            `;

            resultsContainer.innerHTML = "";
            for (const [model, result] of Object.entries(results)) {
                const card = document.createElement("div");

                if (result.error) {
                    card.className = "result-card error";
                    card.innerHTML = `
                        <div class="model-name">${modelLabels[model] || model}</div>
                        <div class="status" style="color:#f59e0b">Error</div>
                        <div style="color:#f59e0b;font-size:0.82rem">${result.error}</div>
                    `;
                } else {
                    const isRusak = result.prediction === 1;
                    const cls = isRusak ? "rusak" : "normal";
                    const icon = isRusak ? "&#9888;" : "&#10003;";

                    card.className = `result-card ${cls}`;
                    card.innerHTML = `
                        <div class="model-name">${modelLabels[model] || model}</div>
                        <div class="status-row">
                            <div class="status-icon">${icon}</div>
                            <div class="status">${result.label}</div>
                        </div>
                    `;
                }

                resultsContainer.appendChild(card);
            }

            resultsEl.classList.remove("hidden");
            resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            errorEl.textContent = err.message;
            errorEl.classList.remove("hidden");
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Prediksi Kerusakan";
        }
    });
});