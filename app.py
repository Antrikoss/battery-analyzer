import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_prep import (
    CELLS,
    EOL_CAPACITY,
    FEATURE_COLS,
    RATED_CAPACITY,
    add_features,
    load_data,
    preprocess_data,
)
from src.pipeline import load_models, predict_health, run_validation

FEATURE_INFO = {
    "v_sag": ("Initial voltage sag", "V", 0.01, "%.3f"),
    "v_at_600s": ("Voltage at 600 s", "V", 0.01, "%.3f"),
    "v_at_1500s": ("Voltage at 1500 s", "V", 0.01, "%.3f"),
    "v_at_2400s": ("Voltage at 2400 s", "V", 0.01, "%.3f"),
    "t_to_3.7v": ("Time to reach 3.7 V", "s", 10.0, "%.1f"),
    "t_to_3.5v": ("Time to reach 3.5 V", "s", 10.0, "%.1f"),
    "t_to_3.2v": ("Time to reach 3.2 V", "s", 10.0, "%.1f"),
    "v_mean": ("Mean discharge voltage", "V", 0.01, "%.3f"),
    "temp_rise": ("Max temperature rise", "°C", 0.1, "%.2f"),
}

st.set_page_config(page_title="Battery Health Analyzer", layout="wide")


@st.cache_data
def get_data():
    return add_features(load_data(CELLS))


@st.cache_data
def get_validation():
    X, y, groups, preprocessor = preprocess_data(get_data())
    return run_validation(X, y, groups, preprocessor)


@st.cache_resource
def get_models():
    return load_models()


df = get_data()

try:
    soh_model, rul_model = get_models()
except FileNotFoundError:
    st.error("Trained models not found. Run `python -m src.pipeline` first.")
    st.stop()

st.title("Battery Health Analyzer")
st.caption(
    "State of Health (SoH) and Remaining Useful Life (RUL) prediction for "
    "Li-ion cells, trained on NASA cells B0006 and B0018."
)

st.header("1. Degradation curves")
tab_cap, tab_volt = st.tabs(["Capacity fade", "Mean discharge voltage"])

with tab_cap:
    fig = px.line(
        df, x="cycle", y="capacity", color="cell",
        labels={"cycle": "Discharge cycle", "capacity": "Capacity (Ah)", "cell": "Cell"},
    )
    fig.add_hline(
        y=EOL_CAPACITY, line_dash="dash",
        annotation_text=f"End of life ({EOL_CAPACITY} Ah)",
    )
    st.plotly_chart(fig)

with tab_volt:
    fig = px.line(
        df, x="cycle", y="v_mean", color="cell",
        labels={"cycle": "Discharge cycle", "v_mean": "Mean discharge voltage (V)", "cell": "Cell"},
    )
    st.plotly_chart(fig)
    st.caption("Mean voltage from the moment the load connects. It is one of the model's input features.")

st.header("2. Predicted vs. true health for a single cycle")
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    cell = st.selectbox("Cell", CELLS)
cell_df = df[df["cell"] == cell]
with col_sel2:
    cycle = st.slider(
        "Discharge cycle",
        int(cell_df["cycle"].min()),
        int(cell_df["cycle"].max()),
        int(cell_df["cycle"].min()),
    )

row = cell_df[cell_df["cycle"] == cycle]
pred = predict_health(row, soh_model, rul_model).iloc[0]
soh_true = row["capacity"].iloc[0] / RATED_CAPACITY
rul_true = int(row["rul"].iloc[0])

m1, m2, m3, m4 = st.columns(4)
m1.metric("True SoH", f"{soh_true:.1%}")
m2.metric(
    "Predicted SoH", f"{pred['soh']:.1%}",
    delta=f"{(pred['soh'] - soh_true) * 100:+.1f} pp vs true", delta_color="off",
)
if rul_true >= 0:
    m3.metric("True RUL", f"{rul_true} cycles")
    m4.metric(
        "Predicted RUL", f"{pred['rul']:.0f} cycles",
        delta=f"{pred['rul'] - rul_true:+.0f} cycles vs true", delta_color="off",
    )
else:
    m3.metric("True RUL", "Past EOL")
    m4.metric("Predicted RUL", f"{pred['rul']:.0f} cycles")
st.caption(
    "These predictions are in-sample: the saved models were trained on both cells, "
    "so they look better than they would on a new battery. See the validation table below."
)

st.header("3. Validation: leave-one-cell-out")
results = get_validation()
left, right = st.columns([2, 1])

with left:
    for target, title in [("soh", "SoH"), ("rul", "RUL")]:
        table = results[results["target"] == target].dropna(axis=1, how="all")
        st.subheader(title)
        st.dataframe(table.drop(columns="target").round(3), hide_index=True)
    st.caption(
        "Each row trains on one cell and tests on the other. "
        "MAE and RMSE are in SoH fraction (0.015 = 1.5 points) and in cycles. "
        "Bias > 0 means the model predicts more remaining life than the cell has."
    )

with right:
    st.subheader("Limitations")
    st.markdown(
        """
- **Only two cells.** Each fold trains on a single degradation trajectory, so the scores are rough estimates, not robust ones.
- **RUL is optimistic.** The bias is positive in both folds, and the linear model can't follow the faster fade near end of life.
- **Different starting capacities** between the cells (about 2.03 vs 1.86 Ah) add an offset to SoH predictions.
- **Lab conditions.** Constant 2 A discharge at room temperature. A robot's variable load is not modeled.
        """
    )

st.header("4. Predict health for a new battery")
st.write("Enter the features measured from one discharge cycle (defaults are the dataset medians).")

with st.form("new_battery"):
    cols = st.columns(3)
    values = {}
    for i, col in enumerate(FEATURE_COLS):
        label, unit, step, fmt = FEATURE_INFO[col]
        with cols[i % 3]:
            values[col] = st.number_input(
                f"{label} ({unit})",
                value=float(df[col].median()),
                step=step,
                format=fmt,
                key=f"in_{col}",
            )
    submitted = st.form_submit_button("Predict health")

if submitted:
    new = pd.DataFrame([values])
    out = predict_health(new, soh_model, rul_model).iloc[0]

    r1, r2 = st.columns(2)
    r1.metric("Predicted SoH", f"{out['soh']:.1%}", help=f"Equivalent capacity: {out['soh'] * RATED_CAPACITY:.2f} Ah")
    r2.metric("Predicted RUL", f"{out['rul']:.0f} cycles")

    outside = [
        c for c in FEATURE_COLS
        if not df[c].min() <= values[c] <= df[c].max()
    ]
    if outside:
        st.warning(
            "These inputs are outside the range seen in training, so the linear model is "
            "extrapolating and the prediction is unreliable: " + ", ".join(outside)
        )