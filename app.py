import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

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
from src.task_power_model import (
    CELL_CAPACITY_AH,
    DRIVER_EFF,
    MOTOR_EFF,
    N_CELLS,
    V_NOM,
    build_load_profile,
)
from src.discharge_model import SHIFT_HOURS, run_discharge_model

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
    st.session_state["battery_estimate"] = {
        "soh": float(out["soh"]),
        "rul": float(out["rul"]),
        "outside": [
            c for c in FEATURE_COLS
            if not df[c].min() <= values[c] <= df[c].max()
        ],
    }
 
if "battery_estimate" in st.session_state:
    battery = st.session_state["battery_estimate"]
    r1, r2 = st.columns(2)
    r1.metric(
        "Predicted SoH", f"{battery['soh']:.1%}",
        help=f"Equivalent capacity: {battery['soh'] * RATED_CAPACITY:.2f} Ah",
    )
    r2.metric("Predicted RUL", f"{battery['rul']:.0f} cycles")
    if battery["outside"]:
        st.warning(
            "These inputs are outside the range seen in training, so the linear model is "
            "extrapolating and the prediction is unreliable: " + ", ".join(battery["outside"])
        )

st.header("5. Runtime estimate for a task")
st.write("Combine the predicted battery health with the load of a robot joint task.")
 
battery = st.session_state.get("battery_estimate")
if battery is None:
    st.info("Predict a battery's health in section 4 first.")
 
csv_files = sorted(Path("data").glob("*.csv"))
col_file, col_shift = st.columns([2, 1])
with col_file:
    chosen = st.selectbox("Task data file (from data/)", csv_files, format_func=lambda p: p.name)
    uploaded = st.file_uploader("...or upload a CSV (time, torque, angular_velocity)", type="csv")
with col_shift:
    shift_hours = st.number_input("Shift length (hours)", min_value=1.0, value=SHIFT_HOURS, step=1.0)
 
source = uploaded if uploaded is not None else chosen
calculate = st.button("Calculate runtime", disabled=battery is None or source is None)
 
if calculate:
    if battery["soh"] <= 0:
        st.error("The predicted SoH is not positive, so no runtime can be computed. Check the inputs in section 4.")
    else:
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            profile, summary = build_load_profile(source)
        except ValueError as err:
            st.error(f"Could not read the task data: {err}")
        else:
            result = run_discharge_model(summary, battery["soh"], battery["rul"], shift_hours)
 
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Task power (X)", f"{summary['avg_power_w']:.0f} W")
            m2.metric("Battery SoH (Y)", f"{battery['soh']:.1%}")
            m3.metric("Battery RUL (Z)", f"{battery['rul']:.0f} cycles")
            m4.metric("Runtime (T)", f"{result['runtime_h']:.2f} h")
 
            st.success(
                f"This task draws **{summary['avg_power_w']:.0f} W**; this battery measures "
                f"SoH **{battery['soh']:.1%}** and RUL **{battery['rul']:.0f} cycles**; "
                f"so the runtime is **{result['runtime_h']:.2f} hours**."
            )
 
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Energy per task", f"{result['energy_per_task_wh']:.1f} Wh")
            d2.metric("Tasks per charge", f"{result['tasks_per_charge']:.1f}")
            d3.metric("Charge cycles per shift", f"{result['cycles_per_shift']:.1f}")
            d4.metric("Shifts left", f"{result['shifts_left']:.1f}")
 
            st.caption(
                f"Average load {summary['avg_c_rate']:.2f}C, peak {summary['peak_c_rate']:.1f}C "
                "(the battery models were trained at about 1C)."
            )
            if not 0.5 <= summary["avg_c_rate"] <= 1.5:
                st.warning(
                    "The average load is far from the 1C condition the battery models were "
                    "trained on, so treat the SoH and RUL with caution."
                )
 
            fig = px.line(
                profile, x="time", y="p_batt_w",
                render_mode="svg",
                labels={"time": "Time (s)", "p_batt_w": "Battery power (W)"},
            )
            st.plotly_chart(fig)
 
st.caption(
    f"Assumptions: single joint, {N_CELLS}-cell series pack of {CELL_CAPACITY_AH:.0f} Ah cells at "
    f"{V_NOM} V nominal, motor efficiency {MOTOR_EFF}, driver efficiency {DRIVER_EFF}, "
    "no regeneration. The bundled task data is synthetic."
)
 