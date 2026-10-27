# Battery Analyzer

![Capacity fade of cells B0006 and B0018](https://github.com/user-attachments/assets/c353987f-ef17-4586-81aa-b9129858de88)

Estimate the health of a lithium-ion battery from a single discharge cycle: how much capacity it has left (State of Health) and how many cycles it can still deliver (Remaining Useful Life). The project is a small end-to-end data pipeline (data loading, feature extraction, models, validation, dashboard) built as a first step towards battery and powertrain analysis for a humanoid robot.

## Problem

Given measurements taken during one discharge cycle (voltage, current and temperature over time), predict:

- **State of Health (SoH):** the cell's current capacity as a fraction of its rated capacity (2.0 Ah). SoH = capacity / 2.0.
- **Remaining Useful Life (RUL):** the number of discharge cycles left until the cell reaches end of life (EOL), defined as capacity falling to 1.4 Ah (a 30% fade from the rated 2 Ah).

A robot operating in a factory cannot stop to run a lab capacity test. Estimating health from the signals it already measures during normal use is the practical version of this problem.

## Dataset

[NASA Prognostics Center of Excellence: Li-ion Battery Aging Data Set](https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip) (direct download, `BatteryAgingARC-FY08Q4` folder).

Commercial 18650 cells (2 Ah rated) were cycled at room temperature (24 °C): charged at a constant 1.5 A until 4.2 V, then held at 4.2 V until the current dropped to 20 mA. They were discharged at a constant 2 A down to a cutoff voltage, and impedance was measured at intervals. The tests stopped when a cell reached 30% capacity fade.

This project uses **B0006** (168 discharge cycles) and **B0018** (132 discharge cycles). Both share the same 2.5 V discharge cutoff, which keeps their capacities directly comparable. B0005 and B0007 use different cutoffs (2.7 V and 2.2 V) and were left out to avoid having to harmonize capacities. Only discharge cycles are used. The raw `.mat` files go in `data/raw/` (not tracked by git).

## Data

**Exploration.** Before writing the pipeline, two scratch scripts in `data/` examined the files:

- `explore.py` counts discharge cycles per cell, the first and last capacity, whether each cell crosses the 1.4 Ah EOL threshold, and the ambient temperatures. Ambient temperature is constant (24 °C), so it is not used as a feature.
- `check_data.py` prints the structure of a discharge cycle and the ranges of duration, voltage and temperature across all cycles. These ranges set the fixed times and voltage thresholds used for the features. It also compares the dataset's `Capacity` field with a coulomb count (the integral of current over time). The two agree closely (correlation above 0.999, mean absolute gap 0.014 Ah, and the first cycle at or below 1.4 Ah differs by at most 3 cycles), so `Capacity` is used for the labels.

**From time series to features.** Each discharge cycle is a few hundred voltage, current and temperature samples at irregular intervals. They are reduced to one row of 9 features per cycle, with all times measured from the moment the load connects (the first sample where the current exceeds 1 A):

| Feature | Description |
|---|---|
| `v_sag` | Voltage drop when the load connects (reflects internal resistance) |
| `v_at_600s`, `v_at_1500s`, `v_at_2400s` | Voltage at fixed times after load connection (interpolated) |
| `t_to_3.7v`, `t_to_3.5v`, `t_to_3.2v` | Time to first reach each voltage threshold (interpolated) |
| `v_mean` | Mean voltage while loaded |
| `temp_rise` | Maximum temperature minus starting temperature |

The fixed times and thresholds were chosen so that every cycle in both cells reaches them.

**Labels.** `capacity` (Ah), `soh` = capacity / 2.0, and `rul` = (first cycle at or below 1.4 Ah) minus the current cycle. The EOL cycles are 109 for B0006 and 97 for B0018. Rows after EOL are kept for SoH but excluded from RUL training and evaluation. The first crossing is used because capacity occasionally rises slightly between cycles (capacity regeneration, commonly attributed to rest periods between tests).

**A feature that was removed.** An early version included the time to peak temperature. In every cycle the temperature peaks at the very last sample, so the feature was simply the discharge duration. At a constant 2 A, duration times current is the capacity, so the feature handed the model its own label (correlation with capacity of 1.00). It was removed. The same reasoning excluded total discharge duration, the current integral and the discharge energy. Cycle number is also excluded from the features, because with only two cells it would let the model memorize each cell's countdown.

## Model and validation

**Models.** Two `Ridge` regressions (alpha = 1.0), each in a scikit-learn `Pipeline` with a `StandardScaler`:

- SoH model: trained on all rows, predicting `soh`.
- RUL model: trained only on rows at or before EOL, predicting `rul`. Predictions are clipped at 0.

Ridge was chosen because the features are strongly correlated, there is very little data (effectively two degradation trajectories), the model is easy to explain, and unlike tree-based models it can extrapolate beyond the range seen in training. The scaler sits inside the pipeline so it is fit only on training data.

**Validation: leave-one-cell-out.** Rows within a cell are near-duplicates of their neighbors, so a random train/test split would put almost identical cycles on both sides and inflate the scores. Instead, each cell is held out in turn: train on one cell, test on the other.

| Held-out cell | SoH MAE | SoH R² | SoH MAPE | RUL MAE (cycles) | RUL RMSE | RUL R² | RUL bias |
|---|---|---|---|---|---|---|---|
| B0006 | 0.015 | 0.983 | 2.0% | 12.6 | 16.6 | 0.72 | +5.2 |
| B0018 | 0.031 | 0.834 | 4.0% | 8.5 | 9.6 | 0.88 | +4.0 |

SoH is estimated to within about 1.5 to 3 percentage points on a cell the model has never seen. RUL is rougher, at 9 to 13 cycles of error.

**Why RUL is optimistic.** The bias is positive in both folds: the model predicts more remaining life than the cell has. The likely reason is that capacity fades faster as a cell approaches end of life, and a straight line cannot follow that curve, so it overestimates RUL late in life, which is where it matters most. An optimistic error is also the costly direction, since a battery could fail before the model says it will. A safety-minded deployment could subtract a margin from the RUL prediction.

**Limitations.**

- Only two cells, so each fold trains on a single trajectory and the scores are rough estimates.
- The cells start at different capacities (about 2.03 Ah and 1.86 Ah), which adds an offset to SoH predictions.
- Constant 2 A discharge at room temperature. A robot's variable load is not represented in the data.
- Features come from a complete discharge, and `t_to_3.2v` is close to a proxy for discharge duration.

**Possible improvements.**

- More cells (B0005 and B0007, after harmonizing their different cutoffs) for a more reliable validation.
- A nonlinear model for RUL, or a time-series / deep learning model (LSTM or GRU) that uses the history of previous cycles, once there is enough data to validate it honestly.
- History features such as a rolling mean of recent capacity and the recent fade rate.
- Ablation studies (with and without `t_to_3.2v` and cycle number) and a sensitivity check on the Ridge alpha.

## Dashboard

A [Streamlit](https://streamlit.io/) app (`app.py`) with:

- Capacity fade curves for both cells with the 1.4 Ah EOL line, and a mean discharge voltage plot.
- A cell and cycle selector showing predicted versus true SoH and RUL. These are in-sample, because the saved models were trained on both cells.
- The leave-one-cell-out metrics tables, with the limitations beside them.
- A form to enter the 9 features of a new battery and get its predicted SoH and RUL, with a warning when inputs fall outside the training range.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download the dataset (link above) and place B0006.mat and B0018.mat in data/raw/

python -m src.pipeline           # validation tables + trains and saves the final models
streamlit run app.py             # launch the dashboard
```

Run both commands from the project root.

## Project structure

```
battery-analyzer/
├── data/
│   ├── raw/                  # NASA .mat files (not tracked)
│   ├── explore.py            # cycle counts, capacities, EOL crossing
│   └── check_data.py         # cycle format, signal ranges, capacity vs coulomb count
├── docs/
│   └── capacity_fade.png     # dashboard screenshot
├── models/                   # saved SoH and RUL models (generated)
├── src/
│   ├── __init__.py
│   ├── data_prep.py          # load data, extract features, build labels, preprocessor
│   ├── degradation_model.py  # train and evaluate the SoH and RUL models
│   └── pipeline.py           # end-to-end run: validation, final fit, prediction
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Possible next steps

The longer-term goal is to connect the battery model to a robot's workload:

1. Generate data by simulating robot tasks (walking, lifting, carrying) in MuJoCo.
2. Model the electrical power consumed by each task, including motor and transmission losses.
3. Feed the resulting power profile into the battery model to estimate the runtime per charge and the SoH and RUL of the battery over repeated shifts. This needs an explicit assumption for mapping a variable load onto data collected at constant current.
4. Perform a correlation analysis to find which features correlate most with capacity loss. Because consecutive cycles are near-duplicates, correlations on this data look stronger than they would for independent samples, so the leave-one-cell-out score remains the meaningful test.