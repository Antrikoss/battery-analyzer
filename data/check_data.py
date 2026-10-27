from pathlib import Path

import numpy as np
from scipy.io import loadmat

RAW_DIR = Path("data/raw")
CELLS = ["B0006", "B0018"]
FIELDS = ["Time", "Voltage_measured", "Current_measured", "Temperature_measured"]


def load_discharge(name):
    mat = loadmat(RAW_DIR / f"{name}.mat", squeeze_me=True, struct_as_record=False)
    return [c for c in mat[name].cycle if c.type == "discharge"]

def check_capacity(df):
    '''
    Check the gap between `Capacity` value and the coulomb count
    '''
    integrate = getattr(np, "trapezoid", None) or np.trapz
    df = df.copy()
    df["capacity_coulomb"] = [
        integrate(np.abs(i), t) / 3600
        for t, i in zip(df["Time"], df["Current_measured"])
    ]
    df["capacity_gap"] = df["Capacity"] - df["capacity_coulomb"]

    for name, g in df.groupby("cell"):
        g = g.sort_values("cycle")
        corr = g["Capacity"].corr(g["capacity_coulomb"])

        def first_eol(col):
            hits = g.loc[g[col] <= 1.4, "cycle"]
            return int(hits.iloc[0]) if len(hits) else None

        print(f"\n--- {name} ---")
        print(f"Correlation (Capacity vs coulomb count): {corr:.4f}")
        print(f"Gap (Capacity - coulomb): mean={g['capacity_gap'].mean():+.3f}  "
              f"mean abs={g['capacity_gap'].abs().mean():.3f}  "
              f"max abs={g['capacity_gap'].abs().max():.3f} Ah")
        print(f"Mean gap, first 20 cycles: {g['capacity_gap'].head(20).mean():+.3f} Ah")
        print(f"Mean gap, last 20 cycles:  {g['capacity_gap'].tail(20).mean():+.3f} Ah")
        print(f"Last cycle: Capacity={g['Capacity'].iloc[-1]:.3f}  "
              f"coulomb={g['capacity_coulomb'].iloc[-1]:.3f} Ah")
        print(f"First cycle at or below 1.4 Ah: Capacity -> {first_eol('Capacity')}, "
              f"coulomb -> {first_eol('capacity_coulomb')}")
        worst = g.reindex(g["capacity_gap"].abs().sort_values(ascending=False).index).head(5)
        print("5 largest gaps:")
        print(worst[["cycle", "Capacity", "capacity_coulomb", "capacity_gap"]]
              .round(3).to_string(index=False))

    return df

# ---------- Part 1: format of one cycle ----------
first = load_discharge(CELLS[0])[0]
print("=== FORMAT OF ONE DISCHARGE CYCLE (B0006, first discharge) ===")
print(f"type:                {first.type}")
print(f"ambient_temperature: {first.ambient_temperature}")
print(f"time (start date):   {first.time}")
print(f"data fields:         {first.data._fieldnames}")
print(f"Capacity:            {first.data.Capacity}  (type: {type(first.data.Capacity).__name__})")

for f in FIELDS:
    arr = np.atleast_1d(getattr(first.data, f))
    print(f"\n{f}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  first 6 values: {np.round(arr[:6], 4)}")
    print(f"  last 3 values:  {np.round(arr[-3:], 4)}")
    print(f"  min={arr.min():.4f}  max={arr.max():.4f}")

t = np.atleast_1d(first.data.Time)
print(f"\nTime steps in this cycle: min={np.diff(t).min():.2f}s  "
      f"median={np.median(np.diff(t)):.2f}s  max={np.diff(t).max():.2f}s")

# ---------- Part 2: ranges across all discharge cycles ----------
print("\n\n=== RANGES ACROSS ALL DISCHARGE CYCLES ===")
for name in CELLS:
    cycles = load_discharge(name)
    dur, v_start, v_end, v_sag_ref = [], [], [], []
    t_start, t_max, n_samples = [], [], []

    for c in cycles:
        d = c.data
        t = np.atleast_1d(d.Time).astype(float)
        v = np.atleast_1d(d.Voltage_measured).astype(float)
        temp = np.atleast_1d(d.Temperature_measured).astype(float)
        dur.append(t[-1])
        v_start.append(v[0])
        v_sag_ref.append(v[min(3, len(v) - 1)])  # a few samples in
        v_end.append(v.min())
        t_start.append(temp[0])
        t_max.append(temp.max())
        n_samples.append(len(t))

    dur, v_start, v_end, v_sag_ref = map(np.array, (dur, v_start, v_end, v_sag_ref))
    t_start, t_max = np.array(t_start), np.array(t_max)

    print(f"\n--- {name} ({len(cycles)} discharge cycles) ---")
    print(f"Duration (s):          min={dur.min():.0f}  max={dur.max():.0f}  "
          f"-> fixed times must be below {dur.min():.0f}")
    print(f"Samples per cycle:     min={min(n_samples)}  max={max(n_samples)}")
    print(f"First voltage (V):     min={v_start.min():.3f}  max={v_start.max():.3f}")
    print(f"Voltage after ~3 samples: min={v_sag_ref.min():.3f}  max={v_sag_ref.max():.3f}")
    print(f"Lowest voltage (V):    min={v_end.min():.3f}  max={v_end.max():.3f}")
    print(f"  -> thresholds must be above {v_end.max():.3f} V (crossed in every cycle)"
          f" and below {v_sag_ref.min():.3f} V")
    print(f"Start temp (C):        min={t_start.min():.2f}  max={t_start.max():.2f}")
    print(f"Max temp (C):          min={t_max.min():.2f}  max={t_max.max():.2f}")