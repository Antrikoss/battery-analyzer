from pathlib import Path

import numpy as np
import pandas as pd

MOTOR_EFF = 0.8
DRIVER_EFF = 0.95
N_CELLS = 19
V_NOM = 3.55
CELL_CAPACITY_AH = 2.0
V_PACK = N_CELLS * V_NOM

REQUIRED_COLS = ["time", "torque", "angular_velocity"]
MAX_TORQUE_NM = 1000.0
MAX_SPEED_RAD_S = 100.0


def load_joint_data(data_file):
    '''
    Input: CSV file with columns time (s), torque (Nm), angular_velocity (rad/s)
    Output: 1 dataframe with the same columns, time starting at 0
    '''
    df = pd.read_csv(data_file)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {data_file}: {missing}")

    df = df[REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Found missing or non-numeric values")
    if (np.diff(df["time"].to_numpy()) <= 0).any():
        raise ValueError("`time` must be strictly increasing")
    if df["torque"].abs().max() > MAX_TORQUE_NM:
        raise ValueError(f"|torque| above {MAX_TORQUE_NM} Nm, check the units")
    if df["angular_velocity"].abs().max() > MAX_SPEED_RAD_S:
        raise ValueError(
            f"|angular_velocity| above {MAX_SPEED_RAD_S} rad/s, "
            "check the units (rpm or deg/s instead of rad/s?)"
        )

    df["time"] = df["time"] - df["time"].iloc[0]
    return df.reset_index(drop=True)


def convert_mech_to_elec(df):
    '''
    Input: dataframe from `load_joint_data`
    Output: copy of the dataframe with two new columns
        - p_mech_w: mechanical power at the joint (can be negative)
        - p_batt_w: power drawn from the battery (never negative)

    P_mech = torque * angular_velocity
    P_batt = max(P_mech, 0) / (MOTOR_EFF * DRIVER_EFF)
    Negative mechanical power (braking) is clipped to zero: no regeneration.
    '''
    out = df.copy()
    out["p_mech_w"] = out["torque"] * out["angular_velocity"]
    out["p_batt_w"] = out["p_mech_w"].clip(lower=0) / (MOTOR_EFF * DRIVER_EFF)
    return out


def compute_current_and_crate(df):
    '''
    Input: dataframe from `convert_mech_to_elec`
    Output: copy of the dataframe with two new columns
        - current_a: pack current (A), equal to the current of each cell
          because the pack is a single series string
        - c_rate: current_a / CELL_CAPACITY_AH

    I = P_batt / V_PACK, with V_PACK = N_CELLS * V_NOM (constant nominal voltage)
    '''
    out = df.copy()
    out["current_a"] = out["p_batt_w"] / V_PACK
    out["c_rate"] = out["current_a"] / CELL_CAPACITY_AH
    return out


def summarize_load(df):
    '''
    Input: dataframe from `compute_current_and_crate`
    Output: dict with the main load statistics (time-weighted averages)
    '''
    t = df["time"].to_numpy()
    duration = t[-1] - t[0]
    integrate = getattr(np, "trapezoid", None) or np.trapz
    avg_power = integrate(df["p_batt_w"].to_numpy(), t) / duration

    return {
        "duration_s": duration,
        "avg_power_w": avg_power,
        "peak_power_w": df["p_batt_w"].max(),
        "avg_current_a": avg_power / V_PACK,
        "avg_c_rate": avg_power / V_PACK / CELL_CAPACITY_AH,
        "peak_c_rate": df["c_rate"].max(),
        "braking_fraction": float((df["p_mech_w"] < 0).mean()),
    }


def build_load_profile(data_file):
    '''
    Input: CSV file with joint torque and angular velocity
    Output: (profile dataframe, summary dict)
    '''
    df = load_joint_data(data_file)
    df = convert_mech_to_elec(df)
    df = compute_current_and_crate(df)
    return df, summarize_load(df)
