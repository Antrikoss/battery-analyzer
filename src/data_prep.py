from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

RAW_DIR = Path("data/raw")
CELLS = ["B0006", "B0018"]
SIGNALS = ["Time", "Voltage_measured", "Current_measured", "Temperature_measured"]
FIXED_TIMES = [600, 1500, 2400]
THRESHOLDS = [3.7, 3.5, 3.2]
LOAD_CURRENT = 1.0
EOL_CAPACITY = 1.4


def load_data(data_files):
    '''
    Input: the 'B0006.mat' and 'B0018.mat` files from data/raw
    Return: a Pandas dataframe
    
    1. Load files: B0006.mat and B0018.mat
    2. Convert them to pandas dataframe
    3. Merge the two dataframes into one
    4. Find and drop any rows with missing values
    5. Return the cleaned dataframe
    '''
    # dataframes for each cell
    frames = []
    for name in data_files:
        # load data file
        mat = loadmat(RAW_DIR / f"{name}.mat", squeeze_me=True, struct_as_record=False)
        # keep only discharge cycles
        discharge_cycles = [c for c in mat[name].cycle if c.type == "discharge"]

        rows = []
        for k, c in enumerate(discharge_cycles, start=1):
            # add cell and cycle (number/index) columns
            row = {"cell": name, "cycle": k}
            # keep only the columns specified in `SIGNALS`
            for signal in SIGNALS:
                row[signal] = np.atleast_1d(getattr(c.data, signal)).astype(float)
            # add capacity label
            row["Capacity"] = float(c.data.Capacity)
            rows.append(row)
        frames.append(pd.DataFrame(rows))

    # merge the two dataframes
    df = pd.concat(frames, ignore_index=True)

    # Check for rows with missing data
    bad_signal = df[SIGNALS].apply(
        lambda col: col.map(lambda a: a.size == 0 or bool(np.isnan(a).any()))
    ).any(axis=1)
    bad_capacity = df["Capacity"].isna()

    # drop any rows with missing data
    df = df[~(bad_signal | bad_capacity)].reset_index(drop=True)
    # return cleaned dataframe 
    return df


def time_to_threshold(t, v, threshold):
    '''
    Support function to calculate the time when a voltage threshold is
    crossed
    '''
    # get the indices of samples where the voltage is below the threshold
    below = np.where(v <= threshold)[0]
    # check if the threshold is never crossed
    if below.size == 0:
        return np.nan
    # get the first sample that crossed the threshold
    i = below[0]
    # handle edge case if it's the first
    if i == 0:
        return t[0]
    # calculate the time until threshold is crossed assuming voltage drops
    # linearly between the two samples
    return t[i - 1] + (v[i - 1] - threshold) / (v[i - 1] - v[i]) * (t[i] - t[i - 1])


def cycle_features(row):
    ''' 
    Support function to add the new features to each row
    '''
    time, volt = row["Time"], row["Voltage_measured"]
    current, temp = row["Current_measured"], row["Temperature_measured"]

    # find the sample where the load was connected
    loaded = np.abs(current) > LOAD_CURRENT
    if not loaded.any():
        raise ValueError(f"No load detected in {row['cell']} cycle {row['cycle']}")
    onset = int(np.argmax(loaded))

    # keep the time frames, and voltages after the load connects
    t = time[onset:] - time[onset]
    v = volt[onset:]

    # add the new features (columns) for the processed row
    feats = {"v_sag": volt[max(onset - 1, 0)] - volt[onset]}
    for ft in FIXED_TIMES:
        feats[f"v_at_{ft}s"] = np.interp(ft, t, v)
    for th in THRESHOLDS:
        feats[f"t_to_{th}v"] = time_to_threshold(t, v, th)
    feats["v_mean"] = v.mean()
    feats["temp_rise"] = temp.max() - temp[0]
    return feats


def add_features(df):
    '''
    Input: the cleaned dataframe returned from `load_data` function
    Output: the final dataframe which has for columns the new features

    Add features:
    - Initial voltage sag 
    - Voltage at 3 fixed times after the load connects
        - 600 s
        - 1500 s
        - 2400 s
    - Time to reach 3 voltage thresholds
        - 3.7 V
        - 3.5 V
        - 3.2 V
    - Mean discharge voltage
    - Max temperature rise
    Add RUL label to each cycle (we'll drop the rows after the EOL cycle
    in the training)
    '''
    feats = pd.DataFrame([cycle_features(row) for _, row in df.iterrows()])

    out = df[["cell", "cycle"]].join(feats)
    out["capacity"] = df["Capacity"]

    eol_cycle = df[df["Capacity"] <= EOL_CAPACITY].groupby("cell")["cycle"].min()
    out["rul"] = df["cell"].map(eol_cycle) - df["cycle"]
    return out