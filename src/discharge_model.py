from pathlib import Path

import pandas as pd

from src.task_power_model import (
    CELL_CAPACITY_AH,
    N_CELLS,
    V_NOM,
    build_load_profile,
)

SHIFT_HOURS = 8.0


def energy_per_task(summary):
    '''
    Input: load summary dict from `summarize_load`
    Output: energy drawn from the battery for one run of the task (Wh)
    '''
    return summary["avg_power_w"] * summary["duration_s"] / 3600


def usable_energy(soh):
    '''
    Input: state of health as a fraction (1.0 = new pack)
    Output: energy the pack can deliver (Wh)

    E = SoH * cell capacity * nominal voltage * number of cells in series
    '''
    return soh * CELL_CAPACITY_AH * V_NOM * N_CELLS


def estimate_runtime(summary, soh):
    '''
    Input: load summary dict and SoH (fraction)
    Output: runtime on one full charge (hours)

    runtime = usable energy / average power
    '''
    if summary["avg_power_w"] <= 0:
        raise ValueError("Average power must be positive")
    return usable_energy(soh) / summary["avg_power_w"]


def estimate_life_use(runtime_h, rul, shift_hours=SHIFT_HOURS):
    '''
    Input: runtime per charge (h), RUL (cycles), shift length (h)
    Output: (cycles used per shift, shifts left before end of life)

    cycles per shift = shift hours / runtime (equivalent full charge cycles)
    shifts left = RUL / cycles per shift
    '''
    cycles_per_shift = shift_hours / runtime_h
    return cycles_per_shift, rul / cycles_per_shift


def run_discharge_model(summary, soh, rul, shift_hours=SHIFT_HOURS):
    '''
    Input: load summary dict, SoH (fraction), RUL (cycles), shift length (h)
    Output: dict with the discharge model results
    '''
    runtime_h = estimate_runtime(summary, soh)
    cycles_per_shift, shifts_left = estimate_life_use(runtime_h, rul, shift_hours)
    return {
        "energy_per_task_wh": energy_per_task(summary),
        "usable_energy_wh": usable_energy(soh),
        "runtime_h": runtime_h,
        "tasks_per_charge": usable_energy(soh) / energy_per_task(summary),
        "cycles_per_shift": cycles_per_shift,
        "shifts_left": shifts_left,
    }
