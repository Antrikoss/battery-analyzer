from pathlib import Path
from scipy.io import loadmat

RAW_DIR = Path("data/raw")
CELLS = ["B0006", "B0018"]

for name in CELLS:
    mat = loadmat(RAW_DIR / f"{name}.mat", squeeze_me=True, struct_as_record=False)
    cycles = mat[name].cycle

    # keep only discharge cycles
    discharge = [c for c in cycles if c.type == "discharge"]

    capacities = [float(c.data.Capacity) for c in discharge]
    ambient_temps = sorted({float(c.ambient_temperature) for c in discharge})

    print(f"--- {name} ---")
    print(f"Total cycles (all types): {len(cycles)}")
    print(f"Discharge cycles:         {len(discharge)}")
    print(f"Capacity first -> last:   {capacities[0]:.3f} -> {capacities[-1]:.3f} Ah")
    print(f"Reaches 1.4 Ah EOL:       {min(capacities) <= 1.4}")
    print(f"Ambient temps (deg C):    {ambient_temps}")