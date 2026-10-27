from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_prep import CELLS, FEATURE_COLS, load_data, add_features, preprocess_data
from src.degradation_model import train_models, evaluate_predictions

MODELS_DIR = Path("models")


def run_validation(X, y, groups, preprocessor):
    '''
    Train and evaluate the SoH and RUL models by running two
    leave-one-cell-out folds
    '''
    rows = []
    for test_cell in CELLS:
        # rows of the test cell (held-out)
        test = groups == test_cell
        # rows of the other cell
        train = ~test
        # train models
        soh_model, rul_model = train_models(X[train], y[train], preprocessor)
        for target, model in [("soh", soh_model), ("rul", rul_model)]:
            metrics = evaluate_predictions(X[test], y[test], model, target)
            rows.append({"held_out": test_cell, "target": target, **metrics})
    return pd.DataFrame(rows)


def fit_final_models(X, y, preprocessor):
    '''
    Train models on all rows
    '''
    soh_model, rul_model = train_models(X, y, preprocessor)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(soh_model, MODELS_DIR / "soh_model.joblib")
    joblib.dump(rul_model, MODELS_DIR / "rul_model.joblib")
    return soh_model, rul_model


def load_models():
    '''
    Load the saved models
    '''
    soh_model = joblib.load(MODELS_DIR / "soh_model.joblib")
    rul_model = joblib.load(MODELS_DIR / "rul_model.joblib")
    return soh_model, rul_model


def predict_health(features, soh_model, rul_model):
    '''
    Run prediction for new data
    '''
    X_new = features[FEATURE_COLS]
    return pd.DataFrame({
        "soh": soh_model.predict(X_new),
        "rul": np.clip(rul_model.predict(X_new), 0, None),
    }, index=features.index)


def main():
    df = add_features(load_data(CELLS))
    X, y, groups, preprocessor = preprocess_data(df)

    results = run_validation(X, y, groups, preprocessor)
    for target in ["soh", "rul"]:
        # keep only relevant columns for SoH and RUL prediction results
        table = results[results["target"] == target].dropna(axis=1, how="all")
        print(f"\n=== {target.upper()} (leave-one-cell-out) ===")
        print(table.drop(columns="target").round(3).to_string(index=False))

    soh_model, rul_model = fit_final_models(X, y, preprocessor)
    print(f"\nFinal models saved to {MODELS_DIR}/")

    sample = df[df["cell"] == "B0018"].iloc[[0, 50, 90]]
    preds = predict_health(sample, soh_model, rul_model)
    demo = pd.DataFrame({
        "cell": sample["cell"],
        "cycle": sample["cycle"],
        "soh_true": sample["capacity"] / 2.0,
        "soh_pred": preds["soh"],
        "rul_true": sample["rul"],
        "rul_pred": preds["rul"],
    })
    print("\n=== Demo predictions (in-sample, not a performance estimate) ===")
    print(demo.round(3).to_string(index=False))


if __name__ == "__main__":
    main()