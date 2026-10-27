import numpy as np

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

ALPHA = 1.0


def train_models(X_train, y_train, preprocessor):
    '''
    Input: the features, labels, cell identifiers, and preprocessor
    Output: two trained models
        1. model for SoH prediction
        2. model for RUL prediction
    
    1. Initialize models
    2. Create and fit a pipeline for each model with the `preprocessor`
    3. Return trained models
    ''' 
    soh_model = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", Ridge(alpha=ALPHA)),
    ])
    rul_model = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", Ridge(alpha=ALPHA)),
    ])

    soh_model.fit(X_train, y_train["soh"])

    before_eol = y_train["rul"] >= 0
    rul_model.fit(X_train[before_eol], y_train.loc[before_eol, "rul"])

    return soh_model, rul_model


def evaluate_predictions(X_test, y_test, model, target):
    '''
    Input: test features, test labels, a trained model, and the target
    name ("soh" or "rul")
    Output: a dict of metrics
        - soh: MAE, RMSE, MAPE (%), R2
        - rul: MAE, RMSE, R2, bias (mean predicted - true)
    '''
    if target == "rul":
        before_eol = y_test["rul"] >= 0
        X_test, y_test = X_test[before_eol], y_test[before_eol]

    y_true = y_test[target]
    y_pred = model.predict(X_test)
    if target == "rul":
        y_pred = np.clip(y_pred, 0, None)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
    if target == "soh":
        metrics["MAPE_%"] = mean_absolute_percentage_error(y_true, y_pred) * 100
    else:
        metrics["bias"] = float(np.mean(y_pred - y_true))
    return metrics
