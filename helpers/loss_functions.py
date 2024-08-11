import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple
from sklearn.metrics import matthews_corrcoef


def matthews_corrcoef_score(model, data, y_true=None, lb=None):

    y_pred_prob = model.predict(data)
    y_pred_binary = np.round(y_pred_prob).astype(int)
    try:
        classes = lb.inverse_transform(y_pred_binary)
    except Exception as e:
        classes = None
        # print(f"Failed to convert y_pred to original form. Detail: {e}")
    try:
        mcc = matthews_corrcoef(y_true, y_pred_binary)
    except Exception as e:
        mcc = None
        # print(f"Failed to calculate matthews_corrcoef. Detail: {e}")

    try:
        validate_df = pd.DataFrame()
        validate_df["actual_value"] = lb.inverse_transform(y_true)
        validate_df["predicted_value"] = classes
    except:
        pass
    return mcc, classes, validate_df


# Custom MCC metric function for XGBoost
def mcc_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """Custom Matthews correlation coefficient metric for XGBoost."""
    y_true = dtrain.get_label()
    y_pred_binary = np.round(predt).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    return "MCC", mcc


def mcc_metric_v2(preds, dtrain):
    labels = dtrain.get_label()
    preds = (preds > 0.51).astype(int)
    return "MCC", matthews_corrcoef(labels, preds)
