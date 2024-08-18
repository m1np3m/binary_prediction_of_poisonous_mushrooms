import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Tuple
from typing import Literal
from sklearn.metrics import matthews_corrcoef
import gc
from catboost import CatBoostClassifier, Pool
import joblib as jl
from scipy.stats import mode


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


def handle_categorical_columns(df: pd.DataFrame, n_common_values=15):
    categorical_cols = df.select_dtypes(include="object").columns.to_list()
    # get top 10 most frequent names
    n = n_common_values
    for c in categorical_cols:
        train_mode_values = df[c].value_counts()[:n].index.tolist()
        df.loc[~df[c].isin(train_mode_values), c] = "other"
        df[c] = pd.Series(df[c], dtype="category")
        gc.collect()
    return df, categorical_cols


def generate_submission(
    model,
    X_test,
    unique_target_values=None,
    categorical_cols=None,
):
    if isinstance(model, CatBoostClassifier) and categorical_cols is not None:
        test_pool = Pool(
            data=X_test,
            cat_features=categorical_cols,
        )
        preds_proba = model.predict_proba(test_pool)
        columns = [f"catboost_{c}" for c in unique_target_values]
        preds_df = pd.DataFrame(preds_proba, columns=columns)
    if isinstance(model, xgb.XGBClassifier):
        preds_proba = model.predict_proba(X_test)
        columns = [f"xgboost_{c}" for c in unique_target_values]
        preds_df = pd.DataFrame(preds_proba, columns=columns)

    submit_df: pd.DataFrame = jl.load("../submit_df.pkl")
    try:
        submit_df.drop(columns=preds_df.columns.to_list(), inplace=True)
    except:
        pass
    submit_df = pd.concat([submit_df, preds_df], axis=1)
    jl.dump(submit_df, "../submit_df.pkl")


def prepare_submission(strategy: Literal["mean"] = "mean"):
    submit_pkl: pd.DataFrame = jl.load("../submit_df.pkl")
    lb = jl.load("../lb.pkl")

    # get columns
    submission_columns = submit_pkl.columns.to_list()
    e_cols = [c for c in submission_columns if c.endswith("_e")]
    p_cols = [c for c in submission_columns if c.endswith("_p")]

    # finalize probability
    if strategy == "mean":
        submit_pkl["final_e_prob"] = submit_pkl[e_cols].mean(axis=1)
        submit_pkl["final_p_prob"] = submit_pkl[p_cols].mean(axis=1)
    class_prob = submit_pkl[["final_e_prob", "final_p_prob"]].to_numpy()

    # usng threshold
    class_prob_int = np.argmax(class_prob, axis=1)
    classes = lb.inverse_transform(class_prob_int)
    # convert to classes
    submit_pkl["classes"] = classes
    submit_pkl.iloc[:, [0, -1]].to_csv("../submission.csv", index=False)
