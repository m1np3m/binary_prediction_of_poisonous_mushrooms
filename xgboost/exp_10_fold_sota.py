#!/usr/bin/env python
# coding: utf-8

# ## Constants

# In[1]:


import sys, os
import pandas as pd
import polars as pl
import numpy as np
import subprocess
import gc
import optuna
from datetime import datetime, timezone
import warnings
import xgboost as xgb
import joblib as jl
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import matthews_corrcoef
from mlflow.models import infer_signature
import mlflow
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from typing import Literal
today = datetime.now(timezone.utc).strftime("%Y_%m_%d")
warnings.filterwarnings("ignore")

from hyper_params import (
    mushroom_tuning_2024_08_06_1722934727_params,
)

sys.path.append("..")
from helpers.loss_functions import generate_submission, prepare_submission

SEED = 108
random.seed(SEED)
N_FOLDS = 7
n_estimators = 10000
early_stop = 100

# model
is_tunning = True
try:
    rs = subprocess.check_output("nvidia-smi")
    device = "cuda" if rs is not None else "cpu"
except (
    Exception
):  # this command not being found can raise quite a few different errors depending on the configuration
    print("No Nvidia GPU in system!")
    device = "cpu"

best_params = {
    "device": device,
    "verbosity": 0,
    "objective": "binary:logistic",
}
best_params.update(mushroom_tuning_2024_08_06_1722934727_params)
best_params


# ## Prepare data

# In[ ]:


y_train_pkl = jl.load("../y_train.pkl")
X_train_pkl = jl.load("../X_train.pkl")

print(f"train size: {X_train_pkl.shape}")


# ## CV

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


clf: xgb.XGBClassifier = xgb.XGBClassifier(
    **best_params,
    n_estimators=n_estimators,
    early_stopping_rounds=early_stop,
    enable_categorical=True,
)


# In[ ]:


from tqdm import tqdm

gc.collect()
skf = StratifiedKFold(n_splits=N_FOLDS)

y_preds = []
y_trues = []
for train_index, test_index in tqdm(skf.split(X_train_pkl, y_train_pkl)):
    X_train, X_test = X_train_pkl[train_index], X_train_pkl[test_index]
    y_train, y_test = y_train_pkl[train_index], y_train_pkl[test_index]

    clf.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])

    y_pred = clf.predict(X_test)
    y_preds.append(y_pred)
    y_trues.append(y_test)

    del X_train, X_test, y_train, y_test, y_pred
    gc.collect()
# Concatenate the predictions and true labels
y_preds_concat = np.concatenate(y_preds)
y_trues_concat = np.concatenate(y_trues)
mcc = matthews_corrcoef(y_trues_concat, y_preds_concat)
jl.dump(clf, "../clf_xgboost.pkl")


# In[ ]:


print(f"Validation mcc score: {mcc}")


# In[ ]:


import joblib as jl

X_test_pkl = jl.load("../X_test.pkl")
proba_preds = clf.predict_proba(X_test_pkl)
generate_submission(
    model=clf,
    X_test=X_test_pkl,
    unique_target_values=["e", "p"],
)


# In[9]:


prepare_submission(strategy="mean")

