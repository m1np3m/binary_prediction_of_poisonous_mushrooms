#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import sys, os
from catboost import CatBoostRegressor, cv, Pool, CatBoostClassifier
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import numpy as np
import joblib as jl
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    LabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.metrics import matthews_corrcoef

from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split
import random
from datetime import datetime

# helpers
sys.path.append("..")
from helpers.loss_functions import (
    update_submission,
)

now = int(datetime.now().timestamp())
SEED = 108
N_FOLDS = 5
random.seed(SEED)
train_path = "/home/manpm/Developers/kaggle/data/mushrooms/train.csv"
test_path = "/home/manpm/Developers/kaggle/data/mushrooms/test.csv"


# In[2]:


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


# In[3]:


# Prepare data
train = pd.read_csv(train_path)
print(f"train size: {train.shape}")
X_test = pd.read_csv(test_path)
print(f"test size: {X_test.shape}")
X_test.drop(columns=["id"], inplace=True)

# prepare columns
target = "class"

X_train = train.drop(columns=[target, "id"], axis=1)
y_train = train[target]
# Binarize the target labels
lb = LabelBinarizer()

y_train = lb.fit_transform(y_train)

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.2, random_state=42
# )


# Category handling
X_train, categorical_training_cols = handle_categorical_columns(X_train)
X_test, categorical_test_cols = handle_categorical_columns(X_test)
# X_val, categorical_val_cols = handle_categorical_columns(X_val)
# test_pool = Pool(
#     X_test,
#     cat_features=categorical_test_cols,
# )
gc.collect()


# In[4]:


lb.classes_.tolist()


# In[5]:


from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

gc.collect()
skf = StratifiedKFold(n_splits=N_FOLDS)

y_preds = []
y_trues = []
X = X_train.to_numpy()
for train_index, test_index in tqdm(skf.split(X, y_train)):
    X_train_splitted, X_test_splitted = (
        X_train.loc[train_index],
        X_train.loc[test_index],
    )
    y_train_splitted, y_test_splitted = y_train[train_index], y_train[test_index]

    train_pool = Pool(
        X_train_splitted,
        label=y_train_splitted,
        cat_features=categorical_training_cols,
    )
    val_pool = Pool(
        X_test_splitted,
        label=y_test_splitted,
        cat_features=categorical_training_cols,
    )

    model = CatBoostClassifier(
        iterations=10000,
        learning_rate=0.0696294726051571,
        l2_leaf_reg=0.9746051811186938,
        loss_function="Logloss",
        min_data_in_leaf=1,
        task_type="GPU",
    )
    # train the model
    model.fit(
        train_pool,
        use_best_model=True,
        eval_set=val_pool,
        metric_period=100,
        early_stopping_rounds=50,
    )
    y_pred = model.predict(val_pool)
    y_preds.append(y_pred)
    y_trues.append(y_test_splitted)
# Concatenate the predictions and true labels
y_preds_concat = np.concatenate(y_preds)
y_trues_concat = np.concatenate(y_trues)
jl.dump(model, f"catboost_clf_{now}.pkl")


# In[6]:


mcc = matthews_corrcoef(y_trues_concat, y_preds_concat)
print(f"Validation mcc score: {mcc}")


# In[7]:


update_submission(
    model=model,
    X_test=X_test,
    unique_target_values=lb.classes_.tolist(),
    categorical_cols=categorical_test_cols,
)

