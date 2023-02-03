import os
from datetime import datetime

import numpy as np
import pandas as pd
DATA_ROOT = f"../data"
df = pd.read_pickle(f"{DATA_ROOT}/fselect/accidents_raw.pkl")
df.head(3)
# convert time to datetime
df["Start_Time"] = pd.to_datetime(df["Start_Time"])

# The task is to predict the impact of accident on traffic from January 2020 to June 2020
df = df.sort_values(by=["Start_Time"], ascending=True, ignore_index=True)
test_data_start_date = pd.to_datetime("2020-01-01")
test_data_end_date = pd.to_datetime("2020-07-01")
print(test_data_end_date)
df
df_test = df[
    (df["Start_Time"] >= test_data_start_date) & (df["Start_Time"] < test_data_end_date)
]

df_test.shape
df_train = df.iloc[~df.index.isin(df_test.index)]

df_train.shape
# check if train test split has any intersections

set(df_train["ID"]).intersection(set(df_test["ID"]))
os.makedirs(f"{DATA_ROOT}/train/raw/", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/test/raw/", exist_ok=True)
df_train.reset_index(drop=True).to_pickle(f"{DATA_ROOT}/train/raw/data.pkl")
df_test.reset_index(drop=True).to_pickle(f"{DATA_ROOT}/test/raw/data.pkl")
