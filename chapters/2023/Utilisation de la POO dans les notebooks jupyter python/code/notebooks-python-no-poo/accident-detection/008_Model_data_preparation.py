import os

import numpy as np
import pandas as pd
DATA_ROOT = f"../data"
df_train = pd.read_pickle(f"{DATA_ROOT}/train/featurized/data.pkl")
df_test = pd.read_pickle(f"{DATA_ROOT}/test/featurized/data.pkl")
df_train.iloc[0]
categorical_features_as_vector_columns = [
    # transformed
    "Source",
    "Side",
    "City",
    "County",
    "State",
    "Timezone",
    "Airport_Code",
    "Wind_Direction",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
    # engineered and transformed
    "kw_vec",
    "zip_02",
    "zip_25",
]
def prepare_vector_columns_for_model(df, vector_column):

    if vector_column == "kw_vec":
        kw_list = [
            "kw_" + i[0] for i in pd.read_pickle(f"{DATA_ROOT}/train/keywords.pkl")[:15]
        ]

        df_vec = pd.DataFrame(df[vector_column].tolist(), columns=kw_list)

    else:
        cols = [f"{vector_column}_{i}" for i in range(len(df[vector_column].iloc[0]))]
        df_vec = pd.DataFrame(df[vector_column].tolist(), columns=cols)

    df_final = pd.concat([df_vec, df], axis="columns").drop(columns=[vector_column])
    return df_final


for c in categorical_features_as_vector_columns:
    df_train = prepare_vector_columns_for_model(df_train, vector_column=c)
    df_test = prepare_vector_columns_for_model(df_test, vector_column=c)
df_train.columns.tolist()
df_train.drop(columns=["ID"], inplace=True, errors="ignore")
df_test.drop(columns=["ID"], inplace=True, errors="ignore")
os.makedirs(f"{DATA_ROOT}/train/model/", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/test/model/", exist_ok=True)
df_train.isna().any().sum()  # check nans any
df_test.isna().any().sum()  # check nans any
df_train.to_pickle(f"{DATA_ROOT}/train/model/data.pkl")
df_test.to_pickle(f"{DATA_ROOT}/test/model/data.pkl")
