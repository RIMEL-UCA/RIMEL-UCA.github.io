import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
DATA_ROOT = "../data/"
df = pd.read_csv(f"{DATA_ROOT}/raw/accidents.csv")
df.head()
df.columns
nan_cols_series = df.isna().any()
nan_cols_series
nan_cols_series[nan_cols_series]
nan_cols = nan_cols_series[nan_cols_series].index.values.tolist()
# get nan percentage

percent_missing = df[nan_cols].isnull().sum() * 100 / len(df)
percent_missing.sort_values(ascending=False)
df["Number"] = df["Number"].fillna(-1)
df["TMC"] = df["TMC"].fillna(0.0)
plt.figure(figsize=[7, 6])
plt.title("Box plot of Wind_Speed(mph)")
plt.boxplot(
    df["Wind_Speed(mph)"].dropna().values, showmeans=True,
)
plt.show()
df["Wind_Speed(mph)"].median()
df["Wind_Speed(mph)"] = df["Wind_Speed(mph)"].fillna(df["Wind_Speed(mph)"].median())
plt.figure(figsize=[10, 7])
plt.title("Value counts of Weather_Condition")
weather_cond_counts = (
    df["Weather_Condition"].value_counts().head(15).reset_index().values
)
plt.barh(
    weather_cond_counts[:, 0], weather_cond_counts[:, 1],
)
plt.show()
weather_cond_counts[:10]
import random

list_of_candidates = weather_cond_counts[:10, 0]
number_of_items_to_pick = df["Weather_Condition"].isna().sum()
probability_distribution = (
    weather_cond_counts[:10, 1] / weather_cond_counts[:10, 1].sum()
)
draw = random.choices(
    list_of_candidates, k=number_of_items_to_pick, weights=probability_distribution
)
# fill nan values with random sample with weighted probabilities
df.loc[df["Weather_Condition"].isnull(), "Weather_Condition"] = draw
df["Visibility(mi)"].fillna(df["Visibility(mi)"].mean(), inplace=True)
df["Humidity(%)"].fillna(df["Humidity(%)"].mean(), inplace=True)
df["Temperature(F)"].fillna(df["Temperature(F)"].mean(), inplace=True)
df["Visibility(mi)"].fillna(df["Visibility(mi)"].mean(), inplace=True)
df["Pressure(in)"].fillna(df["Pressure(in)"].mean(), inplace=True)
df["Wind_Direction"] = df["Wind_Direction"].fillna("none")
# number of days' gap between Start_Time and Weather_Timestamp
(
    pd.to_datetime(df["Start_Time"]) - pd.to_datetime(df["Weather_Timestamp"])
).dt.days.describe().astype(int)
# number of hours' gap between Start_Time and Weather_Timestamp
t = (
    pd.to_datetime(df["Start_Time"]) - pd.to_datetime(df["Weather_Timestamp"])
).dt.total_seconds()
(t / 60).describe()
df["Weather_Timestamp"] = df["Weather_Timestamp"].fillna(df["Start_Time"])
df["Airport_Code"] = df["Airport_Code"].fillna(df["Airport_Code"].mode().values[0])
df["Timezone"] = df["Timezone"].fillna(df["Timezone"].mode().values[0])
df["Nautical_Twilight"] = df["Nautical_Twilight"].fillna(
    df["Nautical_Twilight"].mode().values[0]
)
df["Sunrise_Sunset"] = df["Sunrise_Sunset"].fillna(
    df["Sunrise_Sunset"].mode().values[0]
)
df["Civil_Twilight"] = df["Civil_Twilight"].fillna(
    df["Civil_Twilight"].mode().values[0]
)
df["Astronomical_Twilight"] = df["Astronomical_Twilight"].fillna(
    df["Astronomical_Twilight"].mode().values[0]
)
df["City"] = df["City"].fillna(df["City"].mode().values[0])
df["Description"] = df["Description"].fillna("empty")

df["Zipcode"] = df["Zipcode"].fillna(df["Zipcode"].mode().values[0])
recheck_nans_series = df.isna().any()
recheck_nans_series[recheck_nans_series]
rm_cols = recheck_nans_series[recheck_nans_series].index.tolist()
rm_cols
dir_path = f"{DATA_ROOT}/eda/"
os.makedirs(dir_path, exist_ok=True)
cols = df.columns.tolist()
for i in rm_cols:
    cols.remove(i)
file_path = f"{dir_path}/clean-data.csv"
df.to_csv(file_path, index=False, columns=cols)
