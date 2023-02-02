import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
DATA_ROOT = "../data"
df = pd.read_csv(f"{DATA_ROOT}/eda/clean-data.csv")
df.head(2)
_ = df[["Distance(mi)", "Severity"]]

plt.figure(figsize=[10, 7])
plt.title("Box plot of Severity vs Distance(mi)")
sns.boxplot(x="Severity", y="Distance(mi)", data=_)
plt.grid()
plt.show()
df["Side"].value_counts().reset_index()
pd.options.mode.chained_assignment = None
ix = df[df["Side"] == " "].index

# filling with mode (as only one value is empty)
df.loc[ix, "Side"] = "R"
df_temp = df[["Side", "Severity"]]
df_temp["per_Side_severity_sum"] = 1
df_temp = df_temp.groupby(["Side", "Severity"], as_index=False).sum()
df_temp
plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs Side")
sns.barplot(
    y="per_Side_severity_sum", x="Side", data=df_temp, hue="Severity",
)
plt.grid()
plt.show()
topk = 10
df_temp = df[["Severity", "City"]]
topk_cities = df_temp["City"].value_counts().head(topk).reset_index()["index"].values
df_temp = df_temp[df_temp["City"].isin(topk_cities)]

df_temp["per_city_severity_sum"] = 1

df_temp = df_temp.groupby(["City", "Severity"], as_index=False).sum()
df_temp.head()
plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs City")
sns.barplot(
    y="per_city_severity_sum", x="City", data=df_temp, hue="Severity", order=topk_cities
)
plt.grid()
plt.xticks(rotation=70)
plt.show()
topk = 10
df_temp = df[["Severity", "State"]]
topk_states = df_temp["State"].value_counts().head(topk).reset_index()["index"].values
df_temp = df_temp[df_temp["State"].isin(topk_cities)]

df_temp["per_state_severity_sum"] = 1

df_temp = df_temp.groupby(["State", "Severity"], as_index=False).sum()
df_temp.head()
plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs State")
sns.barplot(
    y="per_state_severity_sum",
    x="State",
    data=df_temp,
    hue="Severity",
    order=topk_states,
)
plt.grid()
plt.xticks(rotation=70)
plt.show()
pd.options.mode.chained_assignment = None  # default='warn'
df_temp = df[["Temperature(F)", "Severity"]]
df_temp["Temperature(C)"] = (df_temp["Temperature(F)"] - 32) / 1.8
df_temp
plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs Temperature(C)")
sns.boxplot(
    y="Temperature(C)", x="Severity", data=df_temp,
)
plt.grid()
plt.show()
df_temp = df[["Humidity(%)", "Severity"]]

plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs Humidity(%)")
sns.boxplot(
    y="Humidity(%)", x="Severity", data=df_temp,
)
plt.grid()
plt.show()
df_temp = df[["Severity", "Wind_Speed(mph)"]]

plt.figure(figsize=[12, 7])
plt.title("Box plot of Severity vs Wind_Speed(mph)")
sns.boxplot(
    y="Wind_Speed(mph)", x="Severity", data=df_temp,
)
plt.grid()
plt.show()
def poi_charts(df, poi_var):
    # prepare data
    df_temp = df[[poi_var, "Severity"]]
    var = f"per_{poi_var}_severity_sum"
    df_temp[var] = 1
    df_temp = df_temp.groupby([poi_var, "Severity"], as_index=False).sum()
    display(df_temp)

    # plot
    plt.figure(figsize=[8, 5])

    plt.title(f"Box plot of Severity vs {i}")
    sns.barplot(
        y=var, x=poi_var, data=df_temp, hue="Severity", order=[True, False],
    )
    plt.grid()
    plt.show()
poi_list = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Calming",
    "Traffic_Signal",
    "Turning_Loop",
]
for i in poi_list:
    poi_charts(df, i)
    print("\n\n")
df_temp = df[["Severity", "Humidity(%)", "Temperature(F)"]]
df_temp["Temperature(C)"] = (df_temp["Temperature(F)"] - 32) / 1.8
sns.kdeplot(
    data=df_temp[df_temp["Severity"] == 1], x="Temperature(C)", y="Humidity(%)",
)
plt.show()
sns.kdeplot(
    data=df_temp[df_temp["Severity"] == 2], x="Temperature(C)", y="Humidity(%)",
)
plt.show()
sns.kdeplot(
    data=df_temp[df_temp["Severity"] == 3], x="Temperature(C)", y="Humidity(%)",
)
plt.show()
sns.kdeplot(
    data=df_temp[df_temp["Severity"] == 4], x="Temperature(C)", y="Humidity(%)",
)
plt.show()
