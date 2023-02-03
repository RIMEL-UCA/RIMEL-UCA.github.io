import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud
DATA_ROOT = "../data"
df = pd.read_csv(f"{DATA_ROOT}/eda/clean-data.csv")

df.head(2)
df.columns
# checking if all ids are unique

df["ID"].nunique() == df.shape[0]
# checking contribution of each source
df["Source"].value_counts()
print(f"""There are {df["TMC"].nunique()} unique TMCs""")
df["TMC"].value_counts()
_ = (
    df["Severity"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Severity", "Severity": "counts"})
)
display(_)
plt.figure(figsize=[10, 5])
plt.title("Box plot of Distance(mi)")
plt.boxplot(df["Distance(mi)"].values)
plt.show()
plt.figure(figsize=[10, 5])
plt.title("KDE plot of Distance(mi)")
sns.kdeplot(df["Distance(mi)"].values)
plt.show()
plt.figure(figsize=[10, 5])
plt.title("KDE plot of Distance(mi) < 2")
sns.kdeplot(df[df["Distance(mi)"] < 2]["Distance(mi)"].values)
plt.show()
print(f"""There are {df["City"].nunique()} unique cities in the dataset""")
_ = (
    df["City"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "City", "City": "counts"})
).head(10)

plt.figure(figsize=[8, 12])
plt.title(f"Bar plot of City occurence counts")
ax = sns.barplot(y="City", x="counts", data=_,)
plt.grid()
plt.show()
print(f"""There are {df["County"].nunique()} unique counties in the dataset""")
_ = (
    df["County"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "County", "County": "counts"})
).head(15)

plt.figure(figsize=[8, 12])
plt.title(f"Bar plot of County occurence counts")
ax = sns.barplot(y="County", x="counts", data=_,)
plt.grid()
plt.show()
print(f"""There are {df['State'].nunique()} unique states in the dataset""")
# top 10 states with accident reports
df["State"].value_counts().head(10)
print(
    f"""There is only {df["Country"].nunique()} unique country ({df["Country"].unique()[0]}) in the dataset"""
)
_ = (df["Temperature(F)"] - 32) / 1.8

plt.figure(figsize=[10, 5])
plt.title("KDE plot of of Temperature(C)")
sns.kdeplot(_)
plt.grid()
plt.show()
_ = df["Humidity(%)"].values

plt.figure(figsize=[10, 5])
plt.title("KDE plot of of Humidity(%)")
sns.kdeplot(_)
plt.grid()
plt.show()
_ = df["Pressure(in)"]

plt.figure(figsize=[10, 5])
plt.title("cumulative distribution plot of of Pressure(in)")
sns.kdeplot(_, cumulative=True)
plt.grid()
plt.show()
_ = df["Visibility(mi)"]

plt.figure(figsize=[10, 5])
plt.title("Cumulative distribution plot of of Visibility(mi)")
sns.kdeplot(_, cumulative=True)
plt.grid()
plt.show()
df["Wind_Direction"].value_counts()
_ = df["Wind_Speed(mph)"]

plt.figure(figsize=[10, 5])
plt.title("box plot of of Wind_Speed(mph)")
plt.boxplot(_)
plt.grid()
plt.show()
_ = df["Wind_Speed(mph)"][df["Wind_Speed(mph)"] < 100]

plt.figure(figsize=[10, 5])
plt.title("box plot of of Wind_Speed(mph) < 100")
plt.boxplot(_)
plt.grid()
plt.show()
"Weather_Condition"

print(
    f"""number of unique weather conditions in dataset :  {df["Weather_Condition"].nunique()}"""
)
df["Weather_Condition"].value_counts().head(10)
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
df[poi_list].apply(lambda x: x.value_counts()).T.fillna(0).astype(int)
tod = [
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
]

for var in tod:
    df_temp = (
        df[var]
        .value_counts()
        .reset_index()
        .rename(columns={"index": var, var: "counts"})
    )
    display(df_temp)
    plt.figure(figsize=[5, 4])
    plt.title(f"Bar plot of {var} value counts")
    ax = sns.barplot(x=var, y="counts", data=df_temp,)
    plt.grid()
    plt.show()
    print("\n\n")
print(f"""number of unique zip codes {df["Zipcode"].nunique()}""")
print(f"""number of unique airport codes {df["Airport_Code"].nunique()}""")
print(f"""number of unique timezones {df["Timezone"].nunique()}""")
df["Timezone"].unique()
df["Description"]
text = "|".join(np.hstack(df["Description"].values))

# Create and Generate a Word Cloud Image
wordcloud = WordCloud().generate(text)
# Display the generated image
plt.figure(figsize=[15, 8])
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()
