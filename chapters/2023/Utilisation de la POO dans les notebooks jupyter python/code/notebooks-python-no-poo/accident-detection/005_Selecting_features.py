import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
DATA_ROOT = f"../data"
df = pd.read_csv(f"{DATA_ROOT}/eda/clean-data.csv")
df.columns
selected_features = [
    "ID",  # will remove this in the end before modelling
    "Source",
    "TMC",
    "Start_Time",  # keeping this to sort the dataframe before train test split
    # "End_Time",
    # "Start_Lat",
    # "Start_Lng",
    "Distance(mi)",
    "Description",
    # "Number",
    # "Street",
    "Side",
    "City",
    "County",
    "State",
    "Zipcode",
    # "Country",
    "Timezone",
    "Airport_Code",
    # "Weather_Timestamp",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Direction",
    "Wind_Speed(mph)",
    "Weather_Condition",
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
    "Sunrise_Sunset",
    "Civil_Twilight",
    "Nautical_Twilight",
    "Astronomical_Twilight",
    #
    "Severity",
]
os.makedirs(f"{DATA_ROOT}/fselect/")
df[selected_features].to_pickle(f"{DATA_ROOT}/fselect/accidents_raw.pkl")
