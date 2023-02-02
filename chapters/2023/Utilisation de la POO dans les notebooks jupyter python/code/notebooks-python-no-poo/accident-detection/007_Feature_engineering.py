import html
import os
import re
import string
import unicodedata
from datetime import datetime

import category_encoders as ce
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from pandarallel import pandarallel

# from gensim.corpora import Dictionary
# from gensim.models import TfidfModel
# from gensim.utils import simple_preprocess
# from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
DATA_ROOT = "../data"

# os.makedirs(f"{DATA_ROOT}/train/features")
df_train = pd.read_pickle(f"{DATA_ROOT}/train/raw/data.pkl")
df_test = pd.read_pickle(f"{DATA_ROOT}/test/raw/data.pkl")
df_train.head(2)
df_test.head(2)
df_train.columns
df_train["Wind_Direction"].unique()
def clean_wind_direction(df):
    df["Wind_Direction"] = df["Wind_Direction"].str.upper()
    df["Wind_Direction"] = df["Wind_Direction"].apply(
        lambda x: "variable" if x == "VAR" else x
    )

    return df


df_train = clean_wind_direction(df_train)
df_test = clean_wind_direction(df_test)
print(sorted(df_train["Wind_Direction"].unique()))
print(sorted(df_test["Wind_Direction"].unique()))
def rm_numbers(x):
    x = re.sub(r"[0-9]+", "", x)
    return x


def rm_html(x):
    x = html.unescape(x)
    x = BeautifulSoup(x).get_text()
    return x


def rm_url(x):
    x = re.sub("http*\S+", " ", x)
    return x


def rm_multiple_dots(x):
    x = re.sub(r"\.+", ". ", x)
    x = re.sub("\।+", ". ", x)
    return x


def rm_unicode(x):
    x = unicodedata.normalize("NFKD", x)
    return x


def rm_punctuation(x):
    x = re.sub("[%s]" % re.escape(string.punctuation.replace(".", "")), " ", x)
    return x


def rm_spaces(x):
    x = re.sub(" +", " ", x)
    return x


def rm_word(x, word):
    x = x.replace(word, "")
    return x


def clean_text(input_string):
    ss = input_string
    ss = rm_html(ss)
    ss = rm_url(ss)
    ss = rm_punctuation(ss)
    ss = rm_multiple_dots(ss)
    ss = rm_unicode(ss)
    ss = rm_spaces(ss)
    # ss = rm_numbers(ss)
    ss = rm_word(ss, "\n")
    ss = rm_word(ss, "\t")
    ss = ss.strip()

    return ss
pandarallel.initialize(verbose=True,)
df_train["Description"] = df_train["Description"].parallel_apply(
    lambda x: clean_text(x)
)
df_train["Description"].sample(5).values
pandarallel.initialize(verbose=True,)
df_test["Description"] = df_test["Description"].parallel_apply(lambda x: clean_text(x))
df_test["Description"].sample(5).values
import yake

kw_extractor = yake.KeywordExtractor()
text = ".".join(
    df_train["Description"].sample(n=100_000).tolist()
)  # extracting from a sample as compute intensive process
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.1
numOfKeywords = 1000
custom_kw_extractor = yake.KeywordExtractor(
    lan=language,
    n=max_ngram_size,
    dedupLim=deduplication_threshold,
    top=numOfKeywords,
    features=None,
    stopwords=stopwords.words("english"),
)
keywords = custom_kw_extractor.extract_keywords(text)
# The lower the score, the more relevant the keyword is.
keywords
pd.to_pickle(
    keywords, f"{DATA_ROOT}/train/keywords.pkl",
)
# using top 15 words
keywords_list = [i[0] for i in keywords[:15]]
keywords_list
fuzz.partial_ratio("hello world 2", "hello world")  # demo of partial_ratio
def get_kw_vec(x, kw_list):
    vec = [fuzz.partial_ratio(i.lower(), x.lower()) for i in kw_list]
    vec = np.array(vec)
    vec = np.where(vec > 60, 1, 0).tolist()
    return vec
pandarallel.initialize(verbose=True,)
df_train["kw_vec"] = df_train["Description"].parallel_apply(
    lambda x: get_kw_vec(x, keywords_list)
)
pandarallel.initialize(verbose=True,)
df_test["kw_vec"] = df_test["Description"].parallel_apply(
    lambda x: get_kw_vec(x, keywords_list)
)
df_train["kw_vec"].sample(5).head()
df_train["zip_02"] = df_train["Zipcode"].str[:2]
df_test["zip_02"] = df_test["Zipcode"].str[:2]
df_train["zip_25"] = df_train["Zipcode"].str[2:5]
df_test["zip_25"] = df_test["Zipcode"].str[2:5]
df_train["zip_len"] = df_train["Zipcode"].apply(len)
df_test["zip_len"] = df_test["Zipcode"].apply(len)
df_train["zip_is_compound"] = df_train["Zipcode"].apply(lambda x: "-" in x)
df_test["zip_is_compound"] = df_test["Zipcode"].apply(lambda x: "-" in x)
df_train.dtypes
# making list of cateogrical features that need encoding
# features that are as dtype string will need encoding

# categorical_feature -> encoding method
categorical_features = {
    "Source": "base_2",  # 3 unique values
    "Side": "base_2",  # 2 unique values
    "City": "base_4",  # 11895 unique values
    "County": "base_4",  # 1713 unique values
    "State": "base_2",  # 49 unique values
    "Timezone": "base_2",  # 4 unique values
    "Airport_Code": "base_4",  # 2001 unique values
    "Wind_Direction": "base_4",  # 24 unique values (after cleaning)
    "Weather_Condition": "base_4",  # 127 unique values
    "Sunrise_Sunset": "base_2",  # 2 unique values
    "Civil_Twilight": "base_2",  # 2 unique values
    "Nautical_Twilight": "base_2",  # 2 unique value
    "Astronomical_Twilight": "base_2",  # 2 unique value
    #
    # engineered features
    "zip_02": "ordinal",
    "zip_25": "ordinal",
}
def category_transformer(
    df_train: pd.DataFrame, df_test: pd.DataFrame, categorical_features: dict
):
    mapping_dict = dict()

    for i in categorical_features:
        method = categorical_features[i]

        # get values which will be encoded
        index_values = df_train[i].drop_duplicates().values.tolist()

        if "base" in method:
            print(f"""encoding {i} with {method}""")
            baseN = int(method.split("_")[1])
            enc = ce.binary.BaseNEncoder(base=2)
            # get values for train
            t_train = enc.fit_transform(df_train[i])
            df_train[i] = t_train.values.tolist()

            # get values for test
            t_test = enc.transform(df_test[i])
            df_test[i] = t_test.values.tolist()

        if method == "ordinal":
            print(f"""encoding {i} as {method}""")
            enc = ce.ordinal.OrdinalEncoder()
            # get values for train
            t_train = enc.fit_transform(df_train[i])
            df_train[i] = t_train.values.tolist()

            # get values for test
            t_test = enc.transform(df_test[i])
            df_test[i] = t_test.values.tolist()

        # store params
        mapping_dict[i] = {
            "enc_model_params": enc.get_params(),
            "enc_model_values": index_values,
        }

    return df_train, df_test, mapping_dict


df_train, df_test, mapping_dict = category_transformer(
    df_train, df_test, categorical_features
)
# sample of how categories are encoded
# -1 denote representation of unknown value
# -2 denotes representation of missing value
mapping_dict["Source"]
# saving this as it will help in interpretation of results
pd.to_pickle(mapping_dict, f"{DATA_ROOT}/train/mapping_dict.pkl")
boolean_features = {
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
    # engineered features
    "zip_is_compound",
}
len(
    set(df_train["Wind_Direction"].unique()).intersection(
        set(df_test["Wind_Direction"].unique())
    )
)
final_feature_list = [
    "ID",  # will be removing this before preparing modelling data
    "Source",
    "TMC",
    # "Start_Time", -> removing this as data is now sorted and split
    "Distance(mi)",
    # "Description", -> extracted features & removing this
    "Side",
    "City",
    "County",
    "State",
    # "Zipcode", -> extracted features & removing this
    "Timezone",
    "Airport_Code",
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
    # engineered features
    "kw_vec",
    "zip_02",
    "zip_25",
    "zip_len",
    "zip_is_compound",
    # to predict
    "Severity",
]
os.makedirs(f"{DATA_ROOT}/train/featurized/", exist_ok=True)
os.makedirs(f"{DATA_ROOT}/test/featurized/", exist_ok=True)
df_train[final_feature_list].to_pickle(f"{DATA_ROOT}/train/featurized/data.pkl")
df_test[final_feature_list].to_pickle(f"{DATA_ROOT}/test/featurized/data.pkl")
