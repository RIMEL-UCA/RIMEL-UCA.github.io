from sklearn import tree
import pandas as pd

data = pd.read_csv("./accidents.zip")



X = data[['City', 'State']]
y = data["Severity"]

X = X.head(50000)
y = y.head(50000)

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


features_to_encode = ["City", "State"]

for feature in features_to_encode:
    X = encode_and_bind(X, feature)

print(X)


from sklearn import tree
from sklearn.model_selection import train_test_split

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)
