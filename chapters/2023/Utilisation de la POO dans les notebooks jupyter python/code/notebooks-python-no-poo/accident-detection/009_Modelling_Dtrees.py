import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dtreeviz.trees import *
from sklearn import tree
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
DATA_ROOT = f"../data"
df_train = pd.read_pickle(f"{DATA_ROOT}/train/model/data.pkl")
df_test = pd.read_pickle(f"{DATA_ROOT}/test/model/data.pkl")

df_train.head()
x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
x_train, x_cv, y_train, y_cv = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=False
)
x_train.shape, y_train.shape
x_cv.shape, y_cv.shape
x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
# demo
cm = confusion_matrix([1, 0, 1, 0, 1, 0, 0], [1, 1, 0, 1, 0, 1, 0])
cm
cm / cm.sum(axis=0)  # precision matrix -> dividing by predicted positives
(cm.T / cm.sum(axis=1)).T  # recall matrix -> dividing by actual positives
plt.figure(figsize=(3, 2), dpi=100)
cmap = sns.light_palette("green")
labels = [1, 2]
sns.heatmap(
    cm / cm.sum(axis=0),
    annot=True,
    cmap=cmap,
    fmt=".3f",
    xticklabels=labels,
    yticklabels=labels,
)
plt.xlabel("Predicted Class")
plt.ylabel("Original Class")
plt.show()
np.seterr(divide="ignore", invalid="ignore")


def get_pr_matrix(y_true, y_pred):
    """
    Get precision recall matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    # avoid nans in matrix, replace with 0
    pr_matrix = cm / cm.sum(axis=0)
    pr_matrix = np.nan_to_num(pr_matrix)
    re_matrix = (cm.T / cm.sum(axis=1)).T
    re_matrix = np.nan_to_num(re_matrix)

    return pr_matrix, re_matrix


def plot_matrix_heatmap(mat, labels=[1, 2, 3, 4], title="None"):
    plt.figure(figsize=(4, 1), dpi=150)
    plt.title(title)
    cmap = sns.light_palette("green")
    sns.heatmap(
        mat, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.show()


def plot_pr_matrix_heatmaps(y_true, y_pred):
    p, r = get_pr_matrix(y_true, y_pred)

    plot_matrix_heatmap(p, title="Precision Matrix")
    plot_matrix_heatmap(r, title="Recall Matrix")
plot_pr_matrix_heatmaps(y_train, y_train)
# demo
log_loss(
    [0, 1, 1, 0],  # true labels
    [[0.1, 0.9], [0.8, 0.2], [0.1, 0.9], [0.3, 0.7]],  # probability scores
)
k_train = x_train.shape[0]
y_train_pred_random = np.random.randint(low=1, high=5, size=k_train)
plot_pr_matrix_heatmaps(y_train, y_train_pred_random)  # on train set
random_log_loss_train = log_loss(
    y_train,
    [np.random.rand(1, 4)[0] for i in range(y_train.shape[0])],
    labels=[1, 2, 3, 4],
)  # on train set


print(
    f"log loss on predicting probabilities randomly on test set {random_log_loss_train}"
)
"""
for "balanced" 
w = n_samples / (n_classes * np.bincount(y))
"""

classes = [1, 2, 3, 4]
w = compute_class_weight("balanced", classes=classes, y=y_train,)
class_weights = {i: j for i, j in zip(classes, w)}

# get value counts
value_counts = df_train["Severity"].value_counts().to_dict()
print("value_counts of labels in train set:")
print(dict(sorted(value_counts.items(), key=lambda x: x[0])))

print("\nweights of labels:")
print(dict(sorted(class_weights.items(), key=lambda x: x[0])))
max_depth = [5, 10, 12, 15, 18, 20]

errors_log = []
for d in max_depth:
    print(f"{'-'*30} max_depth={d} {'-'*30}")

    clf = tree.DecisionTreeClassifier(max_depth=d, class_weight=class_weights)
    clf = clf.fit(x_train, y_train)

    # get log los train
    ll_train = log_loss(y_train, clf.predict_proba(x_train), labels=[1, 2, 3, 4])
    # get pr matrix train
    p_train, r_train = get_pr_matrix(y_train, clf.predict(x_train))

    # get log los cv set
    ll_test = log_loss(y_cv, clf.predict_proba(x_cv), labels=[1, 2, 3, 4])
    # get pr matrix cv set
    p_test, r_test = get_pr_matrix(y_cv, clf.predict(x_cv))

    # append logs to dictionary

    obj = {
        "max_depth": d,
        "log_loss_train": ll_train,
        "log_loss_cv": ll_test,
        "p_train": p_train,
        "r_train": r_train,
        "p_test": p_test,
        "r_test": r_test,
    }
    errors_log.append(obj)

    print(f"log loss train {obj['log_loss_train']:.4f}")
    print(f"log loss cv {obj['log_loss_cv']:.4f}")
l = []
for i in errors_log:
    l.append([i["max_depth"], i["log_loss_train"], i["log_loss_cv"]])
l = np.array(l)
l
plt.figure(figsize=[5, 3], dpi=100)
plt.title("log_loss (Y) with max_depth (X)")
plt.plot(l[:, 0], l[:, 1])
plt.plot(l[:, 0], l[:, 2])

plt.ylabel("log_loss")
plt.xlabel("max_depth")
plt.grid()
plt.legend(["train", "cv"])
plt.show()
p_train = errors_log[2]["p_train"]
r_train = errors_log[2]["r_train"]
p_cv = errors_log[2]["p_test"]  # this key should be p_cv please ignore this
r_cv = errors_log[2]["r_test"]  # this key should be r_cv please ignore this
plot_matrix_heatmap(p_train, title="Precision Matrix train")
plot_matrix_heatmap(r_train, title="Recall Matrix train")
plot_matrix_heatmap(p_cv, title="Precision Matrix CV")
plot_matrix_heatmap(r_cv, title="Recall Matrix CV")
x_train_full = pd.concat([x_train, x_cv], axis=0, ignore_index=True)
x_train.shape[0] + x_cv.shape[0] == x_train_full.shape[0]
y_train_full = pd.concat([y_train, y_cv], axis=0, ignore_index=True)
y_train.shape[0] + y_cv.shape[0] == y_train_full.shape[0]
# train on complete train set
clf = tree.DecisionTreeClassifier(max_depth=12, class_weight=class_weights)
clf = clf.fit(x_train_full, y_train_full)
# save model
pd.to_pickle(clf, f"{DATA_ROOT}/dtree-12.pkl")
y_pred_test = clf.predict(x_test)
ll_test = log_loss(y_test, clf.predict_proba(x_test), labels=[1, 2, 3, 4])
print(f"Test log loss {ll_test}")
plot_pr_matrix_heatmaps(y_test, y_pred_test)
y_test.value_counts()
y_train_full.value_counts()
fn = x_train.columns
cn = [str(i) for i in [1, 2, 3, 4]]
for ix in np.random.randint(low=0, high=x_train.shape[0], size=5):
    print("Predicted class ", clf.predict([x_train.iloc[ix]]))
    print("Actual class ", [y_train[ix]], "\n")

    print("Path taken:\n")
    print(
        explain_prediction_path(
            clf,
            df_train.iloc[ix, :-1],
            feature_names=fn,
            class_names=cn,
            explanation_type="plain_english",
        ),
    )
    print("-" * 30)
df_fimp = pd.DataFrame([fn, clf.feature_importances_]).T.rename(
    columns={0: "feature", 1: "importance"}
)
df_fimp = df_fimp.sort_values(by=["importance"], ascending=False,).reset_index(
    drop=True
)
# top 5 features
df_fimp.head()
# bottom 5 features
df_fimp.tail()
plt.figure(figsize=[8, 14], dpi=150)
sns.barplot(y="feature", x="importance", data=df_fimp.head(50))
plt.grid()
plt.show()
