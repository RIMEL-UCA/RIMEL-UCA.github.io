import json
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ANNOTATED_DATA = Path("3-activite-contributeurs/data/commits_other_for_ml.csv")
UNCLASSIFIED_JSON = Path("3-activite-contributeurs/data/commits_unclassified.json")
COMMITS_TYPES_CSV = Path("3-activite-contributeurs/data/commits_types.csv")
MODEL_OUT = Path("3-activite-contributeurs/models/commit_classifier.joblib")

df = pd.read_csv(ANNOTATED_DATA)
df = df[df["label"].notna() & (df["label"] != "")]

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("=== Evaluation du classifieur ===")
# Afficher un résumé concis plutôt que le rapport complet
report = classification_report(y_test, y_pred, output_dict=True)
acc = report.get("accuracy", 0.0)
macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
print(f"Précision (accuracy): {acc:.2f}")
print(f"Macro F1-score: {macro_f1:.2f}")

joblib.dump(pipeline, MODEL_OUT)
print(f"[OK] Modèle sauvegardé → {MODEL_OUT}")


with open(UNCLASSIFIED_JSON, encoding="utf-8") as f:
    unclassified = json.load(f)

predictions = []
for item in unclassified:
    label = pipeline.predict([item["message"]])[0]
    predictions.append({
        "repo": item["repo"],
        "label": label
    })

pred_df = pd.DataFrame(predictions)


commits_df = pd.read_csv(COMMITS_TYPES_CSV)

for repo, group in pred_df.groupby("repo"):
    counts = Counter(group["label"])
    for label, count in counts.items():
        commits_df.loc[commits_df["repo"] == repo, label] += count

commits_df["other"] = 0

label_cols = ["feat", "fix", "refactor", "ci", "chore"]
commits_df["other"] = (
    commits_df["total_commits"]
    - commits_df[label_cols].sum(axis=1)
)

commits_df.to_csv(COMMITS_TYPES_CSV, index=False)
print(f"[OK] commits_types.csv mis à jour avec les prédictions ML")
