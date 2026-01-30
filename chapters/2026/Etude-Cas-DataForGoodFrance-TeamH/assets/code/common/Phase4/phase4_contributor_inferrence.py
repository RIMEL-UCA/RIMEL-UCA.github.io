import pandas as pd
import re
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from utils.path_classifier import PathClassifier
from utils.contributor_classifier import ContributorClassifier
from utils.csv_helper import CsvHelper

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_COMMITS = RESULTS_DIR / "commits_detailed.csv"
OUTPUT_INFERRED = RESULTS_DIR / "contributors_jobs_inferred.csv"

def normalize_name(name):
    """
    Nettoie le nom pour fusionner les quasi-duplicats.
    Ex: 'Florian_Yun' et 'Florian Yun' deviennent 'florianyun'
    """
    if not isinstance(name, str):
        return "unknown"
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def main():
    required_cols = ['author_name', 'author_email', 'file_path']

    
    df = CsvHelper.read_and_validate(INPUT_COMMITS, required_columns=required_cols)
    
    if df is None:
        return
    
    df['canonical_id'] = df['author_name'].apply(normalize_name)

    identity_map = df.sort_values('author_name', key=lambda x: x.str.len(), ascending=False) \
                     .groupby('canonical_id') \
                     .first()[['author_name', 'author_email']]

    pathClassifier = PathClassifier()

    df["file_path"] = df["file_path"].astype(str)
    df["category"] = df["file_path"].apply(pathClassifier.classify)

    activity = (
        df.groupby(["canonical_id", "category"])
          .size()
          .unstack(fill_value=0)
    )

    contributor_classifier = ContributorClassifier()
    inferred_jobs = activity.apply(contributor_classifier.infer, axis=1)

    final_df = pd.DataFrame({"job": inferred_jobs})

    final_df = final_df.reset_index()

    final_df = final_df.join(identity_map, on='canonical_id')

    final_df = final_df[~final_df['author_name'].str.contains('bot', case=False, na=False)]

    output_df = final_df[["author_name", "author_email", "job"]]
    
    output_df.to_csv(OUTPUT_INFERRED, index=False)

    print(f"Fichier généré : {OUTPUT_INFERRED}")
    print("(Format standardisé : author_name, author_email, job)")

if __name__ == "__main__":
    main()