import pandas as pd
import numpy as np
from scipy.stats import chi2

CSV_PATH = "Comparatif-des-4-projets.csv"
PROJECT_COLUMN = "repo_name"
N_EVALUATORS = 4

CRITERIA = [
    "Documentation",
    "TraitementData",
    "AccessibiliteSourceData",
    "ArchitectureVariee"
]

df = pd.read_csv(CSV_PATH, sep=";")

print(df[["Documentation1", "Documentation2"]].dtypes)

def kendalls_w(matrix):
    n_items, n_raters = matrix.shape

    ranks = np.array([
        pd.Series(matrix[:, i]).rank(method="average").to_numpy()
        for i in range(n_raters)
    ]).T

    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = rank_sums.mean()

    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = (12 * S) / (n_raters**2 * (n_items**3 - n_items))

    chi_square = n_raters * (n_items - 1) * W
    df_chi = n_items - 1
    p_value = 1 - chi2.cdf(chi_square, df_chi)

    return W, chi_square, df_chi, p_value

results = []

for criterion in CRITERIA:
    cols = [f"{criterion}{i+1}" for i in range(N_EVALUATORS)]
    ratings = df[cols].to_numpy()

    W, chi2_val, df_chi, p = kendalls_w(ratings)

    results.append({
        "Criterion": criterion,
        "W": round(W, 2),
        "Chi2": round(chi2_val, 2),
        "df": df_chi,
        "p_value": round(p, 2)
    })

results_df = pd.DataFrame(results)

print("\n=== Kendall's W par critère ===")
print(results_df)

global_W = results_df["W"].mean()

print("\n=== Accord global ===")
print(f"W global : {global_W:.2f}")
