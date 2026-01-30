import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("results/access_churn.csv")
df = pd.read_csv(csv_path)

df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")

labels = {
    "raw_data_files": "Données brutes versionnées",
    "scraping_code": "Scripts de scraping",
    "extraction_processing_code": "Extraction / enrichissement",
    "db_migrations": "Migrations de schéma",
    "db_seeds": "Seeds de base de données",
    "data_access_code": "Code d’accès aux données",
}

series = list(labels.keys())

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

for col in series:
    axes[0].plot(
        df["month"],
        df[col],
        marker="o",
        label=labels[col]
    )

axes[0].set_title(
    "Évolution mensuelle de l’effort de maintenance de l’accès aux données",
    fontsize=12
)
axes[0].set_ylabel("Nombre de commits")
axes[0].legend(loc="upper left")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].stackplot(
    df["month"],
    [df[col] for col in series],
    labels=[labels[col] for col in series],
    alpha=0.85
)

axes[1].set_title(
    "Répartition de l’effort de maintenance par type d’accès",
    fontsize=12
)
axes[1].set_ylabel("Nombre de commits")
axes[1].set_xlabel("Mois")
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.5)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / "access_churn_analysis.png", dpi=200)
plt.show()
