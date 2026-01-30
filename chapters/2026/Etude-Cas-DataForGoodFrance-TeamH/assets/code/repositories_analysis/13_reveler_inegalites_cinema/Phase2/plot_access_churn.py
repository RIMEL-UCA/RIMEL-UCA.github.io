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

plt.figure(figsize=(12, 6))

plt.stackplot(
    df["month"],
    [df[col] for col in series],
    labels=[labels[col] for col in series],
    alpha=0.85
)

plt.title(
    "Répartition de l’effort de maintenance de l’accès aux données",
    fontsize=12
)
plt.xlabel("Mois")
plt.ylabel("Nombre de commits")
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / "access_churn_stacked_area.png", dpi=200)
plt.show()
