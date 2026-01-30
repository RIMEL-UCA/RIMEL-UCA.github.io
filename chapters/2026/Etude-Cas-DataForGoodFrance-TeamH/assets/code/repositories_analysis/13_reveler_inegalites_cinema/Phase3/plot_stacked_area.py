import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "results" / "processing_evolution.csv"
OUTPUT_PATH = BASE_DIR / "results" / "processing_evolution_stacked.png"

# Load data
df = pd.read_csv(CSV_PATH)

# Ensure correct ordering
df = df.sort_values("month")

months = df["month"]

categories = [
    "notebooks",
    "data_processing",
    "db_migrations",
    "db_seeds",
    "ml_integration"
]

values = [df[c] for c in categories]

# Plot
plt.figure(figsize=(10, 5))
plt.stackplot(
    months,
    values,
    labels=[
        "Notebooks (exploration)",
        "Traitements de données",
        "Migrations de schéma",
        "Seeds de base de données",
        "Intégration ML"
    ]
)

plt.legend(loc="upper left")
plt.xlabel("Mois")
plt.ylabel("Nombre de modifications")
plt.title("Évolution des traitements de données au fil du temps")
plt.tight_layout()

# Save figure
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print(f"Figure written to {OUTPUT_PATH}")
