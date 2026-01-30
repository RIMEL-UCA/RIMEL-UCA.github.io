import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Chemin vers le fichier CSV généré
csv_path = Path("results/access_churn_shiftdataportal.csv")
df = pd.read_csv(csv_path)

# Conversion et tri par date
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")

# Labels adaptés pour le Shift Data Portal
labels = {
    "raw_data_files": "Données brutes",
    "data_preparation_scripts": "Scripts de préparation",
    "backend_api": "API Backend (GraphQL)",
    "frontend_client": "Frontend (React)",
    "infrastructure_deployment": "Infra / CI-CD",
    "documentation": "Documentation",
}

series = list(labels.keys())

# Création du graphique en aires empilées
plt.figure(figsize=(14, 7))
plt.stackplot(
    df["month"],
    [df[col] for col in series],
    labels=[labels[col] for col in series],
    alpha=0.85,
    colors=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#34495e']
)

plt.title(
    "Évolution de l'activité de développement - The Shift Data Portal",
    fontsize=14,
    fontweight='bold',
    pad=20
)
plt.xlabel("Mois", fontsize=11)
plt.ylabel("Nombre de commits", fontsize=11)
plt.legend(loc="upper left", fontsize=9, framealpha=0.9)
plt.grid(True, linestyle="--", alpha=0.3)

# Sauvegarde
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(output_dir / "shiftdataportal_churn_stacked_area.png", dpi=200)
print(f"[✓] Graphique sauvegardé dans {output_dir / 'shiftdataportal_churn_stacked_area.png'}")

plt.show()