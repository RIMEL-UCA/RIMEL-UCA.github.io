import pandas as pd
import warnings
from plotnine.exceptions import PlotnineWarning
from plotnine import (
    ggplot, aes, geom_bar, labs, theme_minimal, theme,
    scale_fill_manual, element_text, scale_x_discrete
)
from pathlib import Path

SUMMARY_CSV = Path("1-qualite/outputs/summary.csv")
OUTPUT_DIR = Path("1-qualite/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_summary = pd.read_csv(SUMMARY_CSV)

warnings.filterwarnings("ignore", category=PlotnineWarning)

# Créer des catégories de score
def categorize_score(score):
    if score < 20:
        return "s<20"
    elif score < 40:
        return "20≤s<40"
    elif score < 60:
        return "40≤s<60"
    elif score < 80:
        return "60≤s<80"
    else:
        return "80≤s"

df_summary['score_category'] = df_summary['score'].apply(categorize_score)

# Compter le nombre de repos par catégorie
score_counts = df_summary['score_category'].value_counts().reset_index()
score_counts.columns = ['category', 'count']

# Définir l'ordre des catégories et les couleurs
category_order = ["s<20", "20≤s<40", "40≤s<60", "60≤s<80", "80≤s"]
colors = {
    "s<20": "#8B1538",        # Rouge bordeaux
    "20≤s<40": "#E74C3C",     # Rouge
    "40≤s<60": "#E67E22",     # Orange
    "60≤s<80": "#C4D600",     # Jaune-vert
    "80≤s": "#27AE60"         # Vert
}

score_counts['category'] = pd.Categorical(
    score_counts['category'], 
    categories=category_order, 
    ordered=True
)
score_counts = score_counts.sort_values('category')

# Créer le graphique
plot = (
    ggplot(score_counts, aes(x='category', y='count', fill='category')) +
    geom_bar(stat='identity', width=0.7) +
    scale_fill_manual(values=colors) +
    scale_x_discrete(name="Note qualité") +
    labs(
        title="Note SonarQube des dépôts de code de Data\nFor Good France de l'échantillon",
        y="Nombre de\nrepository"
    ) +
    theme_minimal() +
    theme(
        legend_position='none',
        plot_title=element_text(size=12, face='bold', ha='center'),
        axis_title_y=element_text(size=10),
        axis_title_x=element_text(size=10),
        figure_size=(8, 6)
    )
)

output_path = OUTPUT_DIR / "sonarqube_scores_distribution.png"
plot.save(output_path, dpi=300)
print(f"Graphique: {output_path}")