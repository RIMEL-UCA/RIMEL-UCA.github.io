import pandas as pd
import warnings
from plotnine.exceptions import PlotnineWarning
from plotnine import (
    ggplot, aes, geom_violin, labs, theme_minimal, theme,
    element_text, element_blank, scale_y_continuous, geom_boxplot
)
from pathlib import Path

SUMMARY_CSV = Path("1-qualite/outputs/summary.csv")
OUTPUT_DIR = Path("1-qualite/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_summary = pd.read_csv(SUMMARY_CSV)

warnings.filterwarnings("ignore", category=PlotnineWarning)

print(f"Analyse de {len(df_summary)} repos")

# Ajouter une colonne catégorielle pour l'axe X (nécessaire pour geom_violin)
df_summary['category'] = 'Echantillon de\ndépôts de code'

# Créer le graphique en violon
plot = (
    ggplot(df_summary, aes(x='category', y='score')) +
    geom_violin(fill='#7B83C0', alpha=0.6, color='#7B83C0') +
    geom_boxplot(width=0.1, fill='#2C3E50', alpha=0.8, color='black') +
    scale_y_continuous(limits=(0, 105), breaks=range(0, 101, 20)) +
    labs(
        title="Score de qualité des\ndépôt de code de\nl'échantillon",
        x="",
        y="Score de\nqualité"
    ) +
    theme_minimal() +
    theme(
        plot_title=element_text(size=12, face='bold', ha='center'),
        axis_title_y=element_text(size=10),
        axis_title_x=element_text(size=10),
        axis_text_x=element_text(size=9),
        panel_grid_major_x=element_blank(),
        figure_size=(6, 7)
    )
)

output_path = OUTPUT_DIR / "sonarqube_scores_violin.png"
plot.save(output_path, dpi=300)
print(f"Graphique: {output_path}")