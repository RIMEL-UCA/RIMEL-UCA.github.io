import pandas as pd

print("📊 Étape 2 : Création des fichiers par question de recherche")
print("=" * 80)

# Chargement des données de l'étape 1
df_etape1 = pd.read_csv('../results/2_statistics/statistiques_par_projet.csv')

# Chargement du mapping
df_mapping = pd.read_csv('../results/3_analysis/mapping_projets.csv')

output_dir = '../results/3_analysis/'

# Fusion des données
df_merged = pd.merge(df_mapping, df_etape1, on='Projet', how='inner')

print(f"✅ {len(df_merged)} projets avec mapping chargés")
print(f"⚠️  Projets exclus (pas de mapping): {set(df_etape1['Projet']) - set(df_mapping['Projet'])}")

# ====================================================================
# QUESTION 1 : Impact de la PÉRIODE (Avant/Pendant/Après GenAI)
# ====================================================================
print("\n📝 Question 1 : Impact de la période temporelle")

q1_data = df_merged[['Période', 'Projet', 'Note moyenne test', 'ICC test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'ICC documentation', 'Interprétation ICC documentation']].copy()

# Tri par période
period_order = {'Avant GenAI': 0, 'Pendant GenAI': 1, 'Après GenAI': 2}
q1_data['_order'] = q1_data['Période'].map(period_order)
q1_data = q1_data.sort_values('_order').drop('_order', axis=1)

q1_data.to_csv(output_dir+'question_1/impact_periode.csv', index=False, encoding='utf-8')
print(f"   ✅ impact_periode.csv créé ({len(q1_data)} projets)")
print(f"   - Avant GenAI: {len(q1_data[q1_data['Période'] == 'Avant GenAI'])} projets")
print(f"   - Pendant GenAI: {len(q1_data[q1_data['Période'] == 'Pendant GenAI'])} projets")
print(f"   - Après GenAI: {len(q1_data[q1_data['Période'] == 'Après GenAI'])} projets")

# ====================================================================
# QUESTION 2 : Impact du VOLUME DE CONTRIBUTEURS (Peu/Beaucoup)
# ====================================================================
print("\n📝 Question 2 : Impact du volume de contributeurs")

q2_data = df_merged[['Volume_contributeurs', 'Note moyenne test', 'ICC test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'ICC documentation', 'Interprétation ICC documentation']].copy()

# Tri : Peu avant Beaucoup
volume_order = {'Peu': 0, 'Beaucoup': 1}
q2_data['_order'] = q2_data['Volume_contributeurs'].map(volume_order)
q2_data = q2_data.sort_values('_order').drop('_order', axis=1)

q2_data.to_csv(output_dir+'question_2/impact_volume_contributeurs.csv', index=False, encoding='utf-8')
print(f"   ✅ impact_volume_contributeurs.csv créé ({len(q2_data)} projets)")
print(f"   - Peu de contributeurs: {len(q2_data[q2_data['Volume_contributeurs'] == 'Peu'])} projets")
print(f"   - Beaucoup de contributeurs: {len(q2_data[q2_data['Volume_contributeurs'] == 'Beaucoup'])} projets")

# ====================================================================
# QUESTION 3 : Impact du TYPE (AI-related / Non AI-related)
# ====================================================================
print("\n📝 Question 3 : Impact du type de projet (AI vs non-AI)")

q3_data = df_merged[['Type_AI', 'Projet', 'Note moyenne test', 'ICC test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'ICC documentation', 'Interprétation ICC documentation']].copy()

# Tri : AI-related avant Non AI-related
type_order = {'AI-related': 0, 'Non AI-related': 1}
q3_data['_order'] = q3_data['Type_AI'].map(type_order)
q3_data = q3_data.sort_values('_order').drop('_order', axis=1)

q3_data.to_csv(output_dir+'question_3/impact_type_ai.csv', index=False, encoding='utf-8')
print(f"   ✅ impact_type_ai.csv créé ({len(q3_data)} projets)")
print(f"   - AI-related: {len(q3_data[q3_data['Type_AI'] == 'AI-related'])} projets")
print(f"   - Non AI-related: {len(q3_data[q3_data['Type_AI'] == 'Non AI-related'])} projets")

# ====================================================================
# Statistiques descriptives par question
# ====================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration matplotlib pour de meilleurs rendus
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Création des dossiers de sortie
os.makedirs('../results/3_analysis/question_1', exist_ok=True)
os.makedirs('../results/3_analysis/question_2', exist_ok=True)
os.makedirs('../results/3_analysis/question_3', exist_ok=True)

# ====================================================================
# QUESTION 1 : Évolution temporelle (courbes)
# ====================================================================
print("📊 Génération graphique Question 1...")

df_q1 = pd.read_csv('../results/3_analysis/question_1/impact_periode.csv')

# Calculer les moyennes par période
stats_periode = df_q1.groupby('Période').agg({
    'Note moyenne test': 'mean',
    'Note moyenne documentation': 'mean',
    'ICC test': 'mean',
    'ICC documentation': 'mean'
}).reindex(['Avant GenAI', 'Pendant GenAI', 'Après GenAI'])

# Créer le graphique
fig, ax = plt.subplots(figsize=(12, 7))

periodes = ['Avant GenAI', 'Pendant GenAI', 'Après GenAI']
x = np.arange(len(periodes))

# Tracer les courbes
ax.plot(x, stats_periode['Note moyenne test'], marker='o', linewidth=2.5,
        markersize=10, label='Tests', color='#2E86AB')
ax.plot(x, stats_periode['Note moyenne documentation'], marker='s', linewidth=2.5,
        markersize=10, label='Documentation', color='#A23B72')

# Ajouter les valeurs sur les points
for i, periode in enumerate(periodes):
    ax.text(i, stats_periode.loc[periode, 'Note moyenne test'] + 2,
            f"{stats_periode.loc[periode, 'Note moyenne test']:.1f}",
            ha='center', fontsize=9, color='#2E86AB', fontweight='bold')
    ax.text(i, stats_periode.loc[periode, 'Note moyenne documentation'] - 3,
            f"{stats_periode.loc[periode, 'Note moyenne documentation']:.1f}",
            ha='center', fontsize=9, color='#A23B72', fontweight='bold')

# Configuration des axes
ax.set_xticks(x)
ax.set_xticklabels(periodes, fontsize=11)
ax.set_ylabel('Note moyenne (0-100)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_title('Évolution des notes selon la période GenAI',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# Ajouter les ICC moyens
icc_test_moy = stats_periode['ICC test'].mean()
icc_doc_moy = stats_periode['ICC documentation'].mean()
textstr = f'ICC moyen Tests: {icc_test_moy:.3f}\nICC moyen Documentation: {icc_doc_moy:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('../results/3_analysis/question_1/evolution_temporelle.png', bbox_inches='tight')
plt.close()
print("   ✅ question_1/evolution_temporelle.png créé")

# ====================================================================
# QUESTION 2 : Volume de contributeurs (matrice)
# ====================================================================
print("📊 Génération graphique Question 2...")

df_q2 = pd.read_csv('../results/3_analysis/question_2/impact_volume_contributeurs.csv')

# Calculer les moyennes
stats_volume = df_q2.groupby('Volume_contributeurs').agg({
    'Note moyenne test': 'mean',
    'Note moyenne documentation': 'mean',
    'ICC test': 'mean',
    'ICC documentation': 'mean'
}).reindex(['Peu', 'Beaucoup'])

# Créer la matrice pour le heatmap
matrix_data = stats_volume[['Note moyenne test', 'Note moyenne documentation']].values

fig, ax = plt.subplots(figsize=(8, 5))

# Créer le heatmap
im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Configuration des axes
ax.set_xticks([0, 1])
ax.set_xticklabels(['Tests', 'Documentation'], fontsize=12, fontweight='bold')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Peu de contributeurs', 'Beaucoup de contributeurs'], fontsize=12)

# Ajouter les valeurs dans les cellules
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{matrix_data[i, j]:.1f}',
                      ha="center", va="center", color="black",
                      fontsize=14, fontweight='bold')

# Ajouter la barre de couleur
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Note moyenne (0-100)', rotation=270, labelpad=20, fontsize=11)

# Titre
ax.set_title('Impact du volume de contributeurs',
             fontsize=14, fontweight='bold', pad=20)

# Ajouter les ICC moyens
icc_test_moy = stats_volume['ICC test'].mean()
icc_doc_moy = stats_volume['ICC documentation'].mean()
textstr = f'ICC moyen Tests: {icc_test_moy:.3f}\nICC moyen Documentation: {icc_doc_moy:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(1.45, 0.5, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('../results/3_analysis/question_2/matrice_volume_contributeurs.png', bbox_inches='tight')
plt.close()
print("   ✅ question_2/matrice_volume_contributeurs.png créé")

# ====================================================================
# QUESTION 3 : Type de projet AI (matrice)
# ====================================================================
print("📊 Génération graphique Question 3...")

df_q3 = pd.read_csv('../results/3_analysis/question_3/impact_type_ai.csv')

# Calculer les moyennes
stats_type = df_q3.groupby('Type_AI').agg({
    'Note moyenne test': 'mean',
    'Note moyenne documentation': 'mean',
    'ICC test': 'mean',
    'ICC documentation': 'mean'
}).reindex(['AI-related', 'Non AI-related'])

# Créer la matrice pour le heatmap
matrix_data = stats_type[['Note moyenne test', 'Note moyenne documentation']].values

fig, ax = plt.subplots(figsize=(8, 5))

# Créer le heatmap
im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Configuration des axes
ax.set_xticks([0, 1])
ax.set_xticklabels(['Tests', 'Documentation'], fontsize=12, fontweight='bold')
ax.set_yticks([0, 1])
ax.set_yticklabels(['AI-related', 'Non AI-related'], fontsize=12)

# Ajouter les valeurs dans les cellules
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{matrix_data[i, j]:.1f}',
                      ha="center", va="center", color="black",
                      fontsize=14, fontweight='bold')

# Ajouter la barre de couleur
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Note moyenne (0-100)', rotation=270, labelpad=20, fontsize=11)

# Titre
ax.set_title('Impact du type de projet (AI vs Non-AI)',
             fontsize=14, fontweight='bold', pad=20)

# Ajouter les ICC moyens
icc_test_moy = stats_type['ICC test'].mean()
icc_doc_moy = stats_type['ICC documentation'].mean()
textstr = f'ICC moyen Tests: {icc_test_moy:.3f}\nICC moyen Documentation: {icc_doc_moy:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(1.45, 0.5, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('../results/3_analysis/question_3/matrice_type_projet.png', bbox_inches='tight')
plt.close()
print("   ✅ question_3/matrice_type_projet.png créé")

print("\n✨ Toutes les visualisations ont été générées avec succès !")
print("   📁 Dossiers de sortie:")
print("      - results/3_analysis/question_1/")
print("      - results/3_analysis/question_2/")
print("      - results/3_analysis/question_3/")
