"""
Analyse des désaccords entre les juges lors de la notation des projets.
Ce script analyse les différences de notation entre les 4 juges et identifie
les patterns de désaccord/accord, ainsi que la sévérité de chaque juge.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'results', '1_notes_per_judge')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', '4_inter_rater_analysis')

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Noms des juges
JUGES = ['antoine', 'baptiste', 'roxane', 'theo']

# Critères de notation
CRITERES_TESTS = ['Présence tests (/15)', 'Niveaux UT/IT/E2E (/25)', 
                  'Qualité tests (/25)', 'Coverage (/25)', 'Entretien Git (/10)']
CRITERES_DOC = ['README Description (/25)', 'README Architecture (/20)', 
                'README Installation (/25)', 'README Contributeurs (/5)',
                'CHANGELOG (/10)', 'CONTRIBUTING (/10)', 'LICENSE (/5)']

def load_judge_notes():
    """Charge les notes de tous les juges"""
    notes = {}
    for juge in JUGES:
        filepath = os.path.join(INPUT_DIR, f'notes_{juge}.csv')
        notes[juge] = pd.read_csv(filepath)
    return notes

def calculate_total_scores(notes):
    """Calcule les scores totaux pour tests et documentation"""
    scores = {juge: df.copy() for juge, df in notes.items()}
    
    for juge, df in scores.items():
        df['Score_Tests'] = df[CRITERES_TESTS].sum(axis=1)
        df['Score_Doc'] = df[CRITERES_DOC].sum(axis=1)
    
    return scores

def analyze_judge_severity(scores):
    """Analyse la sévérité de chaque juge"""
    stats = []
    
    for juge, df in scores.items():
        stats.append({
            'Juge': juge,
            'Moyenne_Tests': df['Score_Tests'].mean(),
            'Mediane_Tests': df['Score_Tests'].median(),
            'Ecart_type_Tests': df['Score_Tests'].std(),
            'Moyenne_Doc': df['Score_Doc'].mean(),
            'Mediane_Doc': df['Score_Doc'].median(),
            'Ecart_type_Doc': df['Score_Doc'].std()
        })
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'statistiques_par_juge.csv'), index=False)
    
    return stats_df

def calculate_pairwise_differences(scores, metric='Score_Tests'):
    """Calcule les différences moyennes entre chaque paire de juges"""
    differences = []
    
    for juge1, juge2 in combinations(JUGES, 2):
        df1 = scores[juge1]
        df2 = scores[juge2]
        
        # Calculer les différences absolues pour chaque projet
        diff_abs = abs(df1[metric] - df2[metric])
        diff_signee = df1[metric] - df2[metric]
        
        differences.append({
            'Juge_1': juge1,
            'Juge_2': juge2,
            'Difference_Moyenne_Absolue': diff_abs.mean(),
            'Difference_Mediane_Absolue': diff_abs.median(),
            'Difference_Max': diff_abs.max(),
            'Difference_Moyenne_Signee': diff_signee.mean(),
            'Ecart_type': diff_abs.std()
        })
    
    return pd.DataFrame(differences)

def create_difference_matrix(scores, metric='Score_Tests'):
    """Crée une matrice des différences moyennes absolues entre juges"""
    matrix = pd.DataFrame(index=JUGES, columns=JUGES, dtype=float)
    
    for juge1 in JUGES:
        for juge2 in JUGES:
            if juge1 == juge2:
                matrix.loc[juge1, juge2] = 0.0
            else:
                df1 = scores[juge1]
                df2 = scores[juge2]
                diff_abs = abs(df1[metric] - df2[metric]).mean()
                matrix.loc[juge1, juge2] = diff_abs
    
    return matrix.astype(float)

def analyze_per_project_disagreement(scores):
    """Analyse les désaccords par projet"""
    projets = scores[JUGES[0]]['Dépôt'].tolist()
    results = []
    
    for i, projet in enumerate(projets):
        # Récupérer les scores de tous les juges pour ce projet
        scores_tests = [scores[juge].iloc[i]['Score_Tests'] for juge in JUGES]
        scores_doc = [scores[juge].iloc[i]['Score_Doc'] for juge in JUGES]
        
        results.append({
            'Projet': projet,
            'Ecart_type_Tests': np.std(scores_tests),
            'Ecart_type_Doc': np.std(scores_doc),
            'Range_Tests': max(scores_tests) - min(scores_tests),
            'Range_Doc': max(scores_doc) - min(scores_doc),
            'Coefficient_Variation_Tests': np.std(scores_tests) / (np.mean(scores_tests) + 0.01),  # +0.01 pour éviter division par 0
            'Coefficient_Variation_Doc': np.std(scores_doc) / (np.mean(scores_doc) + 0.01)
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'desaccords_par_projet.csv'), index=False)
    
    return results_df

def create_detailed_comparison(scores):
    """Crée un tableau détaillé de comparaison pour tous les projets et juges"""
    projets = scores[JUGES[0]]['Dépôt'].tolist()
    detailed = []
    
    for i, projet in enumerate(projets):
        row = {'Projet': projet}
        
        # Scores de chaque juge
        for juge in JUGES:
            row[f'{juge}_Tests'] = scores[juge].iloc[i]['Score_Tests']
            row[f'{juge}_Doc'] = scores[juge].iloc[i]['Score_Doc']
        
        # Statistiques pour tests
        scores_tests = [row[f'{juge}_Tests'] for juge in JUGES]
        row['Moyenne_Tests'] = np.mean(scores_tests)
        row['Mediane_Tests'] = np.median(scores_tests)
        row['Ecart_type_Tests'] = np.std(scores_tests)
        row['Min_Tests'] = min(scores_tests)
        row['Max_Tests'] = max(scores_tests)
        
        # Statistiques pour documentation
        scores_doc = [row[f'{juge}_Doc'] for juge in JUGES]
        row['Moyenne_Doc'] = np.mean(scores_doc)
        row['Mediane_Doc'] = np.median(scores_doc)
        row['Ecart_type_Doc'] = np.std(scores_doc)
        row['Min_Doc'] = min(scores_doc)
        row['Max_Doc'] = max(scores_doc)
        
        detailed.append(row)
    
    detailed_df = pd.DataFrame(detailed)
    detailed_df.to_csv(os.path.join(OUTPUT_DIR, 'comparaison_detaillee.csv'), index=False)
    
    return detailed_df

def plot_severity_comparison(stats_df):
    """Crée un graphique comparant la sévérité des juges"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tests
    axes[0].bar(stats_df['Juge'], stats_df['Moyenne_Tests'], alpha=0.7, label='Moyenne')
    axes[0].errorbar(stats_df['Juge'], stats_df['Moyenne_Tests'], 
                     yerr=stats_df['Ecart_type_Tests'], fmt='o', color='red', label='Écart-type')
    axes[0].set_xlabel('Juge')
    axes[0].set_ylabel('Score moyen')
    axes[0].set_title('Sévérité des juges - Tests')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Documentation
    axes[1].bar(stats_df['Juge'], stats_df['Moyenne_Doc'], alpha=0.7, label='Moyenne')
    axes[1].errorbar(stats_df['Juge'], stats_df['Moyenne_Doc'], 
                     yerr=stats_df['Ecart_type_Doc'], fmt='o', color='red', label='Écart-type')
    axes[1].set_xlabel('Juge')
    axes[1].set_ylabel('Score moyen')
    axes[1].set_title('Sévérité des juges - Documentation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'severite_juges.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_difference_heatmaps(scores):
    """Crée des heatmaps des différences entre juges"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tests
    matrix_tests = create_difference_matrix(scores, 'Score_Tests')
    sns.heatmap(matrix_tests, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                ax=axes[0], cbar_kws={'label': 'Différence moyenne absolue'})
    axes[0].set_title('Matrice de désaccord - Tests')
    
    # Documentation
    matrix_doc = create_difference_matrix(scores, 'Score_Doc')
    sns.heatmap(matrix_doc, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                ax=axes[1], cbar_kws={'label': 'Différence moyenne absolue'})
    axes[1].set_title('Matrice de désaccord - Documentation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matrices_desaccord.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Fonction principale"""
    print("Chargement des notes...")
    notes = load_judge_notes()
    
    print("Calcul des scores totaux...")
    scores = calculate_total_scores(notes)
    
    print("Analyse de la sévérité des juges...")
    stats_df = analyze_judge_severity(scores)
    print("\nStatistiques par juge:")
    print(stats_df.to_string(index=False))
    
    print("\nAnalyse des différences par paire de juges (Tests)...")
    diff_tests = calculate_pairwise_differences(scores, 'Score_Tests')
    diff_tests.to_csv(os.path.join(OUTPUT_DIR, 'differences_paires_tests.csv'), index=False)
    print("\nDifférences par paire - Tests:")
    print(diff_tests.to_string(index=False))
    
    print("\nAnalyse des différences par paire de juges (Documentation)...")
    diff_doc = calculate_pairwise_differences(scores, 'Score_Doc')
    diff_doc.to_csv(os.path.join(OUTPUT_DIR, 'differences_paires_doc.csv'), index=False)
    print("\nDifférences par paire - Documentation:")
    print(diff_doc.to_string(index=False))
    
    print("\nAnalyse des désaccords par projet...")
    project_disagreement = analyze_per_project_disagreement(scores)
    print("\nProjets avec le plus de désaccord (Tests):")
    print(project_disagreement.nlargest(5, 'Ecart_type_Tests')[['Projet', 'Ecart_type_Tests', 'Range_Tests']])
    
    print("\nCréation de la comparaison détaillée...")
    detailed_df = create_detailed_comparison(scores)
    
    print("\nGénération des visualisations...")
    plot_severity_comparison(stats_df)
    plot_difference_heatmaps(scores)
    
    print(f"\nAnalyse terminée, Résultats sauvegardés dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
