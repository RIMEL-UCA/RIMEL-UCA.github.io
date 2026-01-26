"""
Génère une visualisation détaillée par projet montrant les scores de chaque juge
et les différences, similaire à un tableau Excel avec mise en forme conditionnelle.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'results', '4_inter_rater_analysis')
OUTPUT_DIR = INPUT_DIR

# Charger les données détaillées
detailed_df = pd.read_csv(os.path.join(INPUT_DIR, 'comparaison_detaillee.csv'))

def create_detailed_visualization():
    """Crée une visualisation détaillée similaire à Excel"""
    
    # Préparer les données pour la visualisation
    projets = detailed_df['Projet'].tolist()
    juges = ['antoine', 'baptiste', 'roxane', 'theo']
    
    # Créer des sous-figures pour tests et documentation
    fig = plt.figure(figsize=(20, 12))
    
    # === TESTS ===
    ax1 = plt.subplot(2, 1, 1)
    
    # Créer une matrice pour les scores des tests
    test_matrix = []
    for _, row in detailed_df.iterrows():
        test_scores = [row[f'{juge}_Tests'] for juge in juges]
        test_matrix.append(test_scores + [row['Moyenne_Tests'], row['Ecart_type_Tests']])
    
    test_matrix = np.array(test_matrix)
    
    # Créer le heatmap
    im1 = ax1.imshow(test_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Configurer les axes
    ax1.set_xticks(np.arange(len(projets)))
    ax1.set_yticks(np.arange(len(juges) + 2))
    ax1.set_xticklabels([p.replace('_', '\n') for p in projets], rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(juges + ['Moyenne', 'Écart-type'])
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(projets)):
        for j in range(len(juges) + 2):
            text = ax1.text(i, j, f'{test_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8, weight='bold')
    
    ax1.set_title('Scores des Tests par Juge et par Projet', fontsize=14, weight='bold', pad=20)
    
    # Ajouter une colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Score', rotation=270, labelpad=15)
    
    # === DOCUMENTATION ===
    ax2 = plt.subplot(2, 1, 2)
    
    # Créer une matrice pour les scores de documentation
    doc_matrix = []
    for _, row in detailed_df.iterrows():
        doc_scores = [row[f'{juge}_Doc'] for juge in juges]
        doc_matrix.append(doc_scores + [row['Moyenne_Doc'], row['Ecart_type_Doc']])
    
    doc_matrix = np.array(doc_matrix)
    
    # Créer le heatmap
    im2 = ax2.imshow(doc_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Configurer les axes
    ax2.set_xticks(np.arange(len(projets)))
    ax2.set_yticks(np.arange(len(juges) + 2))
    ax2.set_xticklabels([p.replace('_', '\n') for p in projets], rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(juges + ['Moyenne', 'Écart-type'])
    
    # Ajouter les valeurs dans les cellules
    for i in range(len(projets)):
        for j in range(len(juges) + 2):
            text = ax2.text(i, j, f'{doc_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8, weight='bold')
    
    ax2.set_title('Scores de Documentation par Juge et par Projet', fontsize=14, weight='bold', pad=20)
    
    # Ajouter une colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tableau_scores_detailles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisation détaillée créée : tableau_scores_detailles.png")

def create_difference_table():
    """Crée un tableau des différences maximales entre juges pour chaque projet"""
    
    projets = detailed_df['Projet'].tolist()
    juges = ['antoine', 'baptiste', 'roxane', 'theo']
    
    results = []
    
    for _, row in detailed_df.iterrows():
        projet = row['Projet']
        
        # Tests
        test_scores = [row[f'{juge}_Tests'] for juge in juges]
        min_test = min(test_scores)
        max_test = max(test_scores)
        range_test = max_test - min_test
        
        # Identifier qui a donné le min et le max
        min_juge_test = juges[test_scores.index(min_test)]
        max_juge_test = juges[test_scores.index(max_test)]
        
        # Documentation
        doc_scores = [row[f'{juge}_Doc'] for juge in juges]
        min_doc = min(doc_scores)
        max_doc = max(doc_scores)
        range_doc = max_doc - min_doc
        
        # Identifier qui a donné le min et le max
        min_juge_doc = juges[doc_scores.index(min_doc)]
        max_juge_doc = juges[doc_scores.index(max_doc)]
        
        results.append({
            'Projet': projet,
            'Min_Tests': min_test,
            'Juge_Min_Tests': min_juge_test,
            'Max_Tests': max_test,
            'Juge_Max_Tests': max_juge_test,
            'Ecart_Tests': range_test,
            'Min_Doc': min_doc,
            'Juge_Min_Doc': min_juge_doc,
            'Max_Doc': max_doc,
            'Juge_Max_Doc': max_juge_doc,
            'Ecart_Doc': range_doc
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'differences_min_max_par_projet.csv'), index=False)
    
    print("\nTableau des différences min-max par projet :")
    print(results_df.to_string(index=False))
    
    return results_df

def create_disagreement_ranking():
    """Crée un classement des projets par niveau de désaccord"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Tests
    sorted_tests = detailed_df.sort_values('Ecart_type_Tests', ascending=False)
    axes[0].barh(range(len(sorted_tests)), sorted_tests['Ecart_type_Tests'], color='coral')
    axes[0].set_yticks(range(len(sorted_tests)))
    axes[0].set_yticklabels(sorted_tests['Projet'], fontsize=10)
    axes[0].set_xlabel('Écart-type des scores', fontsize=12)
    axes[0].set_title('Projets classés par niveau de désaccord - Tests', fontsize=13, weight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(sorted_tests['Ecart_type_Tests']):
        axes[0].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9)
    
    # Documentation
    sorted_doc = detailed_df.sort_values('Ecart_type_Doc', ascending=False)
    axes[1].barh(range(len(sorted_doc)), sorted_doc['Ecart_type_Doc'], color='skyblue')
    axes[1].set_yticks(range(len(sorted_doc)))
    axes[1].set_yticklabels(sorted_doc['Projet'], fontsize=10)
    axes[1].set_xlabel('Écart-type des scores', fontsize=12)
    axes[1].set_title('Projets classés par niveau de désaccord - Documentation', fontsize=13, weight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(sorted_doc['Ecart_type_Doc']):
        axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'classement_desaccords.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classement des désaccords créé : classement_desaccords.png")

def main():
    """Fonction principale"""
    print("Génération des visualisations détaillées...\n")
    
    create_detailed_visualization()
    create_difference_table()
    create_disagreement_ranking()
    
    print(f"\nToutes les visualisations détaillées ont été générées dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
