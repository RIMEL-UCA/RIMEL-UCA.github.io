import pandas as pd
import pingouin as pg
import csv

def interpret_icc(icc):
    """Interpret the inter-rater agreement."""
    if icc < 0.5:
        return 'poor'
    elif icc < 0.75:
        return 'moderate'
    elif icc < 0.9:
        return 'good'
    elif icc <= 1:
        return 'excellent'
    else:
        return 'invalid'

def calculate_icc_for_criteria(notes_dict, repo_name):
    """
    Calcule l'ICC pour un dépôt donné sur l'ensemble des critères.
    notes_dict: {jury_name: [liste des notes pour chaque critère]}
    """
    # Conversion en DataFrame
    df = pd.DataFrame(notes_dict)

    # Restructuration pour pingouin
    df_long = df.reset_index().melt(id_vars="index", var_name="Rater", value_name="Score")
    df_long.rename(columns={"index": "Criteria"}, inplace=True)

    try:
        # Calcul de l'ICC type 3
        icc = pg.intraclass_corr(data=df_long, targets="Criteria", raters="Rater", ratings="Score")
        icc_type3 = icc.loc[icc['Type'] == 'ICC3', 'ICC'].values[0]
        return icc_type3
    except:
        return None

print("📊 Étape 1 : Calcul ICC par projet")
print("=" * 80)

# Chargement des données des 4 personnes (fichiers corrigés)
df_theo = pd.read_csv('../results/notes_theo.csv')
df_baptiste = pd.read_csv('../results/notes_baptiste.csv')
df_antoine = pd.read_csv('../results/notes_antoine.csv')
df_roxane = pd.read_csv('../results/notes_roxane.csv')

# Liste des dépôts
repos = df_theo['Dépôt'].tolist()

# Colonnes pour les tests
test_columns = [
    'Présence tests (/15)',
    'Niveaux UT/IT/E2E (/25)',
    'Qualité tests (/25)',
    'Coverage (/25)',
    'Entretien Git (/10)'
]

# Colonnes pour la documentation
doc_columns = [
    'README Description (/25)',
    'README Architecture (/20)',
    'README Installation (/25)',
    'README Contributeurs (/5)',
    'CHANGELOG (/10)',
    'CONTRIBUTING (/10)',
    'LICENSE (/5)'
]

results = []

for i, repo in enumerate(repos):
    print(f"\n🔍 Analyse: {repo}")

    # Récupération des lignes pour ce repo
    theo_row = df_theo.iloc[i]
    baptiste_row = df_baptiste.iloc[i]
    antoine_row = df_antoine.iloc[i]
    roxane_row = df_roxane.iloc[i]

    # === TESTS ===
    # Collecte des notes de test pour chaque jury
    notes_tests = {
        "Theo": [float(theo_row[col]) for col in test_columns],
        "Baptiste": [float(baptiste_row[col]) for col in test_columns],
        "Antoine": [float(antoine_row[col]) for col in test_columns],
        "Roxane": [float(roxane_row[col]) for col in test_columns]
    }

    # Somme de test par jury
    test_sommes = {jury: pd.Series(notes).sum() for jury, notes in notes_tests.items()}
    print(repo, " - Somme par jury:", test_sommes)
    # Moyenne globale de test
    test_moyenne_globale = pd.Series(test_sommes.values()).mean()
    print(repo, " - Moyenne globale:", test_moyenne_globale)

    # Calcul ICC pour les tests
    icc_tests = calculate_icc_for_criteria(notes_tests, repo)
    icc_tests_interpretation = interpret_icc(icc_tests) if icc_tests is not None else 'N/A'

    print(f"  Tests - Moyenne: {test_moyenne_globale:.2f}, ICC: {icc_tests:.3f} ({icc_tests_interpretation})")

    # === DOCUMENTATION ===
    # Collecte des notes de documentation pour chaque jury
    notes_doc = {
        "Theo": [float(theo_row[col]) for col in doc_columns],
        "Baptiste": [float(baptiste_row[col]) for col in doc_columns],
        "Antoine": [float(antoine_row[col]) for col in doc_columns],
        "Roxane": [float(roxane_row[col]) for col in doc_columns]
    }

    # Somme de documentation par jury
    doc_sommes = {jury: pd.Series(notes).sum() for jury, notes in notes_doc.items()}
    print(repo, " - Sommes par jury:", doc_sommes)
    # Moyenne globale de documentation
    doc_moyenne_globale = pd.Series(doc_sommes.values()).mean()
    print(repo, " - Moyenne globale:", doc_moyenne_globale)

    # Calcul ICC pour la documentation
    icc_doc = calculate_icc_for_criteria(notes_doc, repo)
    icc_doc_interpretation = interpret_icc(icc_doc) if icc_doc is not None else 'N/A'

    print(f"  Doc   - Moyenne: {doc_moyenne_globale:.2f}, ICC: {icc_doc:.3f} ({icc_doc_interpretation})")

    # Stockage des résultats
    results.append({
        'Projet': repo,
        'Note moyenne test': round(test_moyenne_globale, 2),
        'ICC test': round(icc_tests, 3) if icc_tests is not None else 'N/A',
        'Interprétation ICC test': icc_tests_interpretation,
        'Note moyenne documentation': round(doc_moyenne_globale, 2),
        'ICC documentation': round(icc_doc, 3) if icc_doc is not None else 'N/A',
        'Interprétation ICC documentation': icc_doc_interpretation
    })

# Sauvegarde du résultat
output_file = '../results/statistiques_par_projet.csv'
df_results = pd.DataFrame(results)
df_results.to_csv(output_file, index=False, encoding='utf-8')

print("\n" + "=" * 80)
print(f"✅ Fichier créé: {output_file}")
print("\n📊 Résumé:")
print(df_results.to_string(index=False))
