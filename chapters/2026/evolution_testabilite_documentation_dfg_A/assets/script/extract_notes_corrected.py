import csv
from collections import defaultdict

print("🔄 Extraction corrigée des notes depuis all.csv")
print("=" * 80)

# Lecture du fichier all.csv
data = defaultdict(lambda: defaultdict(dict))

with open('all.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        personne = row['Personne']
        repo = row['Repo']
        critere = row['Critère']
        note = row['Note'].strip() if row['Note'].strip() else '0'

        if personne and repo:
            data[personne][repo][critere] = note

# Liste des dépôts dans l'ordre
repos = [
    '13_ia_financement',
    '13_democratiser_sobriete',
    'offseason_missiontransition_categorisation',
    'shiftdataportal',
    '13_ecoskills',
    'batch4_diafoirus_fleming',
    'website2022',
    'batch5_phenix_happymeal',
    'batch11_cartovegetation',
    'bechdelai',
    'protectclimateactivists',
    'offseason_ogre',
    '14_PrixChangementClimatique'
]

# Critères pour TESTS (sans LICENSE)
test_criteria = [
    'Présence de tests /15',
    'Niveaux présents (UT/IT/E2E) /25',
    'Qualité des tests /25',
    'Coverage dispo + couverture /25',
    'Entretien dans le temps (Git) /10'
]

# Critères pour DOCUMENTATION (avec LICENSE)
doc_criteria = [
    'README description fonctionnelle du projet / 25',
    'README Explication architecture technique /20',
    'README instruction installation /25',
    'README nom et contact des contributeurs /5',
    'CHANGELOG à jour /10',
    'CONTRIBUTING complet /10',
    'LICENCE présente /5'
]

# Fonction pour obtenir la note avec valeur par défaut
def get_note(person_data, repo, critere, default='0'):
    return person_data.get(repo, {}).get(critere, default)

# Traitement pour chaque personne
for personne in ['VKT', 'Baptiste', 'Antoine', 'Roxx']:
    person_data = data[personne]

    print(f"\n📝 Traitement: {personne}")

    rows = []
    for repo in repos:
        # TESTS (total /100 maintenant, sans LICENSE)
        test_notes = []
        for crit in test_criteria:
            note = get_note(person_data, repo, crit)
            test_notes.append(float(note))

        total_tests = sum(test_notes)

        # DOCUMENTATION (total /100, avec LICENSE)
        doc_notes = []
        for crit in doc_criteria:
            note = get_note(person_data, repo, crit)
            doc_notes.append(float(note))

        total_doc = sum(doc_notes)

        total_global = total_tests + total_doc

        rows.append([
            repo,
            test_notes[0],  # Présence tests /15
            test_notes[1],  # Niveaux UT/IT/E2E /25
            test_notes[2],  # Qualité tests /25
            test_notes[3],  # Coverage /25
            test_notes[4],  # Entretien Git /10
            f"{total_tests:.1f}",  # Total Tests /100
            doc_notes[0],  # README Description /25
            doc_notes[1],  # README Architecture /20
            doc_notes[2],  # README Installation /25
            doc_notes[3],  # README Contributeurs /5
            doc_notes[4],  # CHANGELOG /10
            doc_notes[5],  # CONTRIBUTING /10
            doc_notes[6],  # LICENSE /5
            f"{total_doc:.1f}",  # Total Doc /100
            f"{total_global:.1f}"  # Total Global /200
        ])

    # Écriture du CSV
    filename = f'analyse_dataforgood_repos_{personne.lower()}_corrected.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Dépôt',
            'Présence tests (/15)', 'Niveaux UT/IT/E2E (/25)', 'Qualité tests (/25)',
            'Coverage (/25)', 'Entretien Git (/10)', 'Total Tests (/100)',
            'README Description (/25)', 'README Architecture (/20)', 'README Installation (/25)',
            'README Contributeurs (/5)', 'CHANGELOG (/10)', 'CONTRIBUTING (/10)',
            'LICENSE (/5)', 'Total Doc (/100)', 'Total Global (/200)'
        ])
        writer.writerows(rows)

    print(f"   ✅ {filename}")

print("\n" + "=" * 80)
print("✨ Extraction corrigée terminée!")
print("\n📊 Nouvelle structure:")
print("   - Tests: /100 (sans LICENSE)")
print("   - Documentation: /100 (avec LICENSE)")
print("   - Total: /200")
