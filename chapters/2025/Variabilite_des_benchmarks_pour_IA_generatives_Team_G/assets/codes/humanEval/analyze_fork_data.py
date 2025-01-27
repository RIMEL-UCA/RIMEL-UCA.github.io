import re
from collections import defaultdict
import matplotlib.pyplot as plt

# Fonction pour analyser le fichier
def parse_forks(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Découper les forks en utilisant "Fork:" comme séparateur
    forks = data.split("Fork: ")[1:]
    fork_data = []
    print(f"{len(forks)} forks trouvés.")

    for fork in forks:
        lines = fork.splitlines()
        # Extraire les informations pertinentes
        try:
            name = lines[0].split(' - ')[0].strip()
            commits_match = re.search(r'Total commits: (\d+)', fork)
            added_lines_match = re.search(r'Lignes ajoutées: (\d+)', fork)
            deleted_lines_match = re.search(r'Lignes supprimées: (\d+)', fork)

            commits = int(commits_match.group(1)) if commits_match else 0
            added_lines = int(added_lines_match.group(1)) if added_lines_match else 0
            deleted_lines = int(deleted_lines_match.group(1)) if deleted_lines_match else 0

            modified_files = re.findall(r'Modifiés: ([^,]+)', fork)
            added_files = re.findall(r'Ajoutés: ([^,]+)', fork)
            deleted_files = re.findall(r'Supprimés: ([^,]+)', fork)
            messages = re.findall(r'Message: (.+)', fork)

            fork_data.append({
                'name': name,
                'commits': commits,
                'added_lines': added_lines,
                'deleted_lines': deleted_lines,
                'modified_files': modified_files,
                'added_files': added_files,
                'deleted_files': deleted_files,
                'messages': messages
            })
        except IndexError:
            # Ignorer les forks mal formatés
            continue

    return fork_data


def analyze_forks(fork_data):
    analysis = defaultdict(int)

    for fork in fork_data:
        # Forks sans modification
        if fork['commits'] == 0:
            analysis['no_changes'] += 1

        # Forks modifiant uniquement execution.py, README.md ou des fichiers spécifiques
        elif fork['commits'] > 0 and set(fork['modified_files']).issubset({
            'human_eval/execution.py',
            'README.md',
            'requirements.txt',
            'setup.py',
            'LICENSE',
            '__init__.py'}):
            analysis['only_execution_py'] += 1

        # Forks ajoutant des fonctionnalités (ajout de fichiers ou messages indiquant un ajout)
        elif fork['added_files'] or any('add' in msg.lower() or 'new' in msg.lower() or 'feature' in msg.lower() or 'feat' in msg.lower() for msg in fork['messages']):
            analysis['new_features'] += 1

    # Calculer les "autres"
    analysis['others'] = len(fork_data) - (analysis['no_changes'] + analysis['only_execution_py'] + analysis['new_features'])

    # Ajout d'autres métriques globales
    analysis['total_forks'] = len(fork_data)
    analysis['total_commits'] = sum(fork['commits'] for fork in fork_data)
    analysis['total_added_lines'] = sum(fork['added_lines'] for fork in fork_data)
    analysis['total_deleted_lines'] = sum(fork['deleted_lines'] for fork in fork_data)

    return analysis

def analyze_forks_commit(fork_data):
    analysis = defaultdict(int)
    analysis['0_commit'] = 0
    analysis['1_2_commits'] = 0
    analysis['3_5_commits'] = 0
    analysis['6_10_commits'] = 0
    analysis['11_20_commits'] = 0
    analysis['21_50_commits'] = 0


    for fork in fork_data:
        if fork['commits'] == 0:
            analysis['0_commit'] += 1

        elif fork['commits'] > 0 and fork['commits'] <= 2:
            analysis['1_2_commits'] += 1
        
        elif fork['commits'] > 2 and fork['commits'] <= 5:
            analysis['3_5_commits'] += 1

        elif fork['commits'] > 5 and fork['commits'] <= 10:
            analysis['6_10_commits'] += 1
        
        elif fork['commits'] > 10 and fork['commits'] <= 20:
            analysis['11_20_commits'] += 1

        elif fork['commits'] > 20 and fork['commits'] <= 50:
            analysis['21_50_commits'] += 1

        elif fork['commits'] > 50:
            analysis['50+_commits'] += 1
    
    analysis['total_forks'] = len(fork_data)
    analysis['total_commits'] = sum(fork['commits'] for fork in fork_data)
    return analysis

def analyze_forks_lines(fork_data):
    analysis = defaultdict(int)
    
    analysis['0_lines'] = 0
    analysis['1_5_lines'] = 0
    analysis['6_10_lines'] = 0
    analysis['11_20_lines'] = 0
    analysis['21_50_lines'] = 0
    analysis['50_100_lines'] = 0
    analysis['100_200_lines'] = 0
    analysis['200_300_lines'] = 0
    analysis['300_500_lines'] = 0
    analysis['500_750_lines'] = 0
    analysis['750_1000_lines'] = 0
    analysis['1000_1500_lines'] = 0
    analysis['1500_2000_lines'] = 0
    analysis['2000_3000_lines'] = 0
    analysis['3000_5000_lines'] = 0
    analysis['5000_7500_lines'] = 0
    analysis['7500+_lines'] = 0

    

    for fork in fork_data:
        if fork['added_lines'] == 0 :
            analysis['0_lines'] += 1
        elif fork['added_lines'] > 0 and fork['added_lines'] <= 5:
            analysis['1_5_lines'] += 1
        elif fork['added_lines'] > 5 and fork['added_lines'] <= 10:
            analysis['6_10_lines'] += 1
        elif fork['added_lines'] > 10 and fork['added_lines'] <= 20:
            analysis['11_20_lines'] += 1
        elif fork['added_lines'] > 20 and fork['added_lines'] <= 50:
            analysis['21_50_lines'] += 1
        elif fork['added_lines'] > 50 and fork['added_lines'] <= 100:
            analysis['50_100_lines'] += 1
        elif fork['added_lines'] > 100 and fork['added_lines'] <= 200:
            analysis['100_200_lines'] += 1
        elif fork['added_lines'] > 200 and fork['added_lines'] <= 300:
            analysis['200_300_lines'] += 1
        elif fork['added_lines'] > 300 and fork['added_lines'] <= 500:
            analysis['300_500_lines'] += 1
        elif fork['added_lines'] > 500 and fork['added_lines'] <= 750:
            analysis['500_750_lines'] += 1
        elif fork['added_lines'] > 750 and fork['added_lines'] <= 1000:
            analysis['750_1000_lines'] += 1
        elif fork['added_lines'] > 1000 and fork['added_lines'] <= 1500:
            analysis['1000_1500_lines'] += 1
        elif fork['added_lines'] > 1500 and fork['added_lines'] <= 2000:
            analysis['1500_2000_lines'] += 1
        elif fork['added_lines'] > 2000 and fork['added_lines'] <= 3000:
            analysis['2000_3000_lines'] += 1
        elif fork['added_lines'] > 3000 and fork['added_lines'] <= 5000:
            analysis['3000_5000_lines'] += 1
        elif fork['added_lines'] > 5000 and fork['added_lines'] <= 7500:
            analysis['5000_7500_lines'] += 1
        elif fork['added_lines'] > 7500:
            analysis['7500+_lines'] += 1

    
    return analysis

def plot_pie_chart(analysis):
    labels = ['No Changes', 'Only execution.py', 'New Features', 'Others']
    sizes = [
        analysis['no_changes'],
        analysis['only_execution_py'],
        analysis['new_features'],
        analysis['others']
    ]

    # Vérification des NaN et remplacement par 0 si nécessaire
    sizes = [0 if isinstance(size, float) and size != size else size for size in sizes]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    explode = (0.1, 0.1, 0.1, 0)  # Mettre en avant chaque catégorie sauf "Autres"

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution des Forks par Type de Modifications')
    plt.savefig('pie_chart_fork.png')  # Enregistre le graphique dans un fichier
    print("Le camembert a été enregistré sous le nom 'pie_chart.png'.")


def plot_pie_chart_commit(analysis):
    labels = ['0 Commit', '1-2 Commits', '3-5 Commits', '6-10 Commits', '11-20 Commits', '21-50 Commits', '50+ Commits']
    sizes = [
        analysis['0_commit'],
        analysis['1_2_commits'],
        analysis['3_5_commits'],
        analysis['6_10_commits'],
        analysis['11_20_commits'],
        analysis['21_50_commits'],
        analysis['50+_commits']
    ]

    # Vérification des NaN et remplacement par 0 si nécessaire
    sizes = [0 if isinstance(size, float) and size != size else size for size in sizes]  # Remplacer NaN par 0

    # Assurez-vous que toutes les tailles sont des entiers
    sizes = [int(size) for size in sizes]

    # Vérifier s'il y a des tailles égales à 0 et les traiter
    if sum(sizes) == 0:
        print("Erreur: Toutes les catégories ont une taille de 0, ce qui empêche l'affichage du graphique.")
        return

    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # Mettre en avant chaque catégorie

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution des Forks par Nombre de Commits')
    plt.savefig('pie_chart_commit.png')  # Enregistre le graphique dans un fichier
    print("Le camembert a été enregistré sous le nom 'pie_chart_commit.png'.")


def plot_pie_chart_line(analysis):
    labels = ['0 Lines', '1-5 Lines', '6-10 Lines', '11-20 Lines', '21-50 Lines', '50-100 Lines', '100-200 Lines', '200-300 Lines', '300-500 Lines', '500-750 Lines', '750-1000 Lines', '1000-1500 Lines', '1500-2000 Lines', '2000-3000 Lines', '3000-5000 Lines', '5000-7500 Lines', '7500+ Lines']
    sizes = [
        analysis['0_lines'],
        analysis['1_5_lines'],
        analysis['6_10_lines'],
        analysis['11_20_lines'],
        analysis['21_50_lines'],
        analysis['50_100_lines'],
        analysis['100_200_lines'],
        analysis['200_300_lines'],
        analysis['300_500_lines'],
        analysis['500_750_lines'],
        analysis['750_1000_lines'],
        analysis['1000_1500_lines'],
        analysis['1500_2000_lines'],
        analysis['2000_3000_lines'],
        analysis['3000_5000_lines'],
        analysis['5000_7500_lines'],
        analysis['7500+_lines']
    ]

    # Vérification des NaN et remplacement par 0 si nécessaire
    sizes = [0 if isinstance(size, float) and size != size else size for size in sizes]  # Remplacer NaN par 0

    # Assurez-vous que toutes les tailles sont des entiers
    sizes = [int(size) for size in sizes]

    # Vérifier s'il y a des tailles égales à 0 et les traiter
    if sum(sizes) == 0:
        print("Erreur: Toutes les catégories ont une taille de 0, ce qui empêche l'affichage du graphique.")
        return

    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2', '#ffcc00', '#ff6600', '#ff0066', '#cc00cc', '#660066', '#330033', '#000000', '#999999', '#cccccc']
    explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)  # Mettre en avant chaque catégorie

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution des Forks par Nombre de Lignes Ajoutées')
    plt.savefig('pie_chart_line.png')  # Enregistre le graphique dans un fichier
    print("Le camembert a été enregistré sous le nom 'pie_chart_line.png'.")

def plot_bar_chart_lines(line_analysis):
    # Créer les barres
    categories = list(line_analysis.keys())
    values = list(line_analysis.values())

    # Créer le graphique en barres
    plt.figure(figsize=(10, 6))
    plt.barh(categories, values, color='skyblue')
    # Ajouter la valeur de chaque barre
    for i, value in enumerate(values):
        plt.text(value, i, str(value))
    plt.xlabel('Nombre de Forks')
    plt.ylabel('Plage de Lignes Ajoutées')
    plt.title('Répartition des Forks par Nombre de Lignes Ajoutées')
    
    # Enregistrer le graphique dans un fichier
    plt.tight_layout()
    plt.savefig('bar_chart_lines.png')
    print("Le graphique en barres a été enregistré sous le nom 'bar_chart_lines.png'.")


# Chemin vers le fichier
file_path = "forks_commit_details_with_stats.txt"

# Extraire les données et effectuer les analyses
fork_data = parse_forks(file_path)
analysis = analyze_forks(fork_data)
analysis_commit = analyze_forks_commit(fork_data)
analysis_lines = analyze_forks_lines(fork_data)

# Afficher les résultats
print("Analyse des forks :")
for key, value in analysis.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

print("\nAnalyse des forks par nombre de commits :")
for key, value in analysis_commit.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

print("\nAnalyse des forks par nombre de lignes ajoutées :")
for key, value in analysis_lines.items():
    print(f"{key.replace('_', ' ').capitalize()}: {value}")

# Afficher les camemberts
plot_pie_chart(analysis)
plot_pie_chart_commit(analysis_commit)
#plot_pie_chart_line(analysis_lines)
plot_bar_chart_lines(analysis_lines)
#Enlever la donnée sur les 0 lignes ajoutées
analysis_lines.pop('0_lines')
plot_bar_chart_lines(analysis_lines)
