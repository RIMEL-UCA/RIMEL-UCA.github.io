from github import Github
from github.GithubException import GithubException
from collections import defaultdict

# Votre token d'authentification GitHub
token = 'Mon Token'

# Initialisez Github en utilisant votre token
g = Github(token)

# Fonction pour obtenir les langages utilisés dans un dépôt
def obtenir_langages(repo):
    try:
        languages = repo.get_languages()
        if languages:
            return list(languages.keys())
    except GithubException as e:
        print(f"Une erreur s'est produite lors de l'accès aux langages du dépôt : {e}")
    return None

# Fonction pour lire les URL des repos depuis le fichier texte
def lire_fichier_repos(fichier):
    with open(fichier, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Votre fichier texte contenant la liste des URL des repos GitHub
fichier_repos = 'results_data_set.txt'

# Lecture des URLs des repos depuis le fichier texte
repos_urls = lire_fichier_repos(fichier_repos)

langages_stats = defaultdict(int)

# Parcourir chaque URL de repo et obtenir les langages utilisés
for repo_url in repos_urls:
    try:
        repo = g.get_repo(repo_url)
        langages = obtenir_langages(repo)
        if langages:
            for langage in langages:
                langages_stats[langage] += 1
    except GithubException as e:
        print(f"Une erreur s'est produite lors de l'accès au dépôt {repo_url} : {e}")

# Calcul des statistiques de répartition des langages
total_repos = len(repos_urls)
statistiques = {langage: (count, f"{(count/total_repos)*100:.2f}%") for langage, count in langages_stats.items()}

# Écriture des statistiques dans un fichier texte
fichier_statistiques = 'statistiques_langages.txt'
with open(fichier_statistiques, 'w') as file:
    for langage, (count, percentage) in statistiques.items():
        file.write(f"{langage}: {count} repos - {percentage}\n")
