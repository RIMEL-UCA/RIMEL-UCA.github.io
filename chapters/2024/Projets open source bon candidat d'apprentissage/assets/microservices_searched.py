import os

# Chemin du fichier contenant la liste des repos GitHub
repos_github_file = "assets/results_data_set_mini.txt"

os.system(f"mkdir work")

# Fonction pour cloner un repo GitHub
def clone_repo(repo_name):
    os.system(f"git clone git@github.com:{repo_name}.git")

# Fonction pour supprimer un repo cloné
def delete_repo(repo_name):
    folder_name = repo_name.split("/")[1]
    os.system(f"rm -rfd {folder_name}")

with open(repos_github_file, "r") as repos_github:
    repos = repos_github.readlines()

os.chdir("work")

# Clonage des repos GitHub
for repo_name in repos:
    clone_repo(repo_name)

    # MON CODE
    print(repo_name)

    delete_repo(repo_name)