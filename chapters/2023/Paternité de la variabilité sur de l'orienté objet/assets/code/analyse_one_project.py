import requests as req
import shutil
import sys
import subprocess
import os
import signal

# Verification de l'existence du dossier de l'analyser POO Varicity
if len(sys.argv) == 1:
    print("Il manque le chemin vers le dossier de varicity")
    exit(1)


def stop_docker():
    docker_stop = subprocess.Popen("docker stop $(docker ps -q  --filter ancestor=deathstar3/symfinder-cli:splc2022)",
                                   shell=True)
    docker_stop.wait()


# Pour gérer Ctrl+C pendant l'éxecution du script
def handler(signum, frame):
    stop_docker()
    exit(1)

def symfinder(item):
    print("Project: " + item["name"])
    commits = req.get("https://api.github.com/repos/" + item["full_name"] + "/commits").json()
    if len(commits) < 1:
        print("Not enough commits for project " + item["name"] + " : ")
        print(commits)

    symfinder_project = open(sys.argv[1] + "/data/" + item["name"] + ".yaml", "w")
    symfinder_project.write(item["name"] + ":\n")
    symfinder_project.write("  repositoryUrl: \"" + item["clone_url"] + "\"\n")
    symfinder_project.write("  path: \"/data/projects\"\n")
    symfinder_project.write("  outputPath: \"/data/output\"\n")
    symfinder_project.write("  sourcePackage: .\n")
    symfinder_project.write("  commitIds:\n")
    symfinder_project.write("    - " + commits[0]["sha"])
    symfinder_project.close()

    print("Starting analyze variability...")
    os.chdir(sys.argv[1])
    symfinder = subprocess.Popen(
        "./run-docker-cli.sh -i /data/" + item["name"] + ".yaml -s /data/symfinder.yaml -verbosity OFF", shell=True)
    try:
        symfinder.wait()
    except subprocess.TimeoutExpired:
        print("The analysis takes too many times (>5m). Skip to the next project")
        stop_docker()
        os.chdir(retro_dir)

    print("Variability finished !")
    try:
        shutil.copy("data/output/symfinder_files/" + item["name"] + ".json", retro_dir + "/db.json")
    except FileNotFoundError:
        print("No variability analysis output found. Does the project finished properly ? Skip to the next project")
        os.chdir(retro_dir)

    os.chdir(retro_dir)
    print("Search of the paternity of the variability...")
    print(os.getcwd())
    retro = subprocess.Popen("python3 paternity_variability.py " + item["clone_url"], shell=True)
    retro.wait()
    print("Finish !")


signal.signal(signal.SIGINT, handler)


project_url = sys.argv[2]
split_url = project_url.split("/")
project_name = split_url[len(split_url)-1]
local_path = "./data/"+project_name

git_hub_name = split_url[len(split_url)-2]+"/"+split_url[len(split_url)-1]

# Clé d'authentification github 
auth = "METTEZ_VOTRE_CLE_API_GITHUB_ICI"
# Recherche des repositories Github en Java les plus connus
api_url = "https://api.github.com/repos/"+git_hub_name


item = req.get(api_url, headers={"Authorization": "Bearer " + auth}).json()
# On récupère le dossier
retro_dir = os.getcwd()
print("Here : "+retro_dir)
symfinder(item)

# Commande : python3 analyse_one_project_2.py /lien/vers/symfinder https://lienv_github
# /lien/vers/synfinder ==> où il y a le script "./run-docker-cli.sh"