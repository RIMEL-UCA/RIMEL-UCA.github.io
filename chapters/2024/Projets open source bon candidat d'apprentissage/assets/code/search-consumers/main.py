"""
@Author: Team C
@Created: 04/12/2023 16:58
@Desc: Ce script permet de récupérer le projet github passé en paramètre, de le cloner, et de parcourir ses fichiers
pour en extraire les connexions à un bus kafka. Ce script se concentre sur les producer kafka des langages Java,
Kotlin, Javascript, Python, Typescript, Go, C#. Ces connexions sont ensuite stockées dans un fichier json où chaque
connexion est représentée par un objet de la forme :
{
    "name": "nom_du_projet",
    "kafka": "nom_du_bus_kafka",
    "topic": "nom_du_topic",
    "type": "producer"
}
"""
import uuid
import os
import json
import shutil
import subprocess
import re
import stat
import time
import argparse
from common.utils import BCOLORS


def delete_folder(project_name):
    """
    Supprimer un dossier
    :param path: chemin du dossier à supprimer
    """
    path = f'./projects/{project_name}'
    # Is the error an access error?
    os.chmod(path, stat.S_IWUSR)
    for r, _, ff in os.walk(path):
        for momo in dirs:
            os.chmod(os.path.join(r, momo), stat.S_IWUSR)
        for momo in ff:
            os.chmod(os.path.join(r, momo), stat.S_IWUSR)
    print(f"{BCOLORS.OKBLUE}Permissions changed{BCOLORS.ENDC}")
    if os.path.exists(path):
        print(f"{BCOLORS.OKBLUE}Deleting project {project_name}...{BCOLORS.ENDC}")
        # Delete using absolute path
        os.rmdir(f'{os.getcwd()}/projects/{project_name}')
        print(f"{BCOLORS.OKBLUE}Project {project_name} deleted{BCOLORS.ENDC}")
    else:
        print(f"{BCOLORS.WARNING}Project {project_name} does not exist{BCOLORS.ENDC}")


def clone_project(project_name):
    """
    Cloner un projet github
    :param project_name: nom du projet à cloner
    """
    # Vérifier si le dossier ./projects existe, sinon le créer
    if not os.path.exists('./projects'):
        print(f"{BCOLORS.WARNING}Folder ./projects does not exist{BCOLORS.ENDC}")
        os.makedirs('./projects')
        print(f"{BCOLORS.OKBLUE}Folder ./projects created{BCOLORS.ENDC}")

    # Cloner le projet s'il n'existe pas
    if os.path.exists(f'./projects/{project_name.split("/")[1]}'):
        print(f"{BCOLORS.WARNING}Project {project_name} already exists{BCOLORS.ENDC}")
        return
    os.chdir('./projects')
    print(f"{BCOLORS.OKBLUE}Cloning project {project_name}...{BCOLORS.ENDC}")
    subprocess.run(['git', 'clone', f'https://github.com/{project}'])
    os.chdir('..')
    print(f"{BCOLORS.OKBLUE}Project {project_name} cloned{BCOLORS.ENDC}")


def save_consumers(consumers_list, consumers_output):
    """
    Sauvegarder les producer kafka dans un fichier json
    :param consumers_list: liste des producer kafka
    :param consumers_output: fichier de sortie
    """
    if len(consumers_list) == 0:
        print(f"{BCOLORS.WARNING}No consumers found{BCOLORS.ENDC}")
        exit(0)
    with open(consumers_output, 'w') as ofile:
        json.dump(consumers_list, ofile, indent=4)
        print(f"{BCOLORS.OKGREEN}Consumers written in {consumers_output}{BCOLORS.ENDC}")


def get_consumers(content, file, project_name, service, config_value_cache=None):
    """
    Extraire les consommateurs Kafka d'un fichier en trouvant les annotations @KafkaListener
    :param content: contenu du fichier
    :param file: nom du fichier
    :param project_name: nom du projet
    :param service: nom du service
    :param config_value_cache: cache des valeurs de configuration
    :return: liste des consommateurs Kafka
    """
    if config_value_cache is None:
        config_value_cache = {}
    
    content = replace_config_values(content, config_value_cache, service)

    consumers = []
    matches = re.findall(r'@KafkaListener\(.*?topics.*?=.*?\"(.*?)\"', content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        consumers.append({
            "project": project_name,
            "topic": match,
            "type": "consumer",
            "file": file,
            "service": service
        })
    
    return consumers



def replace_config_values(content, config_value_cache={}, service=''):
    """
    Remplacer les valeurs de configuration Spring par leur valeur
    :param content: contenu du fichier
    :param config_value_cache: cache des valeurs de configuration
    :param service: nom du service pour restreindre la recherche des fichiers de configuration
    :return: contenu du fichier avec les valeurs de configuration Spring remplacées
    """
    matches = re.findall(r'@Value.*{(.*)}.*\s+([a-zA-Z]+)[,;]$', content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if matches:
        config_name = matches[0][0]
        var_in_file = matches[0][1]
        # Parcours des fichiers de configuration .properties pour récupérer la valeur de la variable de configuration
        # appelée config_name
        config_value = config_value_cache.get(config_name)
        if not config_value:
            for root, dirs, files in os.walk(f'./{service}'):
                for file in files:
                    if file.endswith('.properties'):
                        with open(os.path.join(root, file), 'r') as f:
                            config_file_content = f.read()
                            matches = re.findall(rf'{config_name}=(.*)', config_file_content)
                            if matches:
                                config_value = matches[0]
                                config_value_cache[config_name] = config_value
                                print(
                                    f'{BCOLORS.OKBLUE}Config value {config_name} found: {config_value} in file {os.path.join(root, file)} {BCOLORS.ENDC}')
                                return content.replace(f'{var_in_file})', config_value)
        else:
            print(f'{BCOLORS.OKBLUE}Config value {config_name} found in cache: {config_value} {BCOLORS.ENDC}')
            return content.replace(f'{var_in_file})', config_value)
        # Remplacer la valeur de la variable de configuration par sa valeur
        print(f"{BCOLORS.FAIL}Config value {config_name} not found. (might be out of scope){BCOLORS.ENDC}")
    return content

def get_microservice_name(path):
    while True:
        path, folder = os.path.split(path)
        if folder == "src":
            return os.path.basename(path)
        elif not folder:
            break
    return os.path.basename(path)

if __name__ == '__main__':

    # Générer un identifiant du run
    run_id = str(uuid.uuid4())

    parser = argparse.ArgumentParser(description='Extract kafka Consumers from a github project')
    parser.add_argument('-p', '--project', metavar='project', type=str, help='github project to analyze')
    parser.add_argument('-o', '--output', metavar='output', type=str, help='output file')
    args = parser.parse_args()

    # Récupérer le projet github passé en paramètre
    project = args.project
    project_name = project.split('/')[1]
    project_folder = f"./projects/{project_name}"
    output = args.output if args.output else f'./outputs/{project_name}-{run_id}--search-consumers.json'
    print(f"{BCOLORS.HEADER}Analyzing project {project}...{BCOLORS.ENDC}")

    # Cloner le projet
    clone_project(project)

    # Parcourir les fichiers du projet
    os.chdir(project_folder)
    consumers = []
    config_value_cache = {}
    for root, dirs, files in os.walk('.'):
        for file in files:
            # print(root, file)
            if file.endswith('.java') or file.endswith('.kt') or file.endswith('.js') or file.endswith(
                    '.py') or file.endswith('.ts') or file.endswith('.go') or file.endswith('.cs'):
                # Lire le fichier
                with open(os.path.join(root, file), 'r') as f:
                    service = get_microservice_name(root)
                    content = f.read()
                    consumers_in_file = get_consumers(content, file, project_name, service, config_value_cache)
                    if len(consumers_in_file) > 0:
                        consumers.append(consumers_in_file)

    os.chdir('../..')
    print(f"{BCOLORS.OKGREEN}Project {project_name} analyzed{BCOLORS.ENDC}")

    # Supprimer le projet
    # delete_folder(project_name)

    # Créer le dossier ./outputs s'il n'existe pas
    if not os.path.exists('./outputs'):
        print(f"{BCOLORS.WARNING}Folder ./outputs does not exist{BCOLORS.ENDC}")
        os.makedirs('./outputs')
        print(f"{BCOLORS.OKBLUE}Folder ./outputs created{BCOLORS.ENDC}")

    # Ecrire les connexions dans un fichier json
    save_consumers(consumers, output)
