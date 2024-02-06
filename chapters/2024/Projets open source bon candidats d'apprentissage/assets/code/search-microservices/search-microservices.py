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
import json
import uuid
import os
import shutil
import subprocess
import stat
import argparse
import yaml
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
        shutil.rmtree(f'{os.getcwd()}\\projects\\{project_name}')
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


def get_services(file, project):
    keys_to_remove = ['zookeeper', 'kafka', 'mongo', 'broker', 'redis', 'database', 'postgres', 'grafana', 'cassandra', 'keycloak', 'monitoring', 'prometheus','elasticsearch', 'elastic-search','apache','api-gateway']

    terms_to_check = ['db', 'database', 'kafka', 'redis', 'mongo', 'mongodb','grafana','kibana','cassandra',
                      'keycloak','monitoring','prometheus','zookeeper','broker','rabbitmq','webapp', 'apache', 
                      'postgres','mysql', 'config', 'zipkin', 'discovery','jaeger','elasticsearch','elastic-search','otel-collector'
                      ,'nginx','apm-server', 'control-center','schema','gateway','kafdrop','monitor','PostgreSQL','pgadmin']

    docker_compose_config = yaml.safe_load(file)
    services = docker_compose_config.get('services', {})
    
    def should_remove(key):
        return any(term in key for term in terms_to_check) or key in keys_to_remove
    
    filtered_keys = [key for key in services.keys() if not should_remove(key)]
    
    return filtered_keys



def save_microservices(microservices_list, microservices_output):
    """
    Ecrire les connexions dans un fichier json
    :param producers_list: liste des connexions
    :param producers_output: fichier de sortie
    """
    # Applatire les sous listes
    microservices_list = [item for sublist in microservices_list for item in sublist]
    # Supprimer les doublons
    microservices_list = list(dict.fromkeys(microservices_list))
    print(f"{BCOLORS.OKBLUE}Saving microservices...{BCOLORS.ENDC}")
    with open(microservices_output, 'w') as f:
        json.dump(microservices_list, f, indent=4)
    print(f"{BCOLORS.OKBLUE}Microservices saved in {microservices_output}{BCOLORS.ENDC}")


if __name__ == '__main__':

    # Générer un identifiant du run
    run_id = str(uuid.uuid4())

    parser = argparse.ArgumentParser(description='Extract kafka producers from a github project')
    parser.add_argument('-p', '--project', metavar='project', type=str, help='github project to analyze')
    parser.add_argument('-o', '--output', metavar='output', type=str, help='output file')
    args = parser.parse_args()

    # Récupérer le projet github passé en paramètre
    project = args.project
    project_name = project.split('/')[1]
    project_folder = f"./projects/{project_name}"
    output = args.output if args.output else f'./outputs/{project_name}-{run_id}--search-microservices.json'
    print(f"{BCOLORS.HEADER}Analyzing project {project}...{BCOLORS.ENDC}")

    # Cloner le projet
    clone_project(project)

    # Parcourir les fichiers du projet
    os.chdir(project_folder)
    microservices = []
    config_value_cache = {}
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('docker-compose.yml'):
                # Lire le fichier
                with open(os.path.join(root, file), 'r') as f:
                    services = get_services(f, project)
                    microservices.append(services)

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
    save_microservices(microservices, output)
