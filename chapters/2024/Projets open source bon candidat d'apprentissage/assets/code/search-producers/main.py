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
import argparse


def delete_folder(path):
    """
    Supprimer un dossier
    :param path: chemin du dossier à supprimer
    """
    if os.path.exists(path):
        shutil.rmtree(f'./projects/{project_name}')
        print(f"Project {project_name} deleted")

def clone_project(project_name):
    """
    Cloner un projet github
    :param project_name: nom du projet à cloner
    """
    # Cloner le projet
    os.chdir('./projects')
    subprocess.run(['git', 'clone', project])
    os.chdir('..')
    print(f"Project {project_name} cloned")


if __name__ == '__main__':

    # Générer un identifiant du run
    run_id = str(uuid.uuid4())

    parser = argparse.ArgumentParser(description='Extract kafka producers from a github project')
    parser.add_argument('project', metavar='project', type=str, help='github project to analyze')
    parser.add_argument('output', metavar='output', type=str, help='output file',
                        default=f'./outputs/output-{run_id}.json')
    args = parser.parse_args()

    # Récupérer le projet github passé en paramètre
    project = args.project
    output = args.output
    print(f"Analyzing project {project}...")

    # Cloner le projet
    project_name = project.split('/')[-1]
    clone_project(project_name)

    # Parcourir les fichiers du projet
    os.chdir(project_name)
    producers = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.java') or file.endswith('.kt') or file.endswith('.js') or file.endswith(
                    '.py') or file.endswith('.ts') or file.endswith('.go') or file.endswith('.cs'):
                # Lire le fichier
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    # Extraire les connexions à un bus kafka
                    matches = re.findall(r'new\s+KafkaProducer\(\s*new\s+Properties\(\)\s*\)', content)
                    if matches:
                        # Récupérer le nom du bus kafka et du topic
                        kafka = re.findall(r'bootstrap.servers\s*=\s*"(.*)"', content)
                        topic = re.findall(r'topic\s*=\s*"(.*)"', content)
                        if kafka and topic:
                            # Ajouter la connexion à la liste des connexions
                            producer = {
                                "name": project_name,
                                "kafka": kafka[0],
                                "topic": topic[0],
                                "type": "producer"
                            }
                            producers.append(producer)
                            print(f"Producer found in {os.path.join(root, file)}")
    os.chdir('../..')

    # Supprimer le projet
    delete_folder(project_name)

    # Ecrire les connexions dans un fichier json
    with open(output, 'w') as f:
        json.dump(producers, f)
    print(f"Producers written in {output}")
