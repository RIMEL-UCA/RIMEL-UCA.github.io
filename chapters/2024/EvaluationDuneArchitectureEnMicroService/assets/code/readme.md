# Analyses des patterns

Le script analyse.py permet de faire une analyse en largeur de tous les projets du fichier "repos.csv". Il faut génerer un token Github pour pouvoir explorer les repositories GitHub avec l'api PyGitHub. 

Le script utilise des classes auxiliares mentionnées dans les imports. Chaque classe est responsable d'analyser un pattern spécifique. 
Le résultat de l'analyse sera un fichier output.xlsx  dans un dossier output qui va contenir le résultat de chaque sous-analyse. 

# Script de validation 

Le script validation.py permet de chercher les mots-clés en relation avec les mciroservices dans les readme, descriptions, nom des fichiers, etc. De plus, il permet de trouver le service/ répertoire le plus volumineux dans le projet.
Le résultat du script sera également dans un fichier validation.xlsx dans le dossier output.

## Lancement des scripts 

Analyse des patterns :

python3 analyse.py --token your_token --input path_of_csv_repos

Script de validation : 

python3 validation.py --token your_token --input path_of_csv_repos








