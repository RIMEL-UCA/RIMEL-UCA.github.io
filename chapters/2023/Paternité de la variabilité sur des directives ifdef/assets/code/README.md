# Patternité de la variabilité en se basant sur la directive _#ifdef_

## Technologies utilisées

- Python et Pip
- Git Blame
- Git Fame
- Grep Unix

## Instructions

Pour lancer notre outil, il faut préalablement installer les dépendances python:

`pip install -r ./requirements.txt`

Puis un script `main.py` est disponible:

`python3 ./main.py`

Ce script va parcourir les fichiers C/C++ locaux au dossier à partir de sa racine. Nous avons cloné les sources de BusyBox pour que vous ayez un example fonctionnel.

Les résultats seront notés dans un fichier de sortie `out.json`, dans lequel vous retrouverez pour chaque feature, les contributeurs et le nombre de lignes contribuées associé.

Nous avons proposé le résultat d'exécutions déjà passé dans le répertoire `./outs`

Afin de lancer les visualisations il suffit de lancer `python3 ./plot.py` au même niveau que le `out.json`.

Si vous utilisez les exemples de `./outs`, n'oubliez pas de les renommer en `out.json`.