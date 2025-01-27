# CODES D'ANALYSES DES FORKS
Pour lancer un fichier, il faut aller dans le dossier. Depuis le terminal :
```cd ./nom_du_dossier/```

Puis : ```python3 nom_du_fichier.py```

Il est **obligatoire** de lancer les fichiers dans un certain ordre dans chaque dossier :
- Pour humanEval : [fork-recuperation.py](./humanEval/fork-recuperation.py) puis [analyze_fork_data.py](./humanEval/analyze_fork_data.py)
- Pour evalplus : [commit-recuperation.py](./evalplus/commit-recuperation.py) puis [commit-stats.py](./evalplus/commit-stats.py) ensuite [issues_recuperation.py](./evalplus/issues_recuperation.py) et enfin [show_commit_stats.py](./evalplus/show_commit_stats.py)

Pour cela, il faut en premier lieu remplir la partie token des fichiers de recupération avec le token que vous aura délivré GitHub.
