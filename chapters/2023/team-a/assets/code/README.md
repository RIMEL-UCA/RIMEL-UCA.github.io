# Maintenabilité d’une pipeline : paternité et dépendances de l’implémentation des jobs et des steps 

> Le chapitre disponible est [ici](https://rimel-uca.github.io/chapters/2023/team-a/content).

[Ce dépôt](https://github.com/directionmiage/rimel) est uniquement dédié aux scripts qui amènent à la production d'un résultat de notre recherche.

## Démarrage rapide

### Prérequis

- [Python 3.9+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [Token GitHub](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)

### Installation

Installez les dépendances :

```bash
pip install -r requirements.txt
```

Créez un fichier `.env.dev` à la racine du projet et ajoutez-y votre token GitHub :

```bash
GITHUB_ACCESS_TOKEN=<INSER_YOUR_TOKEN_HERE>
```

### Utilisation

```
python main.py -P <path_to_input_project.yml>
```

> ⭐️ Vous pouvez vous baser sur le fichier [exemple](input.sample.yml) pour créer votre fichier d'entrée.

## Licence

Ce projet est sous licence propriétaire.