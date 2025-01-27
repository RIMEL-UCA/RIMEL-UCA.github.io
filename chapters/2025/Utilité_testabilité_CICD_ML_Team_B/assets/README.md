# Exécution des codes implémentée

## Diagrammes des workflows

Pour générer un diagramme à partir d'un fichier de description de workflow en Mermaid ( `dans le répertoire workflows`), voici les étapes nécessaires :

1. **Installation de `mermaid-cli` :**  
Tout d'abord, installez le client en ligne de commande `mermaid-cli` (mermaid-js) en utilisant npm (Node Package Manager). Cela vous permettra de convertir des fichiers `.mmd` en images vectorielles (SVG). Utilisez la commande suivante :
```bash
npm install -g @mermaid-js/mermaid-cli
```
2. **Génération du diagramme :**
Une fois mermaid-cli installé, vous pouvez générer le diagramme à partir de votre fichier de description devcontainer.mmd. Cette commande lit le fichier .mmd et produit un fichier SVG (devcontainer.svg) qui représente visuellement votre workflow :

```bash
mmdc -i devcontainer.mmd -o ../../images/devcontainer.svg
```

- -i devcontainer.mmd : spécifie le fichier d'entrée, qui est votre description du workflow au format Mermaid.
- -o ../../images/devcontainer.svg : définit le fichier de sortie sous forme d'image SVG, dans ce cas, un fichier nommé devcontainer.svg.

Cette méthode permet de transformer des descriptions textuelles de workflows en représentations graphiques.


## Scripts python

Les images résultats générés par les scripts pythons seront accessible dans : `assests\images`

### `requirements.txt`
Le fichier `requirements.txt` contient toutes les dépendances nécessaires pour exécuter ces scripts. Assurez-vous d'installer ces dépendances avant de lancer les scripts :  
```bash
pip install -r requirements.txt
```

### `datefirstcommit.py`
Ce script génère un tableau contenant :  
- La date du premier commit pour chaque outil.  
- Le nombre de contributeurs associés à chaque dépôt.  
```bash
python3 src/datefirstcommit.py
```

### `gitnbcommits.py`
Ce script produit un graphique interactif avec Plotly, montrant :  
- Le nombre de commits pour chaque cible (par exemple, dossiers `tests` et `workflows`) dans les dépôts analysés.  
- Le graphique est exporté sous forme d'un fichier HTML pour une visualisation interactive.

```bash
python3 src/gitnbcommits.py
```


