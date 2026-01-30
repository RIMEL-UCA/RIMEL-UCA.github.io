# Logique d'Inférence

Ce projet utilise un processus en deux étapes pour transformer un historique Git brut en rôles humains significatifs (ex : "Data Scientist", "Ingénieur DevOps").

### 1. Classification des Fichiers (`PathClassifier`)

D'abord, chaque fichier modifié dans un commit est classé dans une catégorie technique basée sur son **chemin (path)** et son **extension**. Nous utilisons des règles regex avec une priorité stricte pour gérer les ambiguïtés.

**Ordre de Priorité :**

1. **Infrastructure** (`Dockerfile`, `.github/`, `terraform`...) - *L'emporte sur tout le reste.*
2. **Documentation** (`*.md`, `docs/`...)
3. **Testing** (`tests/`, `*_test.py`...) - *Même si c'est du code Python, c'est du Test.*
4. **Data Science** (`notebooks/`, `*.ipynb`...)
5. **ML & Data** (`models/`, `*.parquet`, `etl/`...)
6. **Base de Données** (`migrations/`, `*.sql`...)
7. **Frontend** (`src/ui/`, `*.tsx`, `*.css`...)
8. **Backend** (`api/`, `*.go`, `*.py`...)
9. **Configuration** (`*.json`, `*.yaml`...)

> *Exemple :* Un fichier `backend/Dockerfile` est classé comme **Infra**, et non Backend.

### 2. Inférence des Rôles (`ContributorClassifier`)

Une fois que nous savons *ce que* une personne a touché (ex : 80% de fichiers Python, 20% de SQL), nous calculons un **Score Pondéré** pour chaque intitulé de poste potentiel.

**Comment ça marche :**
Le système compare le profil d'activité du contributeur par rapport à des "Profils Idéaux" prédéfinis.

| Rôle | Signal Fort | Signal Faible |
| --- | --- | --- |
| **Backend Engineer** | 100% Backend | - |
| **Data Engineer** | 70% Data + DB | 30% Backend + Infra |
| **Fullstack** | 70% Front + 70% Back | - |
| **DevOps / SRE** | 80% Infra | 20% Config |
| **Data Scientist** | 70% Exploration | 50% ML, 20% Data |

**L'Algorithme de Décision :**

1. Calcule un score pour chaque rôle basé sur les stats de l'utilisateur.
2. Sélectionne le rôle avec le score le plus élevé.
3. Vérification du Seuil : Si le meilleur score est inférieur à `0.25` (ce qui signifie que l'utilisateur touche à tout sans focus précis), il est étiqueté comme Généraliste.
4. S'il a 0 commit, il est Inactif.
