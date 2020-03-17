# RIMEL-Maven-profiles

## Accéder aux bases de données

Si vous souhaitez accéder aux bases de données en local, vous pouvez suivre la démarche suivante :

- Dézipper l’archive que vous souhaitez (neo4J-BD1 ou neo4J-BD2)

- Placer le dossier extrait au même niveau que le docker compose.

- Lancer le docker compose avec la commande docker-compose up.

- Vous pouvez ensuite aller sur localhost:7474 pour accéder à la console neo4j.

- Les identifiants sont : neo4j / root

## Reconstituer la première base de données

Si vous souhaitez reconstituer la première base de données,  vous pouvez suivre la démarche suivante :

Pré requis : Lancer Neo4J via le docker-compose (il est préférable que la base soit vierge).

- Ouvrir le projet definition-tools dans votre IDE favori.

- Choisir l’intervalle de taille des poms recuperé (en octets) dans la classe Analyzer.

- Fournir un token de connection à l’API GitHub ainsi que son username dans la classe Analyzer.

- Lancer l’outil avec `analyzer.run()` dans la méthode `doSomethingAfterStartup()` dans `CategorizeApplication`.


## Reconstituer la seconde base de données

Si vous souhaitez reconstituer la seconde base de données,  vous pouvez suivre la démarche suivante :

Pré requis : Lancer Neo4J via le docker-compose (il est préférable que la base soit vierge).

Vous devez ensuite placer un fichier nommé stats-final-real.csv qui constitue la liste des repositories à analyser, vous pouvez utiliser celui fourni pour étudier les mêmes projets ou suivre les indications données dans la partie suivante pour recréer ce csv.

Vous pouvez aussi placer un fichier banned.csv qui va permettre de ne pas prendre en compte les repositories spécifiés dans ce fichier dans l’étude.

- Ouvrir le projet event-search-tools dans votre IDE favori.

- Dans la classe vous pouvez spécifier le nombre de commit minimum des projets à étudier.

- Lancer l’outil.

## Refaire la réduction de l’espace de recherche entre la base de données 1 et l’entrée de la seconde analyse.


Pour lancer l’analyse, vous devez lancer la base neo4J-BD1 en suivant les instructions de la première partie. Vous devez ensuite suivre les étapes suivantes :

- Ouvrir le projet definition-tools dans votre IDE favori.

- Fournir un token de connection à l’API GitHub ainsi que son username dans la classe StatisticsAnalyser.

- Lancer l’outil avec `statAnalyser.run();` dans la méthode `doSomethingAfterStartup()` dans `CategorizeApplication`.
