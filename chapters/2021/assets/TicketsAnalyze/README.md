# Groupe C - TicketsAnalyze - 

## API JIRA AUTOMATION
Afin d'avoir les différents graphs qui existent dans notre modèle il suffit d'éxecuter ces étapes : 
* cd jira_api_automation 
* npm install
* npm start 
### Lien vers la collection postman : https://www.getpostman.com/collections/9cca438d452ea0dc099c


## Coupling 

Ce code permet de tenter de vérifier l'hypothèse de couplage entre les composants.

Vous devez nécéssairement avoir Nodejs et npm d'installés pour pouvoir le lancer.

* Vous devez lancer "npm install" quand vous démarrez le projet pour la première fois.
* Pour démarrer le projet, lancez "./start.sh".

Le dossier "results samples" contient les résultats que nous avons trouvés:
* "1000.json" est le fichier qui contient le lien entre les noms des fichiers et les composants métiers, en utilisant 1000 tickets.
* "result-10000.json" est le fichier qui contient le résultat final. Il affiche les tickets dont le couplage a été vérifié par notre algorithme. À la fin il indique le nombre de tickets analysés, qui correspond au nombre de tickets liés à 2 composants qui ont un lien github dans leur description.

Le faible nombre de tickets qu'il est possible d'analyser (peu de liens github en description des tickets dans le projet Server sur le JIRA de MongoDB, et peu de tickets liés à 2 composants) ne permet pas de conclure.

L'API de Jira peut ne pas fonctionner en faisant une requête pour 10 000 tickets. Vous devez parfois réduire ce nombre si vous voulez tester le projet, en modifiant la valeur de "maxResults" en haut du fichier "findCoupling.js". Vous pouvez essayer avec 5000 tickets, puis 1000 (qui est certain de fonctionner).

Vous n'êtes pas obligés de lancer "./start.sh" si le fichier "1000.json" est déjà créé dans le dossier "coupling", et qu'il est correctement complété. Vous pouvez simplement lancer la commande "node findCoupling.js".

Le projet Server de MongoDB utilisé : https://jira.mongodb.org/browse/SERVER-52224?jql=project%20%3D%20SERVER%20ORDER%20BY%20priority%20DESC%2C%20updated%20DESC


