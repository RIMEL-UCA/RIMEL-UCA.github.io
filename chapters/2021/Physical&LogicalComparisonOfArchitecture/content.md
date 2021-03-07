---
layout: default
title : Comparaison architecturale physique et logicielle
date:   2021-01-17 14:40:00 +0100
---

---

> **Date de rendu finale : Mars 2021 au plus tard**
> - Respecter la structure pour que les chapitres soient bien indépendants
> - Remarques :
>>    - Les titres peuvent changer pour être en adéquation avec votre étude.
>>    - De même il est possible de modifier la structure, celle qui est proposée ici est là pour vous aider.
>>    - Utiliser des références pour justifier votre argumentaire, vos choix etc.

---

**_janvier 2021_**

## Authors

We are four students in last year of Polytech' Nice-Sophia specialized in Software Architecture :

* Loïc Bertin &lt;loic.bertin@etu.univ-cotedazur.fr&gt;
* Virgile Fantauzzi &lt;virgile.fantauzzi@etu.univ-cotedazur.fr&gt;
* Guillaume Ladorme &lt;guillaume.ladorme@etu.univ-cotedazur.fr&gt;
* Stéphane Viale &lt;stephane.viale2@etu.univ-cotedazur.fr&gt;

## I. Research context /Project

Notre contexte d'étude se base sur la comparaison des applications de détection et de tracage du COVID-19 et en particulier sur les applications françaises et canadiennes. Nous allons donc nous baser sur l'application TousAntiCovid (ex StopCovid) et l'application CovidShield. 
- Projet canadien : https://github.com/CovidShield/
- Projet français : https://gitlab.inria.fr/stopcovid19

Avec toutes les plaintes autour de la sécurisation de nos données et de la conservation de la vie privée, l'étude de ces deux projets est très intéréssante afin de comprendre si l'inquiétude générale est justifiée ou non. 
De plus, en tant qu'élève architecte logicielle, la rétro ingénierie de projet tels que ceux ci nous permet de comprendre et d'analyser les choix qui ont été fait, à nuancer évidemment avec la rapidité des décisions et les contraintes temporelles dûes à la crise.

## II. Observations/General question

Notre problématique est issue d'une idée qui nous est venu en tant que citoyen français qui commençons à découvrir petit à petit la complexité de notre système. En effet, lors du développement de l'application, les développeurs ont du prendre en compte toutes les nuances du système de santé français afin d'informer les bonnes institutions et de s'interfacer correctement avec les organismes déjà en place. C'est cette complexité qui nous a intrigué et qui nous a donné envie de répondre à cette problématique générale : 
**En quoi l'architecture des projets reflète l'organisation administrative des pays et leur gestion de la crise du covid-19**

Cette problématique va nous permettre de nous intérésser à la fois à l'architecture globale des projets mais aussi de venir investiguer dans le code l'implémentation concrète des mesures gouvernementales. Il y aura donc 2 axes de reflexion à suivre, portés sur différentes échelles de vision de l'architecture, une vision gros grain et un zoom dans le code.
La question étant beaucoup trop générale et impossible à traiter par une équipe de 4 personnes avec le temps accordé, nous nous sommes intéréssés à deux sous questions qui seront détaillées plus bas dans ce rapport. Ces deux sous questions nous paraissent très intéréssante car elles reprennent l'idée générale de la problématique globale mais axée sur de la compréhension des projets et leur comparaison.

## III. information gathering

Préciser vos zones de recherches en fonction de votre projet,

1. les articles ou documents utiles à votre projet
2. les outils
 
## IV. Hypothesis & Experiences

1. Il s'agit ici d'énoncer sous forme d' hypothèses ce que vous allez chercher à démontrer. Vous devez définir vos hypothèses de façon à pouvoir les _mesurer facilement._ Bien sûr, votre hypothèse devrait être construite de manière à v_ous aider à répondre à votre question initiale_.Explicitez ces différents points.
2. Test de l’hypothèse par l’expérimentation. 1. Vos tests d’expérimentations permettent de vérifier si vos hypothèses sont vraies ou fausses. 2. Il est possible que vous deviez répéter vos expérimentations pour vous assurer que les premiers résultats ne sont pas seulement un accident.
3. Explicitez bien les outils utilisés et comment.
4. Justifiez vos choix

## V. Result Analysis and Conclusion

1. Analyse des résultats & construction d’une conclusion : Une fois votre expérience terminée, vous récupérez vos mesures et vous les analysez pour voir si votre hypothèse tient la route. 

### En quoi les dépendances externes reflètent l’organisation administrative du pays autour de la crise du COVID-19 ?

#### France

En se basant sur cette architecture nous sommes allé cherché dans le code comment tous-anti-covid communiquait et ce qu’elle communiquait aux services externes.
![Figure 1: tous anti covid archi](../assets/Physical&LogicalComparisonOfArchitecture/TousAntiCovidArchi.png)
Nous voyons ici deux points qui vont nous intéresser dans cette question:
- le back-end européen pour discovery
- L’autorité de santé française dont la SIDEP fait partie

Pour le lien avec la SIDEP, nous voyons dans le bucket submission code server_snapshot que dans le fichier submission-code-server-ws-rest.src.main.java.fr.gouv.stopc.submission.code.server.ws.service.SFTPServiceImpl.java il y a un transfert de fichier par SFTP où un zip est transmis à la SIDEP. Ce zip est créé dans la méthode zipExport de la même classe où l’on va créer un zip avec les informations récoltées entre deux dates. En effet, la SIDEP étant un organisme de suivi exhaustif des tests réalisés ainsi que différentes autres informations pour suivre l’évolution de l’épidémie, la SIDEP est donc intéressée de recevoir les informations sur les cas contacts détectés par tous-anti-covid.
	
 En ce qui concerne les communications avec le back-end européen, pour identifier le pays source, chaque pays a un code, il s’agit en réalité du même code que l’indicateur téléphonique du pays, soit 33 écrit en hexadécimal soit 0x21 pour la France.
	
 Dans le bucket robert-server-develop, dans le fichier robert-server-batch.src.main.java.fr.gouv.stopc.robert.server.batch.processor.ContactProcessor.java Cette méthode est utilisée pour traiter chacun des messages et valider si ceux-ci sont correct et atteste d’un cas contact (à la fin de cette méthode on vérifie que le temps est inférieur au temps max d’une contamination). Mais ce qui nous intéresse ici c’est que l’on observe une tentative d’application européenne à la ligne 119. 

![Figure 2: tous anti covid communication europeenne](../assets/Physical&LogicalComparisonOfArchitecture/TousAntiCovid-communication-europeenne.png)

En effet, on observe ci-dessus que cette gestion du countryCode devait permettre de rediriger un message vers le bon serveur de traitement et ainsi pouvoir identifier des cas contact dans toute l’Europe. Ce qui peut expliquer pourquoi l’Union européenne à toujours chercher à garder la zone schengen ouverte au maximum. Toutefois, une communication avec l’union européenne est bien présente dans l’application.
En effet, le code que je vous ai présenté ci-dessus est en réalité du code mort, on voit dans la documentation de la fonction que les spécifications ont évoluées et que la validation des messages se fait maintenant dans le “crypto back-end”. En parcourant le repository on voit que dans robert-crypto-grpc-server.src.main.java.fr.gouv.stopc.robert.crypto.grpc.server.impl.CryptoGrpcServiceBaseImpl.java nous trouvons la méthode “getInfoFromHelloMessage” Qui réalise le travail décrit précédemment. Toutefois cette implémentation en grpc est cryptée et dans un premier temps il faut décrypter l’ECC (Encrypted Country Code) où cette fois, si celui-ci ne correspond pas à la celui de la France il est bien renvoyé au serveur européen correspondant dans le but de lui envoyer l’information qu’un de ses citoyens est un cas contact.

En conclusion, la France utilise différents services externes. Tout d’abord, elle communique son nombre de cas contact à la SIDEP pour qu’elle puisse évaluer la situation journalière de plus lorsqu’un cas contact européen est détecté le pays concerné est prévenu ce qui montre que la gestion française de la crise est en fait une gestion européenne. Bien que certains services propres à la France soient utilisés dans cette crise (comme Ameli qui permet aux professionnels de santé de se connecter au web pro).


#### Canada

![Figure 3: method claim kay canada](../assets/Physical&LogicalComparisonOfArchitecture/canadaCodeClaimKey.png)

### Comment est implémenté la gestion de la distanciation sociale et des cas contacts dans les applications ?

#### France

#### Canada

L’application canadienne Android et iOS font appel au service Exposure Notification, un service réalisé conjointement par Apple et Google pour combattre le COVID-19 qui permet à tous les appareils (iOS et Android) de pouvoir communiquer à travers ce service.
Nous retrouvons des explications détaillées du service Exposure Notification sur chacun des systèmes d’exploitation:
- Android : [https://support.google.com/android/answer/9888358](https://support.google.com/android/answer/9888358)
- Apple : [ExposureNotification-BluetoothSpecificationv1.2.pdf](https://covid19-static.cdn-apple.com/applications/covid19/current/static/contact-tracing/pdf/ExposureNotification-BluetoothSpecificationv1.2.pdf)
Malheureusement, ce service ne calcule pas la distance entre deux appareils. Cependant, si on suit la spécification de Google :

![Figure 4: Détails sur la gestion de la distanciation](../assets/Physical&LogicalComparisonOfArchitecture/exposureNotificationDetails.png)

On remarque que la distance entre deux appareils est basée sur l’intensité du signal bluetooth. Sachant que ce service fait appel au Bluetooth Low Energy. Si on se réfère au spécification de cette technologie, cela correspond à une distance théorique inférieure à 100 mètres.

Pour ce qui est des cas contacts, l’implémentation est toujours basée sur le service Exposure Notification. Le service génère un identifiant qui correspond à une séquence de nombre aléatoire toutes les 10-20 minutes. Le service partage cet identifiant en arrière-plan avec les autres appareils aux alentours. Lorsque le service détecte un identifiant d’un autre appareil, il l’enregistre et si cet identifiant est signalé comme positif au COVID, le service notifie l’application.

### Ces implémentations ont-elles évoluées au fil des décisions gouvernementales ?

#### France

#### Canada

Ces implémentations sur les applications canadiennes se basent essentiellement sur le service Exposure Notification. La majorité des commits sur le répertoire des applications ([https://github.com/CovidShield/mobile/commits/master](https://github.com/CovidShield/mobile/commits/master)) sont des commits de maintenance pour fixer des soucis techniques. C’est encore plus flagrant avec ce graphe qui affiche le nombre de commits par jours sur le répertoire des applications :

![Figure 5: Commits par jour sur le répertoire des applications iOS et Android](../assets/Physical&LogicalComparisonOfArchitecture/commitsByDaysMobile.png)

Les derniers commits datent du 28 octobre 2020. Ce qui nous a surpris, en cherchant un peu, il semblerait que le Canada n’utilise plus CovidShield mais COVID Alert ([https://github.com/cds-snc](https://github.com/cds-snc)).

## VI. Tools

Pour avoir le graphique des commits par jour pour voir l'évolution des applications avec les décisions gouvernementales, nous avons développé un script qui communique avec l'API de Github. Qui peut se trouver [ici](../assets/Physical&LogicalComparisonOfArchitecture/code/commitsAnalyzer.py). Pour l'executer, il faut utiliser : `python commitsAnalyzer.py`. Par défaut, le script analyse le répertoire mobile de l'organisation CovidShield. Pour modifier de répertoire, il suffit de modifier les variables qui se situe à la ligne 36 et 37.

## VI. References

1. ref1
1. ref2

