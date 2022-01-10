---
layout: default
title:  Automatiser la génération des fichiers de configuration (IaC) en Terraform d'un projet existant ?
date:   2022-01-10 21:241:00 +0100
---

**_janvier 2022_**

## Auteurs

Nous sommes quatre étudiants en dernière année à Polytech Nice Sophia, spécialité Architectures Logicielles :

* ANAGONOU Patrick &#60;sourou-patrick.anagonou@etu.univ-cotedazur.fr&#62;
* FRANCIS Anas &#60;anas.francis@etu.univ-cotedazur.fr&#62;
* ANIGLO Jonas &#60;jonas-vihoale.aniglo@etu.univ-cotedazur.fr&#62;
* ZABOURDINE Souleiman &#60;mohamedsoulaiman-tha.zabourdine@etu.univ-cotedazur.fr&#62;

Sujet 5 : Quelles (bonnes) pratiques automatisables pour les systèmes de déploiement comme Ansible ou Terraform ? 

​
## Question principale :
​
Comment automatiser la génération des fichiers de configuration (IaC) en
Terraform d\'un projet existant ?
​
Nous pensons que cette question est pertinente parce qu'un Template de
configuration peut aider les développeurs à développer et déployer plus
vite. Pour cela nous comptons analyser des projets open source utilisant
GitHub pour leur déploiement. A partir de ces projets nous comptons
identifier les pratiques qui se répètent (utilisation fréquente de
certains ressources ou providers) ensemble, identifier quels sont les
morceaux de configuration qui sont les plus utilisés par type de projet
(langage, framework), Nous pouvons aussi utiliser des règles de bonnes
pratiques définies par [SonarSource](https://www.sonarsource.com/) pour
les appliquer aux Templates. Nous avons fait le choix d'utiliser
Terraform au lieu d'Ansible parce qu'il a sa propre extension de fichier
et langage dédié HCL (contrairement à Ansible qui utilise YAML) ce qui
peut rendre l'identification d'un projet utilisant Terraform plus
facile. Etant un outil déclaratif Terraform nous permettra de définir le
modèle d'exécution du projet à automatisé contrairement à Ansible
(impératif) qui est plus orienté sur comment le faire.
​
« Terraformer » un projet existant n'est pas une idée complètement
nouvellement, par exemple l'outil terraformer
([Terraformer](https://github.com/GoogleCloudPlatform/terraformer)) a
pour objectif de générer les fichiers terraform correspondants à un
déploiement cloud existant en y accédant via l'API du fournisseur. Nous
comptons étudier s'il est possible de partir des codes sources sans
accès à un déploiement existant pour générer des fichiers de déploiement
complets ou des Templates sur lesquels les développeurs peuvent se
baser.
​
## Sous questions :
​
-   Comment analyser les fichiers d'un projet pour permettre la
    génération du fichier (IaC)
​
-   Quelles sont les fichiers à analyser en priorité ? (Le type et le
    contenu des fichier de build, Dockerfiles présents dans un projet,
    nous indiquera le type de projet et nous permettra de générer les
    fichiers terraform).
​
-   Quels sont les outils dont on aura besoin afin d'analyser un
    projet ? (Parseur, Antlr).
​
-   Est-ce que les fichiers de configuration (IaC) sont
    \"templatables\" ?
​
## Démarche à suivre :
​
1)  Nous allons identifier des projets Open sources utilisant Terraform,
​
2)  Faire une analyse manuelle d'un certain nombre de projets et
    identifier les propriétés d'intérêts (Dockerfile, langage,
    Framework, build tool ...) que les projets utilisant Terraform ont
    en commun ;
​
3)  Construire des outils scripts, parseur, pour automatiser
    l'identification des propriétés d'intérêts ;
​
4)  Déduire des règles implicites que les développeurs utilisent dans
    leur code Terraform
​
5)  Confronter ces règles avec des règles explicites existants dans des
    blogs ou sur SonarSource.
​
6)  Utiliser les règles explicites et implicites identifiées pour créer
    des outils de génération de Templates Terraform.
​
## Bibliographie :
​
-	Règles et bonnes pratiques Terraform sur SonarSource :
<https://rules.sonarsource.com/terraform>
​
-	Trouver des projets utilisant Terraform :
<https://awesomeopensource.com/projects/terraform>
​
-	Projet Terraformer sur GitHub :
<https://github.com/GoogleCloudPlatform/terraformer>
​
-	Comparaison Terraform et Ansible :
<https://k21academy.com/ansible/terraform-vs-ansible/>
​
-	Site officiel de Terraform : <https://www.terraform.io/>
