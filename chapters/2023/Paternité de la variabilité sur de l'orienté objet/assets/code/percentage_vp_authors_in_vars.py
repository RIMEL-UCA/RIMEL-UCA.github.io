import json
import os
import sys
from itertools import chain


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def getClassName(path):
    decomposed_path = path.split(".")
    for folder in decomposed_path:
        if folder[0].isupper():
            return folder
    return decomposed_path[-1]


def get_authors(variant_name):
    path = find(getClassName(variant_name) + ".java", os.getcwd())
    authors = os.popen("git blame -p " + path[len(os.getcwd()) + 1:] + " | grep '^author-mail' | grep -Eio '\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}\\b' | sort -u").read()
    authors = authors.split("\n")
    authors.pop()
    return authors


def lines_modified(variant_name, author):
    path = find(getClassName(variant_name) + ".java", os.getcwd())
    return int(os.popen("git blame --line-porcelain " + path[len(os.getcwd()) + 1:] + " | grep 'author-mail <" + author + ">' | wc -l").read())


if len(sys.argv) == 2:
    print("Il faut le chemin vers le fichier de résultat normal + dossier du projet")
    exit(0)

variability = json.loads(open("db.json").read())
result = list(filter(lambda vp_vars_aut: len(vp_vars_aut["vars"]) > 0, json.loads(open(sys.argv[1]).read())))
cpw = os.getcwd()
os.chdir(sys.argv[2])

original_authors_found = 0
new_authors_count = 0
original_authors_contribution = 0

for vp_vars_aut in result:
    vp = vp_vars_aut["vp"]
    vars = vp_vars_aut["vars"]
    if len(vars) == 0:
        continue
    vp_authors = get_authors(vp)
    vars_authors = set(chain(*map(lambda var: get_authors(var), vars)))
    original_authors_found += all(x in vars_authors for x in vp_authors)
    new_authors_count += not all(x in vp_authors for x in vars_authors)

    contributions_vp_authors = 0
    for variant in vars:
        contributions_vp_authors += sum(lines_modified(variant, aut) for aut in vp_authors) / sum(lines_modified(variant, aut) for aut in vars_authors)
    contributions_vp_authors /= len(vars)
    original_authors_contribution += contributions_vp_authors

per_authors_found = original_authors_found/len(result) * 100
per_new_authors_found = new_authors_count/len(result) * 100
per_original_authors_contribution = original_authors_contribution/len(result) * 100
print("Pourcentage d'auteurs d'un VP qui ont contribués à ses Variants:", round(per_authors_found, 2), "%")
print("Pourcentage de Groupe de Variants avec au moins un contributeur sur un Variant mais pas sur le VP:", round(per_new_authors_found, 2), "%")
print("Pourcentage de la moyenne de la contribution des auteurs de VP sur les Variants:", round(per_original_authors_contribution, 2), "%")

# Exemple: python3 percentage_vp_authors_in_vars.py results/FizzBuzzEnterpriseEdition_paternity_result.txt FizzBuzzEnterpriseEdition/
