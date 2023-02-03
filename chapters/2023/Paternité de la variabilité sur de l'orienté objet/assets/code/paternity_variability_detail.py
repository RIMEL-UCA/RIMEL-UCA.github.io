import json
import os
import sys
import subprocess


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


if len(sys.argv) == 1:
    print("Il manque le lien vers le git")
    exit(1)

project_name = sys.argv[1].split('/')[-1].split(".")[0]
file_name = "./results/"+ project_name + "_paternity_result_detail.txt"
result_file = open(file_name, "w")
if not os.path.exists(file_name):
    print("Result file has not been created. Exiting...")
    exit(1)
print("##########################          FILE OPEN")

if len(sys.argv) == 3:
    clone = subprocess.Popen("git clone --depth 1 --branch " + sys.argv[2] + " " + sys.argv[1], shell=True )
    clone.wait()
else:
    clone = subprocess.Popen("git clone " + sys.argv[1], shell=True )
    clone.wait()
os.chdir(project_name)

vars = dict()

with open("../db.json") as symfinder_output:
    output = json.loads(symfinder_output.read())

    variants_class = filter(lambda node: "VARIANT" in node["types"] or "VP" in node["types"], output["nodes"])
    for variant_class in variants_class:
        variant_name = variant_class["name"]
        print("Variant: " + variant_name)
        path = find(getClassName(variant_name) + ".java", os.getcwd())
        if path is not None:
            authors = os.popen("git blame -p " + path[len(os.getcwd()) + 1:] + " | grep '^author-mail' | grep -Eio '\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}\\b' | sort -u").read()
            authors = authors.split("\n")
            authors.pop()
            for author in authors:
                lines_modified = int(os.popen("git blame --line-porcelain " + path[len(os.getcwd()) + 1:] + " | grep 'author-mail <" + author + ">' | wc -l").read())
                if variant_name not in vars:
                    vars[variant_name] = dict()
                vars[variant_name][author] = lines_modified
            lines_count = sum(list(vars[variant_name].values()))
            for author, lines_modified in vars[variant_name].items():
                vars[variant_name][author] = lines_modified / lines_count

result_file.write(json.dumps(vars))
result_file.close()
