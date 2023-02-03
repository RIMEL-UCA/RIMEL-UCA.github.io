import json
import os
import sys
import subprocess

ent_attrs = ["ABSTRACT", "INNER", "VP", "VARIANT"] # "METHOD_LEVEL_VP" -> trop difficile, "OUT_OF_SCOPE", "HOTSPOT" -> jsp c quoi

class VarAuthors:
    def __init__(self, var_name):
        self.VariabilityType = var_name
        self.Paternity = dict()
        
class VarAuthorsProcessing:
    def __init__(self, var_name):
        self.var_type = var_name
        self.paternity = dict()
        self.lines_count = 0


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
file_name = "./results/"+ project_name + "_paternity_result_general.txt"
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

vars_authors_processing = []
for var in ent_attrs:
    vars_authors_processing.append(VarAuthorsProcessing(var))

with open("../db.json") as symfinder_output:
    output = json.loads(symfinder_output.read())
    
    #variants_class = filter(lambda node: all(x in node["types"] for x in ent_attrs), output["nodes"])
    #print(str(len(list(variants_class))) + " variants class found")
    for variant_class in output["nodes"]:
        variant_name = variant_class["name"]
        print("Variant: " + variant_name)
        path = find(getClassName(variant_name) + ".java", os.getcwd())
        print("Path: " + path)
        
        for type in variant_class["types"]:
            var_authors = next(filter(lambda attr: attr.var_type == type, vars_authors_processing), None)
            if var_authors != None:
                authors = os.popen("git blame -p " + path[len(os.getcwd()) + 1:] + " | grep '^author-mail' | grep -Eio '\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}\\b' | sort -u").read()
                authors = authors.split("\n")
                authors.pop()
                print("Authors: " + ", ".join(authors))
                lines_count = 0
                #authors_lines_modified = dict()
                for author in authors:
                    lines_modified = int(os.popen("git blame --line-porcelain " + path[len(os.getcwd()) + 1:] + " | grep 'author-mail <" + author + ">' | wc -l").read())
                    #authors_lines_modified[author] = lines_modified
                    lines_count += lines_modified
                    if author not in var_authors.paternity:
                        var_authors.paternity[author] = 0
                    var_authors.paternity[author] += lines_modified
                var_authors.lines_count += lines_count
                    

var_authors = list()
for var_author_processing in vars_authors_processing:
    var_author = VarAuthors(var_author_processing.var_type)
    var_authors.append(var_author)
    for author, line_modified in var_author_processing.paternity.items():
        var_author.Paternity[author] = line_modified / var_author_processing.lines_count
    
result_file.write(json.dumps(var_authors, default=vars))
result_file.close()