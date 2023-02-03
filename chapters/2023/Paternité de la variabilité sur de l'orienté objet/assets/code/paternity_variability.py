import json
import os
import sys
import subprocess

class VpVarsAuthors:
    def __init__(self):
        self.vp = ""
        self.vars = list()
        self.authors = dict()


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
file_name = "./results/"+ project_name + "_paternity_result.txt"
result_file = open(file_name, "w")
if not os.path.exists(file_name):
    print("Result file has not been created. Exiting...")
    exit(1)
print("##########################          FILE OPEN")

#result_file.write("RESULT FOR PROJECT " + project_name + "\n")
if len(sys.argv) == 3:
	clone = subprocess.Popen("git clone --depth 1 --branch " + sys.argv[2] + " " + sys.argv[1], shell=True )
	clone.wait()
else:
	clone = subprocess.Popen("git clone " + sys.argv[1], shell=True )
	clone.wait()
os.chdir(project_name)

parents = dict()
vps_vars = dict()

with open("../db.json") as symfinder_output:
    output = json.loads(symfinder_output.read())
    vps = map(lambda node: node["name"], filter(lambda node: "VP" in node["types"], output["nodes"]))
    for vp in vps:
        vps_vars[vp] = list()
    
    variants_class = filter(lambda node: "VARIANT" in node["types"] or "VP" in node["types"], output["nodes"])
    #print(str(len(list(variants_class))) + " variants class found")
    for variant_class in variants_class:
        variant_name = variant_class["name"]
        print("Variant: " + variant_name)

        #result_file.write("Variant: " + variant_name + "\n")
        path = find(getClassName(variant_name) + ".java", os.getcwd())
        # print("Path: " + path)
        print("Path: ", path)
        if path != None:
            link = list(
                filter(lambda node: node["target"] == variant_name and node["type"] in ["IMPLEMENTS", "EXTENDS"],
                       output["alllinks"]))
            if len(link) > 0:
                vps_vars[link[0]["source"]].append(variant_name)
            authors = os.popen("git blame -p " + path[len(os.getcwd()) + 1:] + " | grep '^author-mail' | grep -Eio '\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}\\b' | sort -u").read()
            authors = authors.split("\n")
            authors.pop()
            print("Authors: " + ", ".join(authors))
            #result_file.write("Authors: " + ", ".join(authors) + "\n")
            for author in authors:
                lines_modified = int(os.popen("git blame --line-porcelain " + path[len(os.getcwd()) + 1:] + " | grep 'author-mail <" + author + ">' | wc -l").read())
                if variant_name not in parents:
                    parents[variant_name] = dict()
                parents[variant_name][author] = lines_modified
            lines_count = sum(list(parents[variant_name].values()))
            for author, lines_modified in parents[variant_name].items():
                parents[variant_name][author] = lines_modified / lines_count
                print(lines_modified, "/", lines_count)
                #result_file.write(str(lines_modified) + " / " + str(lines_count) + "\n")
        else:
            parents[variant_name] = dict()
    
    print("Result: ")
    print(parents)
    #result_file.write("Result: " + "\n")
    #result_file.write(json.dumps(parents) + "\n")

vps_vars_authors = list()
for vp, vps in vps_vars.items():
    vp_vps_authors = VpVarsAuthors()
    vp_vps_authors.vp = vp
    for vp in vps:
        vp_vps_authors.vars.append(vp)
    for vp_or_v in vps + [vp]:
        authors_modify = parents[vp_or_v]
        for author, line_modified_per in authors_modify.items():
            if author not in vp_vps_authors.authors:
                vp_vps_authors.authors[author] = line_modified_per
            else:
                vp_vps_authors.authors[author] += line_modified_per
    for author, lin_sum in vp_vps_authors.authors.items():
        vp_vps_authors.authors[author] = lin_sum / (len(vp_vps_authors.vars) + 1)
    vps_vars_authors.append(vp_vps_authors)  

# result_file.write("VP Variants Authors\n")
result_file.write(json.dumps(vps_vars_authors, default=vars))
result_file.close()