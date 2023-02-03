import os
import re
import json

from git import *
from difflib import SequenceMatcher

repo = Repo("")
path = "."

# r=root, d=directories, f = files

similary_ratio_threshold = 0.8

def is_similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > similary_ratio_threshold

total_map = {}

for r, d, files in os.walk(path):
    files = [ file for file in files if file.endswith( ('.c','.h', '.cpp', '.hpp') ) ]    
    #if(r.find(".git") > 0):
    #    continue
    for file in files:
        filepath = os.path.join(r, file)

        line_per_commit = []
        
        if(file == "main.py"):
            continue

        for commit, lines in repo.blame("HEAD", filepath):            
            for line in lines:
                line_per_commit.append((line, commit))

        nested_if_def_component = []

        for i in range(len(line_per_commit)):
            line = str(line_per_commit[i][0])
            commit = line_per_commit[i][1]

            splitted_line = line.split(" ")

            if(splitted_line[0] == "#ifdef"):              
                nested_if_def_component.append(splitted_line[1]) 

            if(splitted_line[0] == "#endif"):
                if(len(nested_if_def_component) == 0):
                    continue

                nested_if_def_component.pop()

            if(len(nested_if_def_component) == 0):
                continue

            joined_line_components = "/".join(nested_if_def_component)

            if joined_line_components not in total_map:
                total_map[joined_line_components] = {}
            
            if commit.author.name not in total_map[joined_line_components]:
                total_map[joined_line_components][commit.author.name] = 0
            

            total_map[joined_line_components][commit.author.name] = total_map[joined_line_components][commit.author.name] + 1
            #print(str(i) + " " + str(line_per_commit[i]))

# Merge similar authors
for component in total_map:
    authors = list(total_map[component].keys())

    for i in range(0, len(authors)):
        for j in range(i + 1, len(authors)):
            author1 = authors[i]
            author2 = authors[j]

            if(is_similar(author1.lower(), author2.lower())):
                total_map[component][author2] = total_map[component][author2] + total_map[component][author1]
                del total_map[component][author1]



top_contributors = {}

for component in total_map:
    top_contributors[component] = [(k, v) for k, v in total_map[component].items()]


# Find top 80% contributors
for component in top_contributors:
    authors = top_contributors[component]
    
    top_contributors[component].sort(key=lambda x: x[1], reverse=True)

    total_contributions = 0

    for i in range(0, len(authors)):
        total_contributions = total_contributions + top_contributors[component][i][1]
    

    i = 0

    threshold = 0.8

    threshold_top_contributors = []
    
    for i in range(len(top_contributors[component])):
        if(threshold < 0.0):
            break
        
        threshold_top_contributors.append(top_contributors[component][i])
        threshold = threshold - top_contributors[component][i][1] / total_contributions
    
    top_contributors[component] = threshold_top_contributors



json_out = json.dumps(top_contributors)

f = open("out.json", "w")
f.write(json_out)
f.close()