#!/bin/bash

# Chemins vers les dossiers contenant les scripts Python
folders=("../search-consumers" "../search-producers" "../search-microservices")

project_name=$(echo "$1" | cut -d "/" -f 2)

output_directories=()

rm -rf ./outputs/*
rm -rf ./metrics/*

for folder in "${folders[@]}"
do
    echo "Recherche de scripts Python dans le dossier : $folder"
    
    if [ -d "$folder" ]; then
        for script in "$folder"/*.py
        do
            echo "Exécution de $script ..."
            python3 "$script" -p $1
        done
    fi

done

absolute_path_outputs=$(readlink -f "./outputs")
absolute_path_project=$(readlink -f "./projects/$project_name")

python3 ../search-topic/main.py "$absolute_path_outputs" "$absolute_path_project" 

python3 analyze-project-data.py $1

python3 generate-metrics-ui.py $1
