#!/bin/bash

if [ ! -f "$1" ]; then
    echo "Le fichier spécifié n'existe pas."
    exit 1
fi

while IFS= read -r nom || [ -n "$nom" ]; do
    echo "Analyse du projet $nom ..."
    if [ -n "$nom" ]; then
        ./analyze-data-one-project.sh "$nom"
        rm -rf ./outputs
    fi
done < "$1"
