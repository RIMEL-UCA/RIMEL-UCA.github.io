#!/usr/bin/env bash

docker run -v $(pwd)/images:/app/images -v $(pwd)/projects:/app/projects github_action_parser:latest python3 main.py $@

cd images

rm *.dot.png

for file in $(ls *.dot)
do
    echo "Generating image for $file"
    dot -Tpng $file >${file%.*}.png
done

cd ..