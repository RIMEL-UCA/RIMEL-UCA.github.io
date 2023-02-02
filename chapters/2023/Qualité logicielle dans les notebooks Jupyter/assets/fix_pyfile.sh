#! /bin/bash
for file in `ls $1 `
do
    python3 fix_script.py $1/$file
done
