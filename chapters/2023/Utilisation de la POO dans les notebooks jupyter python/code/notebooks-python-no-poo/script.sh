#!/bin/bash

# execute pylint on all python files in the current directory
# iterate over all subdirectories
for dir in $(find . -type d); do
    # run pylint on folder and save only the score as a text file called score.txt
    pylint $dir | grep "Your code has been rated at" > $dir/score.txt
     
done