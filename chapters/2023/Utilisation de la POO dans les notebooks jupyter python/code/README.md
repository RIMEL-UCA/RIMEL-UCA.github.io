# github-repository-scrapper

## Steps of process :

### script clone.py
* We get all repositories as dictionary file : key (repository-name) and value (lignes of code) we do this to have a lighter file instead of having large files

### Exctracting informations

Once we have created the json file, we run the process over the extracted repositories :

* we get the oop keyword usage using regular expressions (class definition, inhertance...)

* repos_stars.py : script to get repositories stars and forks and plot the results to compare repos that uses oop and the one that don't

* Repository.py : script to convert the jupyter notebook to python file to get pylint score

* plots.py : scripts to draw different plots of the different matrics

* script.sh: inside notebooks-python iterate over all repositories and run the pylint command to get the score and save it in txt file.


