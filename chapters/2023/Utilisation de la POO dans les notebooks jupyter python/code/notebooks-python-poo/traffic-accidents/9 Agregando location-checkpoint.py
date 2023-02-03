from ast import literal_eval
import sys

import pandas as pd
## Variables para importar modelos y demás
dir_ = "../../data/v1/NER/"

file = 'ner_dataset_entities.tsv' # Dataset
## Importando Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
dataset.entities = dataset.entities.apply(literal_eval)
del dataset['Unnamed: 0']
print(dataset.shape)
dataset.head(5)
def add_location(entities):
    loc = [ t for (t,l) in entities  if l == 'loc' ]
    loc = ' '.join(loc)
    return loc
dataset['location'] = dataset.entities.apply(add_location)
dataset['location']
dataset.to_csv(dir_+"ner_dataset_location.tsv",sep='\t')

