from ast import literal_eval
import googlemaps
import sys

import pandas as pd

sys.path.insert(0, '../../')

from classes.tweet2accident.enviroments import Global
## Variables para importar modelos y demás
dir_ = "../../data/v1/NER/"

file = 'ner_dataset_entities.tsv' # Dataset
## Importando Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
dataset.entities = dataset.entities.apply(literal_eval)
del dataset['Unnamed: 0']
print(dataset.shape)
dataset.head(5)
#i = 46
i = 46
print(dataset['text'][i])

loc = [ t for (t,l) in dataset.iloc[i]['entities']  if l == 'loc' ]
loc = ' '.join(loc)
print(loc)
gmaps = googlemaps.Client(key='AIzaSyAuHkP89-zjfiZCWT4MXNDD7fdn2F7Pqpk')
# Geocoding an address
#geocode_result = gmaps.geocode("%s, Bogotá, Colombia" % loc)
st = "Bogotá %s" % loc
print(st)
geocode_result = gmaps.geocode(st, region='co')
if len(geocode_result) > 0:
    geocode = geocode_result[0]['geometry']['location']    
    print(geocode)
else:
    print("Ningún resultado encontrado")
geocode_result[0]['geometry']['location']
geo = tuple(geocode_result[0]['geometry']['location'].values())
print(geo)
dataset['location'] = ''
for i in range(len(dataset)):
    loc = [ t for (t,l) in dataset.iloc[i]['entities']  if l == 'loc' ]
    loc = ' '.join(loc)
    #loc = "%s, Bogotá, Colombia" % loc    
    loc = "Bogotá %s" % loc    
    geocode_result = gmaps.geocode(loc)
    if len(geocode_result) > 0:
        geocode = geocode_result[0]['geometry']['location']        
    else:        
        geocode = "Ningún resultado encontrado"    
    dataset.at[i,'location'] = geocode
    if i%100 == 0:
        print(i)
dataset[::-20]
dataset['location']
dataset.to_csv(dir_+"ner_dataset_geocoding_v2.tsv",sep='\t')

