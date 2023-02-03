import gmplot

from ast import literal_eval
import sys

import pandas as pd

sys.path.insert(0, '../../')

from classes.tweet2accident.enviroments import Global
## Variables para importar modelos y demás
dir_ = "../../data/v1/NER/"
#dir_ = "../../data/v1/NER/src/prueba_bad_location/"
#dir_ = "../../data/v1/NER/src/prueba_ok_location/"

file = 'ner_dataset_norm_geocoding.tsv' # Dataset
#file = 'ner_dataset_test_ok_norm_geocoding.tsv' # Dataset
#file = 'ner_dataset_test_bad_norm_geocoding.tsv' # Dataset

api_key = Global()
gmap = gmplot.GoogleMapPlotter(4.626383,-74.105074, 13, apikey=api_key.getMapsKey(), title="Geo Twitter")
## Importando Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
dataset.entities = dataset.entities.apply(literal_eval)
#dataset = dataset[dataset['location'] != 'Ningún resultado encontrado']
#dataset.location = dataset.location.apply(literal_eval)
dataset = dataset[dataset['gmap'] != 'Ningún resultado encontrado']
dataset.gmap = dataset.gmap.apply(literal_eval)
del dataset['Unnamed: 0']
print(dataset.shape)
dataset.head(5)
dataset.info()
#dataset['location']
type(dataset['gmap'][0])
def info_window_fn(row):
    link = "https://twitter.com/i/status/"+str(row['id_tweet'])
    info = "<b>Tweet: </b>%s<br><b>Created at: </b>%s<br><b>Link: </b><a href='%s' target='_blank'>%s</a><br><b>Entities: </b>%s<br><b>Coordinates: </b>%s<br><b>Address: </b>%s"%(
        row['text'], 
        row['created_at'], 
        link,
        link,
        row['entities'], 
        #row['location']
        row['gmap'],
        row['address_normalization']
    )
    return info

for i in range(len(dataset)):
    info_window = info_window_fn(dataset.iloc[i])    
    #gmap.marker(dataset.iloc[i]['location']['lat'], dataset.iloc[i]['location']['lng'], info_window=info_window)
    gmap.marker(dataset.iloc[i]['gmap']['lat'], dataset.iloc[i]['gmap']['lng'], info_window=info_window)
                                            
gmap.draw(dir_+'ner_dataset_norm_geocoding_maps.html')
dataset.iloc[1]

