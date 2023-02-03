from postal.expand import expand_address
from ast import literal_eval

import pandas as pd
import re
import math
def add_location(entities):
    #print(entities)
    loc = [ t for (t,l) in entities  if l == 'loc' ]
    if len(loc) > 4: ## Descartar tweets con más de 4 entidades de ubicaciones reconocidas
        loc = ''    
    loc = ' '.join(loc)
    if len(loc.split(' ')) < 4:   ## Descartar Tweets con menos de 4 palabras     
        loc = ''
    return loc

def address_normalization(location):    
    expansions = expand_address(location, 
                                roman_numerals=False, 
                                split_alpha_from_numeric=True, 
                                expand_numex=False,
                                languages=["es"])
    if len(expansions) > 0:
        expansions = re.sub(r'\b(\w+)( \1\b)+', r'\1', expansions[-1])
        expansions = 'Bogota '+expansions
        expansions = removeWords(expansions.upper())
    else:
        expansions = ''
    return expansions

def removeWords(text):
    stopwords = ['CON','POR','Y']
    return ' '.join([word for word in text.split() if word not in stopwords])
## Variables para importar modelos y demás
#dir_ = "../../data/v1/NER/src/prueba_bad_location/"
#dir_ = "../../data/v1/NER/src/prueba_ok_location/"
#dir_ = "../../data/v1/NER/"
dir_ = "../../data/database/output_ml/M1/NER_extractor/"

#file = 'ner_dataset_test_ok.tsv' # Dataset
file = 'entities_accident_4_server_follow_timeline_user.tsv'
## Importando Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
#del dataset['Unnamed: 0']
#del dataset['gmap']
print(dataset.shape)
dataset.head(5)
dataset = dataset[dataset['entities']!='[]']
dataset = dataset.reset_index(drop=True)
dataset.shape
#Conversión de tweets
dataset.entities = dataset.entities.apply(literal_eval)
#Conversión de entities loc a una dirección escrita
dataset['location'] = dataset.entities.apply(add_location)
dataset['location']
### Descartar ubicaciones de tweets con menos de 4 palabras en su contenido
### Descartar ubicaciones de tweets con más de 4 entidades de ubicación reconocidad.
dataset = dataset[dataset['location']!='']
dataset = dataset.reset_index(drop=True)
dataset.shape
dataset['address_normalization'] = dataset.location.apply(address_normalization)
dataset['address_normalization']
dataset = dataset[dataset['address_normalization']!='']
dataset = dataset.reset_index(drop=True)
dataset.shape
dataset.to_csv(dir_+"norm_"+file,sep='\t',index=False)
locations = []
for i in range(len(dataset)):
    loc = []
    #norm = 'Bogota '+dataset.iloc[i]['address_normalization']
    #norm = removeWords(norm.upper())
    loc += [dataset.iloc[i]['address_normalization']] + ['CO']
    locations.append(loc)
#locations
df = pd.DataFrame(locations,columns=['address','iso2'])
#df.rename(columns={0:'address',1:''}, inplace=True)
df
df.to_csv(dir_+'format_norm_'+file.split(".")[0]+".csv", index=False)
batch_geocode = pd.read_csv(dir_+'geocode_results_2021_03_10.csv')
batch_geocode = batch_geocode[['best_lat','best_long']]
batch_geocode
dataset['gmap'] = ''
for i in range(len(dataset)):
    if(not math.isnan(batch_geocode.iloc[i]['best_lat'])):
        dataset.at[i,'gmap'] = "{'lat': "+str(batch_geocode.iloc[i]['best_lat'])+", 'lng': "+str(batch_geocode.iloc[i]['best_long'])+"}"
    else:
        dataset.drop(index=dataset.index[[i]])
geocoding = dataset.copy()
geocoding
geocoding = geocoding[dataset['gmap']!='']
geocoding = geocoding.reset_index(drop=True)
geocoding
geocoding.to_csv(dir_+"ner_dataset_norm_geocoding.tsv",sep='\t')

