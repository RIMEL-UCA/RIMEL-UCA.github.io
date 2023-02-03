import pandas as pd
from ast import literal_eval

dir_ = "../../data/v1/NER/"
file = 'ner_dataset_norm_geocoding.tsv' # Dataset
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
dataset.gmap = dataset.gmap.apply(literal_eval)
del dataset['Unnamed: 0']
print(dataset.shape)
dataset
dataset['lat'] = ''
dataset['lon'] = ''
for i in range(len(dataset)):
    dataset.at[i,'lat'] = dataset.iloc[i]['gmap']['lat']
    dataset.at[i,'lon'] = dataset.iloc[i]['gmap']['lng']
dataset.head()
dataset.info()
dataset.to_csv(dir_+"ner_dataset_norm_lat_lon.tsv",sep='\t')

