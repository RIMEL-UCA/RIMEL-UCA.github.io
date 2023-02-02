import pandas as pd
import sys
sys.path.insert(0, '../../../')

from classes.tweet2accident.ner_preprocessing import NerPreprocessing
from classes.tweet2accident.ner_extractor import NerExtractor
## Variables para importar modelos y demás
dir_ = "../../../data/v1/NER/"
dir_dataset = "../../../data/database/output_ml/M1/"

#file = 'ner_dataset.tsv' # Dataset
file = 'accident_4_server_follow_timeline_user.tsv' # Dataset

spacy_model = dir_+"spacy_model_complete/" #Spacy model entrenado previamente
corpus_segmentation = dir_+'spanish_count_1w_small_v2_twitter.txt' # Corpus para entrenar el wordsemgentation
## Importando Dataset
dataset = pd.read_csv(dir_dataset+file, delimiter = "\t", quoting = 3)
del dataset['Unnamed: 0']
print(dataset.shape)
dataset.head(5)
dataset.shape
ner_preprocessing = NerPreprocessing(spacy_model=spacy_model, corpus_segmentation=corpus_segmentation,njobs=4)
txt = ner_preprocessing.transform(dataset['text'])
dataset['clean'] = txt
dataset[['text','clean']].head(5)
dataset.iloc[1]['text']
ner_extractor = NerExtractor(spacy_model=spacy_model, njobs=4)
txt = ner_extractor.transform(dataset['clean'])
dataset['entities'] = txt 
dataset[['text','entities']].head(5)
dataset.shape
dataset.to_csv(dir_dataset+"NER_extractor/entities_"+file,sep='\t')
i = 46
print(dataset.iloc[i]['text'])
print(dataset.iloc[i]['entities'])
print(type(dataset.iloc[i]['entities']))
print(dataset.iloc[i]['entities'][0])
print(type(dataset.iloc[i]['entities'][0]))
ent = [ t for (t,l) in dataset.iloc[i]['entities']  if l == 'loc' ]
ent
ent = [ t for (t,l) in dataset.iloc[i]['entities']  if l == 'time' ]
ent
dataset.to_csv(dir_+"ner_dataset_entities.tsv",sep='\t')
