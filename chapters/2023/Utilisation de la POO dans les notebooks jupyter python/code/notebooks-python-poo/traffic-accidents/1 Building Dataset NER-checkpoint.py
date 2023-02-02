import pandas as pd
import spacy  # For preprocessing
import re

import sys
sys.path.insert(0, '../../../')

from classes.wordsegmentation import WordSegmentation
nlp = spacy.load("es_core_news_lg") # disabling Named Entity Recognition for speed
file = 'ner_dataset_shuffle.tsv'
dir_ = "../../../data/v1/NER/"
dataset = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3)
del dataset['Unnamed: 0']
dataset = dataset['text']
dataset[0]
dataset.shape
#Seleccionando 50
#df = dataset.sample(n=50)
df = dataset
seg = WordSegmentation(dir_+'spanish_count_1w_small_v2_twitter.txt')
def word_segmentation(pattern, text):
    search = re.search(pattern,text)
    while search:    
        s = search.start()
        e = search.end()                
        text = text[:s] + ' ' + ' '.join(seg.segment(text[s+1:e])) +' '+ text[e:]        
        search = re.search(pattern,text)        
    return text

def clean_fn(doc):    
    text = doc
    pattern = "(@[A-Za-z0-9äÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ_]+)"
    text = word_segmentation(pattern,text)
    pattern = "(#[A-Za-z0-9äÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ_]+)"
    text = word_segmentation(pattern,text)
    return text
    
def preText(text):        
    pre = re.sub("&[A-Za-z]+;", ' ', text) #Eliminar códigos ASCII
    pre = re.sub("(\w+:\/\/\S+)",' ',pre) #Eliminar links http y https
    pre = re.sub("([^A-Za-z0-9äÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ,;.:*\-\[\]¿?¡!\"\"()_'/])",' ',pre) #Eliminar caracteres especiales como emoticones, exceptuando los signos de puntuación y tildes.
    pre = re.sub(r'([;.:\-\[\]¿?¡!#\"()]){3,}',r'\1\1',pre) #Si repite un caracters especial más de 3 veces ej. """"
    pre = re.sub(r'([a-zA-Z])\1{2,}',r'\1\1',pre) #Si repite una letra más de dos veces las reduce a dos repeticiones goool => gool        
    pre = re.sub(r'(\s){2,}',r' ',pre) #Eliminar espacios seguidos              
    return pre.strip()
brief_cleaning = (clean_fn(str(row)) for row in df)
txt = [preText(doc.text) for doc in nlp.pipe(brief_cleaning, batch_size=50, n_threads=4)]
txt
new = pd.DataFrame(df)
new['pre'] = txt
new.to_csv('50_tweets_test.tsv',sep='\t')
#Generando txt y ann
i=1
for tweet in txt:
    name = dir_+'brat/'+'tweet-'+str(i)+'.txt'
    #print(name)  
    with open(name, "w") as file:
        file.write(tweet)
    name = dir_+'brat/'+'tweet-'+str(i)+'.ann'
    with open(name, "w") as file:
        file.write('')
    i+=1
txt[18]

