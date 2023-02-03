import pandas as pd
import re
from time import time  # To time our operations
from nltk.stem import SnowballStemmer
import multiprocessing

import spacy  # For preprocessing
#!pip3 install -U spacy
#!python3 -m spacy download es_core_news_md
stemmer = SnowballStemmer('spanish')

def cleaning_stem_stopwords(doc):
    # Stemming and removes stopwords    
    #txt = [token.lemma_ for token in doc if not token.is_stop]    
    txt = [stemmer.stem(token.text) for token in doc if not token.is_stop]    
    if len(txt) > 2:
        return ' '.join(txt)
    
def cleaning_lemma_stopwords(doc):
    # Lemma and removes stopwords        
    txt = [(token.lemma_ if token.text != 'calle' else token.text) for token in doc if not token.is_stop]        
    if len(txt) > 2:
        return ' '.join(txt)

def cleaning_stopwords(doc):
    # Only removing stopwords        
    txt = [token.text for token in doc if not token.is_stop]    
    if len(txt) > 2:
        return ' '.join(txt)
    
def cleaning_special_chars(doc):
    #All characteres, without @, urls, # and numbers.        
    txt = [token.text for token in doc]    
    if len(txt) > 2:
        return ' '.join(txt)

def cleaning_stem(doc):
    #Stem without removes stopwords
    txt = [stemmer.stem(token.text) for token in doc]    
    if len(txt) > 2:
        return ' '.join(txt)
    
def cleaning_lemma(doc):
    #Lemma without removes stopwords
    txt = [(token.lemma_ if token.text != 'calle' else token.text) for token in doc]    
    if len(txt) > 2:
        return ' '.join(txt)
cores = multiprocessing.cpu_count()

proposals = ['dataset_propuesta1_5050.tsv', 'dataset_propuesta2_complete.tsv']
dir_data = "../data/v1/doc2vec/"
dir_output = 'output/'
nlp = spacy.load("es_core_news_md",disabled=['ner','parser']) # disabling Named Entity Recognition for speed
nlp.vocab["rt"].is_stop = True #Add RT to Stopwords
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']    
        
    print('## 1. Stem and stopwords')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_stem_stopwords(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_stem_stopwords = pd.DataFrame({'text': txt})
    df_clean_stem_stopwords = df_clean_stem_stopwords.dropna()
    filename = dir_output+"1_clean_stem_stopwords_"+proposal
    df_clean_stem_stopwords.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']
    
    print('## 2. Lemma and stopwords')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_lemma_stopwords(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_lemma_stopwords = pd.DataFrame({'text': txt})
    df_clean_lemma_stopwords = df_clean_lemma_stopwords.dropna()
    filename = dir_output+"2_clean_lemma_stopwords_"+proposal
    df_clean_lemma_stopwords.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']
    
    print('## 3. Only stopwords')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_stopwords(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]    
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_stopwords = pd.DataFrame({'text': txt})
    df_clean_stopwords = df_clean_stopwords.dropna()
    filename = dir_output+"3_clean_stopwords_"+proposal
    df_clean_stopwords.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']
    
    print('## 4. Only special characters')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_special_chars(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_special_chars = pd.DataFrame({'text': txt})
    df_clean_special_chars = df_clean_special_chars.dropna()
    filename = dir_output+"4_clean_special_chars_"+proposal
    df_clean_special_chars.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']
    
    print('## 5. Stem without removes stopwords')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_stem(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_stem = pd.DataFrame({'text': txt})
    df_clean_stem = df_clean_stem.dropna()
    filename = dir_output+"5_clean_stem_"+proposal
    df_clean_stem.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
for proposal in proposals:
    print('# Starting...')
    print('Reading file '+proposal)
    dataset = pd.read_csv(dir_data+proposal, delimiter = "\t", quoting = 3)
    del dataset['Unnamed: 0']
    
    print('## 6. Lemma without removes stopwords')
    #Clean @, url, special characters,
    brief_cleaning = (re.sub("(@[A-Za-z0-9]+)|((?<=[A-Za-z])(?=[A-Z][a-z]))|([^A-Za-zäÄëËïÏöÖüÜáéíóúáéíóúÁÉÍÓÚÂÊÎÔÛâêîôûàèìòùÀÈÌÒÙñÑ])|(\w+:\/\/\S+)",
                             ' ', str(row)).lower() for row in dataset['text'])
    
    t = time()    
    txt = [cleaning_lemma(doc) for doc in nlp.pipe(brief_cleaning, n_threads=cores)]
    #print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    print('-- -- Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    
    df_clean_lemma = pd.DataFrame({'text': txt})
    df_clean_lemma = df_clean_lemma.dropna()
    filename = dir_output+"6_clean_lemma_"+proposal
    df_clean_lemma.to_csv(filename,sep='\t')    
    print('-- -- Se genero archivo tsv: '+filename)
print('### Finish clean dataset '+proposal+' ####')
print('############################################################')
