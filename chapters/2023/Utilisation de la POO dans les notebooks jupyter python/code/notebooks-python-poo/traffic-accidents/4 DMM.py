import pandas as pd  # For data handling
from time import time  # To time our operations
import multiprocessing

from gensim.models.phrases import Phrases
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#import spacy  # For preprocessing
#!pip3 install -U spacy
#!python3 -m spacy download es_core_news_md
__dir = "data/v1/doc2vec/clean/"
__file = "6_clean_lemma_dataset_propuesta2_complete"

dataset = pd.read_csv(__dir + __file + ".tsv", delimiter= "\t", quoting = 3)
del dataset["Unnamed: 0"]

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
max_epochs = 12
vec_size = 200
alpha = 0.025
min_alpha = 0.0001
try:
    unigram = [row.split() for row in dataset['text']]
    bigram = Phrases(unigram, min_count=5, progress_per=10000)
    trigram = Phrases(bigram[unigram], min_count=5, progress_per=10000)        
    trigram.save("data/v1/doc2vec/model_dmm/trigram/trigram_"+__file+".model")
    
    sentences = trigram[bigram[unigram]]
    
    print("Creando documentos para doc2vec")
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(sentences)]
    
    #DMM + Trigram    
    model = Doc2Vec(vector_size=vec_size,
                    window=5,
                    alpha=alpha, 
                    min_alpha=min_alpha,
                    min_count=4,
                    dm=1, #dm = 1 is "distribuides memory (no-order words)), if dm = 0 is "DBOW" (order words))
                    dm_mean=1,
                    epochs=15,
                    workers=cores,
                    seed=123) 
      

    print("Comenzando a contruir vocab")
    t = time()
    model.build_vocab(tagged_data)    
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    print"Comenzando a entrenar")
    t = time()
    for epoch in range(max_epochs):        
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    print('Time to train model doc2vec: {} mins'.format(round((time() - t) / 60, 2)))

    __dir_save = "data/v1/doc2vec/model_dmm/"
    model.save(__dir_save+__file+".model")    
    print("###Finalizando entrenamiento de modelo DMM###")    
    
except Exception as e:
    print('Unhandled exception:')
    print(e)

