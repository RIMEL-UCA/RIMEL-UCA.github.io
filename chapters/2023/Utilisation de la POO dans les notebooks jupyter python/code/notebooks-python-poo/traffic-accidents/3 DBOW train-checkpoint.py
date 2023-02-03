import pandas as pd  # For data handling
from time import time  # To time our operations
import multiprocessing

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging  # Setting up the loggings to monitor gensim
#logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


logger = logging.getLogger("doc2vec")
hdlr = logging.FileHandler("logs-dbow.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
__dir = "../../../data/v1/doc2vec/clean/"
#__dir = "../../../data/v1/7030/"
__file = "6_clean_lemma_dataset_propuesta1_5050"
#__file = "test30"

dataset = pd.read_csv(__dir + __file + ".tsv", delimiter= "\t", quoting = 3)
#del dataset["Unnamed: 0"]
print("dataset cargado") 

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
#max_epochs = 12
max_epochs = 1
vec_size = 200
alpha = 0.025
min_alpha = 0.0001
try:    
    logger.info("#####Comenzando a entrenar modelo DBOW######")    
    unigram = [row.split() for row in dataset['text']]
    
    #Creando documentos para doc2vec
    logger.info("Creando documentos para doc2vec")
    tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(unigram)]

    #DBOW + Unigram   
    logger.info("Creando objeto del modelo doc2vec")
    model = Doc2Vec(vector_size=vec_size,
                    window=5,
                    alpha=alpha,
                    min_alpha=min_alpha,
                    min_count=4,
                    dm=0, #dm = 1 is "distribuides memory (no-order words)), if dm = 0 is "DBOW" (order words))                
                    #epochs=15,
                    epochs=1,
                    workers=cores,
                    seed=123,
                    negative=5, hs=0)
      

    print("Comenzando a contruir vocab")
    logger.info("Comenzando a contruir vocab")
    t = time()
    model.build_vocab(tagged_data)    
    logger.info('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    print("Comenzando a entrenar DBOW")
    logger.info("Comenzando a entrenar DBOW")
    t = time()
    for epoch in range(max_epochs):        
        print('iteration {0}'.format(epoch))
        logger.info('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)        
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    print('Time to train model doc2vec: {} mins'.format(round((time() - t) / 60, 2)))    
    logger.info('Time to train model doc2vec: {} mins'.format(round((time() - t) / 60, 2)))
    
    __dir_save = "output/"
    __name = __dir_save+__file
    model.save(__name+".model")        
    logger.info("Model Saved file: "+__name+".model")
    logger.info("#####Finalizando entrenamiento de modelo DBOW######")    
    print("#####Finalizando entrenamiento de modelo DBOW######")    

except Exception as e:
    print('Unhandled exception:')
    print(e)

