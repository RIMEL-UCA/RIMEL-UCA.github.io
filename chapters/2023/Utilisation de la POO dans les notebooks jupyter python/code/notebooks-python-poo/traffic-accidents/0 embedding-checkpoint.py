import sys
sys.path.insert(0, '../../../')
import spacy

from classes.tweet2accident.doc2vec import Embedding
from gensim.models.doc2vec import Doc2Vec
directory = "../../../data/v1/doc2vec/"
file = "6_clean_lemma_dataset_propuesta1_5050"

model = Embedding(type_transform='dbow',directory=directory, file=file)
model
dbow = Doc2Vec.load(directory+"model_dbow/"+file+".model")
try:
    print(dbow.wv['19'])
except KeyError as e:
    print(e)


spacy_model = "../../../data/v1/NER/spacy_model_complete/"
nlp = spacy.load(spacy_model)
nlp = spacy.load("es_core_news_lg",disabled=['ner','parser'])
token_id = nlp.vocab.strings["calle"]
token_id
nlp.vocab.vectors.shape
nlp.vocab.vectors.resize((500000,200))
token_id = nlp.vocab.strings["fuck"]
token_vector = nlp.vocab.vectors[token_id]
token_vector.shape
token_id = nlp.vocab.strings["calle"]
token_vector = nlp.vocab.vectors[token_id]
token_vector
import spacy  # For preprocessing
from spacy import displacy
nlp = spacy.load("es_core_news_lg")
doc = nlp("movilidad bogota acueducto trancon accidente llevó 3 horas en el carro bajando de la calera y muchos Buses escolares con niños pequeños de los colegios, nada que quitan el camión del acueducto que se accidentó en la circunvalar con 85, TERRIBLE!!")

for token in doc:
    print(token.text, token.ent_iob_, token.ent_type_.lower())
    
#displacy.render(doc, style="ent")
doc = nlp("89c transito Bogota y en spanish")

for token in doc:
    print(token.text, token.ent_iob_+"-"+token.ent_type_.lower())
    

