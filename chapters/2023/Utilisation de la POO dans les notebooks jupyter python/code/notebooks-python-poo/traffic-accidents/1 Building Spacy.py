import pandas as pd
import spacy

from sklearn.model_selection import train_test_split
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        #agg_func = lambda s: " ".join(s["Word"].values.tolist())
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            #s = self.grouped["Sentence: {}".format(self.n_sent)]
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None
spacy_model = "../../../data/v1/NER/spacy_model_complete/"
nlp = spacy.load(spacy_model)
file = 'ner-crf-test-data.tsv'
dir_ = "../../../data/v1/NER/test/"
data = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3, names=['Sentence #','Word','POS','Tag'])
getter = SentenceGetter(data)
sentences = getter.sentences
sentences[0]
X = [" ".join([w[0] for w in s]) for s in sentences]
y = [[w[2] for w in s] for s in sentences]

print(X[1])
print(y[1])
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
def predict_fn(sentence):
    labels = []
    words = []
    doc2 = nlp(sentence)
    for token in doc2:
        l = token.ent_type_ if token.ent_type_ != '' else 'O'
        w = token.text
        labels.append(l)
        words.append(w)
    return words,labels

def predict_all_fn(test):
    pred = []
    words = []
    for s in test:
        w, p = predict_fn(s)
        pred.append(p)
        words.append(w)
    return words, pred
        
tokens, y_pred = predict_all_fn(X)
#tokens
#print("F1-score: {:.1%}".format(f1_score(y_test, y_pred)))
print("F1-score: {:.6%}".format(f1_score(y, y_pred)))
#print(classification_report(y_test, y_pred))
print(classification_report(y, y_pred))
cont = 0
for i in range(len(y_pred)-1):
    if len(y_pred[i]) != len(y[i]):
        print(i)
        del y_pred[i]
        del y[i]
from sklearn_crfsuite import metrics

labels = ['B-loc', 'I-loc', 'B-time', 'I-time']
# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y, y_pred, labels=sorted_labels, digits=4
))
print("F1-score: {:.4%}".format(metrics.flat_f1_score(y, y_pred, average='weighted', labels=labels)))
print("Accuracy: "+str(round(metrics.flat_accuracy_score(y, y_pred),6)))

print("F1-Micro: "+str(round(metrics.flat_f1_score(y, y_pred, average='micro', labels=labels),6)))
print("Recall-Micro: "+str(round(metrics.flat_recall_score(y, y_pred, average='micro', labels=labels),6)))
print("Precision-Micro: "+str(round(metrics.flat_precision_score(y, y_pred, average='micro', labels=labels),6)))

print("F1-Macro: "+str(round(metrics.flat_f1_score(y, y_pred, average='macro', labels=labels),6)))
print("Recall-Macro: "+str(round(metrics.flat_recall_score(y, y_pred, average='macro', labels=labels),6)))
print("Precision-Macro: "+str(round(metrics.flat_precision_score(y, y_pred, average='macro', labels=labels),6)))

print("F1-Weighted: "+str(round(metrics.flat_f1_score(y, y_pred, average='weighted', labels=labels),6)))
print("Recall-Weighted: "+str(round(metrics.flat_recall_score(y, y_pred, average='weighted', labels=labels),6)))
print("Precision-Weighted: "+str(round(metrics.flat_precision_score(y, y_pred, average='weighted', labels=labels),6)))
data
i = 0
n = 64

test_text = X[n]
sent, y_predict = predict_fn(test_text)

tags = data[data['Sentence #']==n+1073]['Tag']
print("{:15} ({:8}): {}".format("Word", "True", "Pred"))
for w, true,  pred in zip(sent, tags, y_predict):        
    print("{:15} ({:8}): {}".format(w, true, pred))    
from spacy import displacy
n=64
test_text = X[n]
print(test_text)
doc2 = nlp(test_text)
entities = []
entity = []
for ent in doc2.ents:
    if ent.label_.split("-")[0] == 'B' and len(entity) > 0:
        print(entity)
        entities.append((' '.join(entity),ent.label_.split("-")[1]))
        entity = []
    entity.append(ent.text)
    print(entity)
    print(ent.text, ent.label_)

entities.append((' '.join(entity),ent.label_.split("-")[1]))

print(entities)
colors = {"B-LOC": "#fc9ce7", "I-LOC": "#fc9ce7","B-TIME":'#3371ff',"I-TIME":'#3371ff'}
options = {"ents": ["B-LOC","I-LOC","B-TIME","I-TIME"], "colors": colors}
displacy.render(doc2, style="ent",options=options)
ents = [(e.text, e.label_) for e in doc2.ents]
print(ents)
from spacy import displacy


n=64
test_text = X[n]

doc2 = nlp(test_text)
colors = {"B-LOC": "#fc9ce7", "I-LOC": "#fc9ce7","B-TIME":'#3371ff',"I-TIME":'#3371ff'}
options = {"ents": ["B-LOC","I-LOC","B-TIME","I-TIME"], "colors": colors}
displacy.render(doc2, style="ent",options=options)

ents = [(e.text, e.label_) for e in doc2.ents]

def get_entities():
    entities = []
    entity = ''
    tokens = []
    for e in range(len(ents)):
        token = ents[e][0]
        ner = ents[e][1]
        ner_iob = ner.split("-")[0]
        ner_text = ner.split("-")[1] 

        if (ner_iob == 'B' and len(tokens) > 0):        
            t = ' '.join(tokens)
            entities.append((t,entity))
            tokens = []

        entity = ner_text        
        tokens.append(token)
        if e == len(ents)-1:
            t = ' '.join(tokens)
            entities.append((t,entity))


    return entities

entities = get_entities()
entities
ent = [ t for (t,l) in entities  if l == 'loc' ]
ent
' '.join(ent)

