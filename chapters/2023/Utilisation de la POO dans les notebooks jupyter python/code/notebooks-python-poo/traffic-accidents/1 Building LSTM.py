import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from gensim.models.doc2vec import Doc2Vec
import spacy

import fasttext
import fasttext.util
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
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
file = 'ner-crf-training-data.tsv'
dir_ = "../../../data/v1/NER/train/"
train = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3, names=['Sentence #','Word','POS','Tag'])
#dataset[:50]
file = 'ner-crf-test-data.tsv'
dir_ = "../../../data/v1/NER/test/"
test = pd.read_csv(dir_+file, delimiter = "\t", quoting = 3, names=['Sentence #','Word','POS','Tag'])
train['Tag'].value_counts()
test['Tag'].value_counts()
#tags = list(set(data["Tag"].values))
tags = ['B-loc', 'I-loc', 'B-time', 'I-time', 'O']
n_tags = len(tags); n_tags
getter = SentenceGetter(train)
train_sentences = getter.sentences
#sentences
plt.hist([len(s) for s in train_sentences], bins=50)
plt.show()
getter = SentenceGetter(test)
test_sentences = getter.sentences
plt.hist([len(s) for s in test_sentences], bins=50)
plt.show()
max_len = 70
#word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
tag2idx["B-time"] #Id para la etiqueta b-time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#zeros = np.zeros(300,dtype=np.float32)
#X_train = [[word2idx[w[0]] for w in s] for s in train_sentences]
#X_train = pad_sequences(maxlen=max_len, dtype='float32', sequences=X_train, padding="post", value=zeros) # (1072,70,200)

y_train = [[tag2idx[w[2]] for w in s] for s in train_sentences]
y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=tag2idx["O"])

#print(X_train[1])
print(y_train[1])
#zeros = np.zeros(300,dtype=np.float32)
#X_test = [[word2idx[w[0]] if w[0] in words else word2idx['UNDEFINED'] for w in s] for s in test_sentences]
#X_test = pad_sequences(maxlen=max_len, dtype='float32', sequences=X_test, padding="post", value=zeros)

y_test = [[tag2idx[w[2]] for w in s] for s in test_sentences]
y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])

#print(X_test[1])
print(y_test[1])
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]
y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]
_dir = "../../../data/v1/NER/numpy/"
X_train = np.load(_dir+'X_train-ft-300.npy')
X_test = np.load(_dir+'X_test-ft-300.npy')
print("Train:",X_train.shape, len(y_train))
print("Test:",X_test.shape, len(y_test))
from tensorflow.keras.models import Model
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow import keras
import tensorflow as tf

"""
from sklearn_crfsuite import metrics

acc = [] 
r_micro = []
r_macro = []
r_weighted = []
p_micro = []
p_macro = []
p_weighted = []
f1_micro = []
f1_macro = []
f1_weighted = []
loss_val = []
acc_val = []
for i in range(25):
    print(i)
    """
dropout = 0.5
input_shape=(70,300)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)


model = Sequential()
#model.add(Embedding(input_dim=n_words, output_dim=max_len, input_length=max_len)) # 70-dim embedding; input_dim=n_words+1 por el 9999 para otros tokens
model.add(Bidirectional(LSTM(units=100, return_sequences=True, dropout=dropout, recurrent_dropout=dropout), input_shape=input_shape))
model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
optimizer = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

#print(model.summary())
history = model.fit(X_train, np.array(y_train), batch_size=16, epochs=40, validation_split=0.2, verbose=1, callbacks=[callback])
"""
    test_pred = model.predict(X_test, verbose=0)

    pred_labels = pred2label(test_pred)
    test_labels = pred2label(y_test)
    
    metrics.flat_f1_score(test_labels, pred_labels, average='weighted', labels=labels)
    
    metrics.flat_precision_score(test_labels, pred_labels, average='weighted', labels=labels)
    metrics.flat_recall_score(test_labels, pred_labels, average='weighted', labels=labels)

    acc.append(metrics.flat_accuracy_score(test_labels, pred_labels))
  
    r_micro.append(metrics.flat_recall_score(test_labels, pred_labels, average='micro', labels=labels))
    r_macro.append(metrics.flat_recall_score(test_labels, pred_labels, average='macro', labels=labels))
    r_weighted.append(metrics.flat_recall_score(test_labels, pred_labels, average='weighted', labels=labels))
    
    p_micro.append(metrics.flat_precision_score(test_labels, pred_labels, average='micro', labels=labels))
    p_macro.append(metrics.flat_precision_score(test_labels, pred_labels, average='macro', labels=labels))
    p_weighted.append(metrics.flat_precision_score(test_labels, pred_labels, average='weighted', labels=labels))
    
    f1_micro.append(metrics.flat_f1_score(test_labels, pred_labels, average='micro', labels=labels))
    f1_macro.append(metrics.flat_f1_score(test_labels, pred_labels, average='macro', labels=labels))
    f1_weighted.append(metrics.flat_f1_score(test_labels, pred_labels, average='weighted', labels=labels))

    loss_val.append(history.history['val_loss'][::-1][0])
    acc_val.append(history.history['val_accuracy'][::-1][0])
"""
model.summary()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(15, 8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(figsize=(15, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
test_pred = model.predict(X_test, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:        
        out_i = []
        for p in pred_i:               
            p_i = np.argmax(p)                                        
            out_i.append(idx2tag[p_i].replace("PAD", "O"))            
        out.append(out_i)        
    return out

pred_labels = pred2label(test_pred)
test_labels = pred2label(y_test)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))
from sklearn_crfsuite import metrics

labels = ['B-loc', 'I-loc', 'B-time', 'I-time']
# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    test_labels, pred_labels, labels=sorted_labels, digits=4
))

print("F1-score: {:.4%}".format(metrics.flat_f1_score(test_labels, pred_labels, average='weighted', labels=labels)))
i = 1
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
np.argmax(y_test[i], axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, true, pred in zip(test_sentences[i], np.argmax(y_test[i],axis=-1), p[0]):    
    print("{:15} ({:5}): {}".format(w[0], tags[true], tags[pred]))


