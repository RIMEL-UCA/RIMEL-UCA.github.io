#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# # Семинар 22

# In[1]:


import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

# # Анализ текстов
# 
# Ранее мы изучали интуитивно простые способы формирования признаков для текстов, такие как векторизация и tf-idf — как правило, эти методы используются в задачах обучения с учителем, где похожие объекты имеют свою строгую специфику и радикально отличаются от объектов других классов (например, задача определения спам-писем).
# 
# Однако иногда после получения выборки необходимо узнать больше о природе корпуса текстов для успешного решения исходной задачи, поэтому семинар будет посвящен применению изученных нами в курсе unsupervised методов к выборкам текстов.

# ## Данные
# 
# В качестве корпуса текстов в нашем случае будет выступать выборка, состоящая из синопсисов к 100 лучшим фильмам всех времён по версии IMDB. Попробуем при помощи изученых методов выявить причины, по которым данные фильмы вызывают повышенный интерес.
# 
# Загрузим названия, синопсисы (с IMDB и Википедии) и жанры фильмов:

# In[2]:


#import three lists: titles, links and wikipedia synopses
titles = open('data/title_list.txt').read().split('\n')

#ensures that only the first 100 are read in
titles = titles[:100]

links = open('data/link_list_imdb.txt').read().split('\n')
links = links[:100]

synopses_wiki = open('data/synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_clean_wiki = []
for text in synopses_wiki:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_wiki.append(text)

synopses_wiki = synopses_clean_wiki
    
    
genres = open('data/genres_list.txt').read().split('\n')
genres = genres[:100]

print(str(len(titles)) + ' titles')
print(str(len(links)) + ' links')
print(str(len(synopses_wiki)) + ' synopses')
print(str(len(genres)) + ' genres')

# In[3]:


synopses_imdb = open('data/synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

synopses_clean_imdb = []

for text in synopses_imdb:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean_imdb.append(text)

synopses_imdb = synopses_clean_imdb

# In[4]:


synopses = [synopses_wiki[i] + synopses_imdb[i] for i in range(len(synopses_wiki))]
ranks = np.arange(100).tolist()

# ## Предобработка данных
# 
# Как и ранее, текстовые данные требуют тщательной предобработки — в частности, необходимо произвести:
#  * токенизацию (разбиение текста на термины);
#  * удаление стоп-слов (часто встречающиеся всюду слова не оказывают важного влияния на смысловую составляющую текста, а потому могут быть опущены);
#  * лемматизацию/стемминг (для устранения влияния склонений, спряжений и пр.).

# In[5]:


# load nltk's English stopwords
stopwords = nltk.corpus.stopwords.words('english')

# In[6]:


# load nltk's SnowballStemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# In[7]:


def tokenize_and_preprocess(text, stem=True):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    if stem:
        filtered_tokens = [stemmer.stem(t) for t in filtered_tokens]
    return filtered_tokens

# Создадим словари всех токенов из корпуса с использованием стемминга и без:

# In[8]:


totalvocab_stemmed = []
totalvocab_tokenized = []

for i in synopses:
    allwords_stemmed = tokenize_and_preprocess(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_and_preprocess(i, stem=False)
    totalvocab_tokenized.extend(allwords_tokenized)

# Посмотрим на результаты стемминга слов из корпуса:

# In[9]:


vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
vocab_frame

# Как правило, стемминг показывает неплохие результаты для английских текстов в отличие от русских в силу особенностей морфологии русского языка.

# ## tf-idf
# 

# Для начала используем уже известные нам методы — преобразуем тексты при помощи tf-idf и вычислим расстояние между полученными векторами для дальнейшей кластеризации и визуализации. В качестве функции расстояния будем использовать $\rho (x, z) = 1 - cosine(x, z),$ где $cosine(x, z)$ — косинусная мера между векторами $x, z$.

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer= (lambda x: tokenize_and_preprocess(x)), ngram_range=(1,3))

%time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
terms = tfidf_vectorizer.get_feature_names()
print(tfidf_matrix.shape)

# In[11]:


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
dist = np.maximum(np.zeros(dist.shape), dist) # overriding precision errors
pd.DataFrame(dist)

# На основе полученных расстояний для каждого фильма можно найти самые близкие к нему, основываясь на множестве слов каждого синопсиса:

# In[12]:


film_number = 0

print(titles[film_number])
print([titles[i] for i in dist[film_number, :].argsort()[:10]])

# ## Кластеризация при помощи K-means
# 
# Основываясь на построенных признаках, попробуем кластеризовать фильмы при помощи метода K-means.

# In[13]:


from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters, random_state=5)

%time km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# In[14]:


import pandas as pd

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }
frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])
frame

# Посмотрим на количество фильмов в каждом кластере и их средний ранг:

# In[15]:


frame['cluster'].value_counts()

# In[16]:


grouped = frame['rank'].groupby(frame['cluster'])

grouped.mean()

# Можно заметить, что средний ранг фильмов в кластерах 1 и 3 меньше, чем в остальных, что значит, что в эти кластеры вошли фильмы, которые в среднем лучше, чем остальные. Посмотрим, какие слова характеризуют фильмы каждого кластера:

# In[17]:


from __future__ import print_function

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()

# Внутри каждого кластера можно заметить определенные паттерны — например, кластер 3 включает в основном фильмы про семью, а 1 — про войну и потери. Это подтверждается также и самыми популярными жанрами внутри кластера:

# In[18]:


genre_set = set()

for i in range(frame.shape[0]):
    tmp = frame.iloc[i, -1][1:-1]
    genre_set |= set([genre.strip()[3:-1] for genre in tmp.split(',')])
genre_set = list(genre_set)

# In[19]:


for genre in genre_set:
    frame['Is_genre_{}'.format(genre)] = frame['genre'].apply(lambda x: int(genre in x))

grouped = frame[frame.columns.tolist()[4:]].groupby(frame['cluster'], as_index=False).mean()

for i in range(num_clusters):
    print (i, [genre_set[num] for num in np.array(frame.iloc[i, 4:]).argsort()[-5:]])

# Однако также можно заметить, что разметка некоторых объектов кажется некорректной — такое происходит из-за того, что учитываются отдельные слова, а не темы, поэтому синонимы вноят некоторый хаос в полученную кластеризацию. Для выявления тем, которые порождают повышенный интерес к этому списку фильмов, необходимо использовать несколько другой подход.

# ## Визуализация кластеров
# 
# Визуализируем полученные ранее кластеры:

# In[20]:


import os

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, metric="precomputed", random_state=1)

pos = tsne.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]

# In[21]:


cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {i: ','.join(['%s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore') for ind in order_centroids[i, :6]])
                 for i in range(5)}

# In[22]:


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x', 
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',
        which='both',
        left='off',
        top='off',
        labelleft='off')
    
ax.legend(numpoints=1)

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

    
    
plt.show()

# In[23]:


#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

# In[24]:


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 

#group by cluster
groups = df.groupby('label')

#define custom css to format the font and to remove the axis labeling
css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }
"""

# Plot 
fig, ax = plt.subplots(figsize=(14,6)) #set plot size
ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, label=cluster_names[name], mec='none', color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    #set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                       voffset=10, hoffset=10, css=css)
    #connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    
    #set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    
    #set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    
ax.legend(numpoints=1) #show legend with only one dot

mpld3.display() #show the plot

#uncomment the below to export to html
#html = mpld3.fig_to_html(fig)
#print(html)

# ## Иерархическая кластеризация
# 
# Попробуем оставить используемые признаки, основанных только на вхождениях определенных слов, и использовать другой метод кластеризации — например, иерархическую кластеризацию:

# In[25]:


from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 30)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

# Процесс иерархической кластеризации, как правило, изображается в виде дендрограммы, что позволяет наблюдать за процессом слияния, однако при использовании этого метода нет возможности получить какую-либо информацию о том, что общего у объектов каждого кластера.

# ## Латентное размещение Дирихле (LDA)
# 
# До сих пор при попытке выделить причины интереса к фильмам мы опирались лишь на конкретные слова, что приводило к неоднозначным результатам. Так происходит в том числе потому, что язык содержит синонимы, и одно слово может относиться к разным тематикам и нести разный смысл. При использовании признаков, основанных на отдельных словах, темы с похожими словарями будут смешиваться.
# 
# Будем считать, что каждый текст относится к некому малому числу тем, присутствующих в корпусе, а для каждого токена задана вероятность его присутствия в тексте соответствующей тематики. Данный подход называется тематическим моделированием и используется в том числе для выделения тематик из текстов.
# 
# Используем метод LDA, рассмотренный на лекции, для извлечения тематик из синопсисов. Для начала создадим словарь токенов на основе корпуса:

# In[26]:


import string
def strip_proppers(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()

# In[28]:


from gensim import corpora, models, similarities 

preprocess = [strip_proppers(doc) for doc in synopses]

%time tokenized_text = [tokenize_and_preprocess(text) for text in preprocess]

%time texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

# In[29]:


# dictionary is a set of tokens
dictionary = corpora.Dictionary(texts)
dictionary[3]

# А также отбросим слишком редкие и слишком частые слова (по аналогии с min_df и max_df):

# In[30]:


dictionary.filter_extremes(no_below=1, no_above=0.8)

# Несмотря на то, что LDA выделяет тематики текста, он также работает в предположении "мешка слов" — то есть предполагается, что все слова независимы, и произвольная перестановка слов в тексте ни на что не влияет. Поэтому каждый текст можно представлять в виде пар (токен, количество вхождений токена в текст):

# In[31]:


corpus = [dictionary.doc2bow(text) for text in texts]
corpus[0]

# После приведения текстов к указанному виду можно строить соответствующую модель:

# In[32]:


%time lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, update_every=5, chunksize=10000, passes=100, random_state=1)

# Каждая из 5 выделенных тематик характеризуется наиболее вероятными токенами и значениями их вероятностей в рамках данной тематики $p(w|t):$

# In[33]:


topics = lda.print_topics(5, num_words=20)
topics

# Полученное разбиение на тематики кажется более обоснованным, чем полученное ранее, — теперь, к примеру, слово kill по-прежнему фигурирует в различных топиках, однако улавливается общая смысловая составляющая объектов одного кластера — для кластера 3 речь идет о внутренних междоусобицах, а для кластера 2 — о войнах.
# 
# Можно также получить распределение на токенах для каждой из полученных тем:

# In[34]:


topics_matrix = np.array([val for (key, val) in lda.show_topics(formatted=False, num_words=20)])
topics_matrix.shape

# In[35]:


topics_matrix

# In[36]:


topic_words = topics_matrix[:,:,0]

for i in topic_words:
    print([str(word) for word in i])
    print()

# Зачастую методы тематического моделирования используют для формирования нового признакового пространства документов — для этого для каждого документа в качестве признаков рассматривают полученное распределение на топиках:

# In[44]:


film_number = 0
print(titles[film_number], lda[corpus[film_number]])

# In[45]:


film_number = 0
print(titles[film_number], lda.get_document_topics(corpus[film_number], minimum_probability=0))

# Используем новые признаки для поиска похожих фильмов и визуализации, как ранее:

# In[39]:


topics_matrix = np.array([[doc_topic_prob[1] for doc_topic_prob in lda.get_document_topics(doc, minimum_probability=0)] for doc in corpus])
topics_matrix

# In[40]:


dist_topics = 1 - cosine_similarity(topics_matrix)
dist_topics = np.maximum(np.zeros(dist.shape), dist) # overriding precision errors
pd.DataFrame(dist_topics)

# In[41]:


film_number = 0

print(titles[film_number])
print([titles[i] for i in dist_topics[film_number, :].argsort()[:10]])

# In[42]:


%time km.fit(topics_matrix)

clusters = km.labels_.tolist()

films = { 'title': titles, 'rank': ranks, 'synopsis': synopses, 'cluster': clusters, 'genre': genres }
frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster', 'genre'])
frame

# In[43]:


tsne = TSNE(n_components=2, metric="precomputed", random_state=1)

pos = tsne.fit_transform(dist_topics)

xs, ys = pos[:, 0], pos[:, 1]

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {i: ','.join(['%s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore') for ind in order_centroids[i, :6]])
                 for i in range(5)}

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 


groups = df.groupby('label')



fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',
        which='both',
        left='off',
        top='off',
        labelleft='off')
    
ax.legend(numpoints=1)

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.show()

# ## Материалы
# 
# * [Анализ дневника Марты Баллард](http://www.cameronblevins.org/posts/topic-modeling-martha-ballards-diary/)
# * [Анализ текстов газет во время гражданской войны в США](http://dsl.richmond.edu/dispatch/pages/intro)
# * [Document clustering with Python](http://brandonrose.org/clustering)

# In[ ]:



