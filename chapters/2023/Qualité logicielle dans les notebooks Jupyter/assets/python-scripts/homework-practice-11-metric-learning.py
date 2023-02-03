#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# # Практическое задание 11. Поиск ближайших соседей
# 
# ## Общая информация
# 
# Дата выдачи: 09.04.2021
# 
# **Мягкий дедлайн: 24.04.2021 23:59 MSK**
# 
# **Жёсткий дедлайн: 27.04.2021 23:59 MSK**
# 
# ## Оценивание и штрафы
# 
# Каждая из задач имеет определенную «стоимость» (указана в скобках около задачи). Максимально допустимая оценка за работу — 7 баллов.
# 
# 
# Сдавать задание после указанного жёсткого срока сдачи нельзя. При выставлении неполного балла за задание в связи с наличием ошибок на усмотрение проверяющего предусмотрена возможность исправить работу на указанных в ответном письме условиях.
# 
# Задание выполняется самостоятельно. «Похожие» решения считаются плагиатом и все задействованные студенты (в том числе те, у кого списали) не могут получить за него больше 0 баллов (подробнее о плагиате см. на странице курса). Если вы нашли решение какого-то из заданий (или его часть) в открытом источнике, необходимо указать ссылку на этот источник в отдельном блоке в конце вашей работы (скорее всего вы будете не единственным, кто это нашел, поэтому чтобы исключить подозрение в плагиате, необходима ссылка на источник).
# 
# Неэффективная реализация кода может негативно отразиться на оценке.
# 
# ## Формат сдачи
# 
# Задания сдаются через систему anytask. Посылка должна содержать:
# 
# * Ноутбук homework-practice-11-knn-Username.ipynb
# 
# Username — ваша фамилия и имя на латинице именно в таком порядке.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tqdm.notebook import tqdm

# Возьмем [датасет](https://www.kaggle.com/delayedkarma/impressionist-classifier-data)  с картинами известных импрессионистов. Работать будем не с самими картинками, а с эмбеддингами картинок, полученных с помощью сверточного классификатора.
# 
# ![](https://storage.googleapis.com/kagglesdsdata/datasets/568245/1031162/training/training/Gauguin/190448.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210405%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210405T125358Z&X-Goog-Expires=172799&X-Goog-SignedHeaders=host&X-Goog-Signature=a271b474bf9ec20ba159b951e0ae680fc2b0c694666031f7ea6fc39598172cc55e10f75c12b678b21da9e6bdc20e46886133c219625648b407d2f600eebfdda909b29e0f7f13276d8fea2f8d0480d6298bd98e7f118eb78e8b632fc3d141365356b0e3a2fdd4f09119f99f0907a31da62e8dae7e625e32d831238ecc227b1f5ad2e96a8bfb43d93ef6fe88d7e663e51d387d3550dcad2a7eefc5c941028ba0d7751d18690cf2e26fcdfaa4dacd3dcbb3a4cbb355e62c08b158007b5e764e468cecd3292dae4cfc408e848ecf3e0e5dbe5faa76fcdd77d5370c868583c06e4e3d40c73a7435bd8c32a9803fe6b536e1c6f0791219aadd06120291e937e57c214a)

# In[ ]:


%%bash

mkdir embeddings

GIT="https://github.com/esokolov/ml-course-hse/raw/master/2021-spring/homeworks-practice/homework-practice-11-metric-learning/embeddings"
wget -P ./embeddings $GIT/embeds_train.npy
wget -P ./embeddings $GIT/embeds_test.npy
wget -P ./embeddings $GIT/labels_train.npy
wget -P ./embeddings $GIT/labels_test.npy

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# In[ ]:


X_train = np.load('embeddings/embeds_train.npy')
y_train = np.load('embeddings/labels_train.npy')
X_test = np.load('embeddings/embeds_test.npy')
y_test = np.load('embeddings/labels_test.npy')

# Будем смотреть на обычную долю верных ответов и на долю верных ответов в топ-3.

# In[ ]:


def top_3_accuracy_score(y_true, probas):
    preds = np.argsort(probas, axis=1)[:, -3:]
    matches = np.zeros_like(y_true)
    for i in range(3):
        matches += (preds[:, i] == y_true)
    return matches.sum() / matches.size

def scorer(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# **Задание 1. (1 балл)**
# 
# Обучите классификатор k ближайших соседей (из sklearn) на данных, подобрав лучшие гиперпараметры. Замерьте качество на обучающей и тестовой выборках.

# In[ ]:


#  (*・ω・)ﾉ

# **Задание 2. (2 балла)** 
# 
# Теперь будем пользоваться метрикой Махалонобиса. Обучите её одним из методов [отсюда](http://contrib.scikit-learn.org/metric-learn/supervised.html). Напомним, что вычисление метрики Махалонобиса эквивалентно вычислению евклидова расстояния между объектами, к которым применено некоторое линейное преобразование (вспомните семинары). Преобразуйте данные и обучите kNN на них, перебрав гиперпараметры, замерьте качество.
# 
# Заметим, что в библиотеке metric-learn есть несколько способов обучать матрицу преобразования. Выберите лучший, аргументируйте свой выбор.

# Note: Некоторые методы с дефолтными параметрами учатся очень долго, будьте внимательны. Советуем выставить параметр `tolerance=1e-3`.
# 

# In[ ]:


# ⊂(￣▽￣)⊃

# **Задание 3. (1 балл)** 
# 
# Что будет, если в качестве матрицы в расстоянии Махалонобиса использовать случайную матрицу? Матрицу ковариаций?

# In[ ]:


# (•)(•)ԅ(≖‿≖ԅ)

# **Задание 4. (1 балл)** Обучите какой-нибудь градиентный бустинг на обычных и трансформированных наборах данных, замерьте качество, задумайтесь о целесообразности других методов.

# In[ ]:


# C=C=C=C=C=┌(;・ω・)┘

# **Бонус. (1 балл)**
# 
# Достигните доли верных ответов 0.75 на тестовой выборке, не используя нейросети.

# In[ ]:


# ( ・・)つ―{}@{}@{}-

# **Бонус. (до 0.5 баллов)**
# 
# Ассистентам часто очень скучно проверять однотипные работы и они грустят, поэтому вставьте сюда лучший мем из вашей коллекции, чтобы поднять им (и всем, с кем они поделятся) настроение.

# In[ ]:



