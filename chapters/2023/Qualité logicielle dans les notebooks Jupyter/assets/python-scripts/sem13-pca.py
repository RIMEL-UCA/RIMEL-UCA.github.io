#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# # Семинар 13. Понижение размерности

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# На прошлом семинаре мы рассмотрели приложение понижения размерности для 
# визуализации данных. Для этого по исходному пространству высокой размерности
# строились признаки в 2 или 3 измерениях, и после этого объекты можно наглядно
# изобразить используя эти признаки.
# 
# Понижение размерности можно использовать и для других целей:
# 
# * Сокращение ресурсоемкости алгоритмов
# * Ослабление влияния проклятия размерности и тем самым уменьшение переобучения
# * Переход к более информативным признакам
# 
# На этом семинаре мы будем понижать размерность ориентируясь как раз на эти цели.
# Тогда этот процесс также можно называть и выделением признаков.

# ## Отбор признаков
# 
# Самый простой способ выделения признаков - их отбор. Не будем заострять много внимания
# на этом методе, так как он очень простой, просто приведем пример, показывающий, что
# так можно примитивно сокращать ресурсоемкость алгоритмов.
# 
# Отберем признаки на основе их корреляции с целевым признаком, и сравним результаты с исходными.

# In[2]:


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

ds = load_boston()
X, y = ds.data, ds.target
indexes = np.arange(len(y))
np.random.seed(52342)
np.random.shuffle(indexes)
X = X[indexes, :]
y = y[indexes]

features_ind = np.arange(X.shape[1])
corrs = np.abs([pearsonr(X[:, i], y)[0] for i in features_ind])
importances_sort = np.argsort(corrs)
plt.barh(features_ind, corrs[importances_sort])
X = X[:, importances_sort]

# In[3]:


from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

features_counts = np.arange(1, X.shape[1] + 1)

def scores_by_features_count(reg):
    scores = []
    for features_part in features_counts:
        X_part = X[:, :features_part]
        scores.append(cross_val_score(reg, X_part, y).mean())
    return scores
    
plt.figure()
linreg_scores = scores_by_features_count(LinearRegression())
plt.plot(features_counts, linreg_scores, label='LinearRegression')
rf_scores = scores_by_features_count(RandomForestRegressor(n_estimators=100, max_depth=3))
plt.plot(features_counts, rf_scores, label='RandomForest')
plt.legend(loc='best')

# В общем, если мы захотим немного сократить потребление ресурсов, пожертвовав частью качества,
# видно, что это можно сделать.

# ## Метод главных компонент (Principal Component Analysis, PCA)
# 
# Выделение новых признаков путем их отбора часто дает плохие результаты, и
# в некоторых ситуациях такой подход практически бесполезен. Например, если
# мы работаем с изображениями, у которых признаками являются яркости пикселей,
# невозможно выбрать небольшой поднабор пикселей, который дает хорошую информацию о
# содержимом картинки. 
# 
# Поэтому признаки нужно как-то комбинировать. Рассмотрим метод главных компонент.
# 
# Этот метод делает два важных упрощения задачи
# 
# 1. Игнорируется целевая переменная
# 2. Строится линейная комбинация признаков
# 
# П. 1 на первый взгляд кажется довольно странным, но на практике обычно не является
# таким уж плохим. Это связано с тем, что часто данные устроены так, что имеют какую-то
# внутреннюю структуру в пространстве меньшей размерности, которая никак не связана с
# целевой переменной. Поэтому и оптимальные признаки можно строить не глядя на ответ.
# 
# П. 2 тоже сильно упрощает задачу, но далее мы научимся избавлятся от него.

# ### Теория
# 
# Кратко вспомним, что делает этот метод (подробно см. в лекции).
# 
# Обозначим $X$ - матрица объекты-признаки, с нулевым средним каждого признака,
# а $w$ - некоторый единичный вектор. Тогда
# $Xw$ задает величину проекций всех объектов на этот вектор. Далее ищется вектор,
# который дает наибольшую дисперсию полученных проекций (то есть наибольшую дисперсию
# вдоль этого направления):
# 
# $$
#     \max_{w: \|w\|=1} \| Xw \|^2 =  \max_{w: \|w\|=1} w^T X^T X w
# $$
# 
# Подходящий вектор тогда равен собственному вектору матрицы $X^T X$ с наибольшим собственным
# значением. После этого все пространство проецируется на ортогональное дополнение к вектору
# $w$ и процесс повторяется.

# ### PCA на плоскости
# 
# Для начала посмотрим на метод PCA на плоскости для того, чтобы
# лучше понять, как он устроен.
# 
# Попробуем специально сделать один из признаков более значимым и проверим, что PCA это обнаружит. Сгенерируем выборку из двухмерного гауссовского распределения. Обратите внимание, что выборка
# изначально выбирается центрированной.

# In[4]:


np.random.seed(314512)

data_synth_1 = np.random.multivariate_normal(
    mean=[0, 0], 
    cov=[[4, 0], 
         [0, 1]],
    size=1000)

# Теперь изобразим точки выборки на плоскости и применим к ним PCA для нахождения главных компонент.
# В результате работы PCA из sklearn в `dec.components_` будут лежать главные направления (нормированные), а в `dec.explained_variance_` - дисперсия, которую объясняет каждая компонента. Изобразим на нашем графике эти направления, умножив их на дисперсию для наглядного отображения их
# значимости.

# In[5]:


from sklearn.decomposition import PCA


def PCA_show(dataset):
    plt.scatter(*zip(*dataset), alpha=0.5)
    
    dec = PCA()
    dec.fit(dataset)
    ax = plt.gca()
    for comp_ind in range(dec.components_.shape[0]):
        component = dec.components_[comp_ind, :]
        var = dec.explained_variance_[comp_ind]
        start, end = dec.mean_, component * var
        ax.arrow(start[0], start[1], end[0], end[1],
                 head_width=0.2, head_length=0.4, fc='k', ec='k')
    
    ax.set_aspect('equal', adjustable='box')

plt.figure(figsize=(7, 7))
PCA_show(data_synth_1)

# Видим, что PCA все правильно нашел. Но это, конечно, можно было сделать и просто посчитав
# дисперсию каждого признака. Повернем наши данные на некоторый фиксированный угол и проверим,
# что для PCA это ничего не изменит.

# In[6]:


angle = np.pi / 6
rotate = np.array([
        [np.cos(angle), - np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
data_synth_2 = rotate.dot(data_synth_1.T).T

plt.figure(figsize=(7, 7))
PCA_show(data_synth_2)

# Ну вот, все нормально. 
# 
# Ниже пара примеров, где PCA отработал не так хорошо (в том смысле, что направления задают не очень хорошие признаки).
# 
# **Упражнение.** Объясните, почему так произошло.

# In[7]:


from sklearn.datasets import make_circles, make_moons, make_blobs

np.random.seed(54242)
data_synth_bad = [
    make_circles(n_samples=1000, factor=0.2, noise=0.1)[0]*2,
    make_moons(n_samples=1000, noise=0.1)[0]*2,
    make_blobs(n_samples=1000, n_features=2, centers=4)[0]/5,
    np.random.multivariate_normal(
        mean=[0, 1.5], 
        cov=[[3, 1], 
             [1, 1]],
        size=1000),
]


plt.figure(figsize=(10, 8))
rows, cols = 2, 2
for i, data in enumerate(data_synth_bad):
    plt.subplot(rows, cols, i + 1)
    PCA_show(data)
    plt.gca().set_aspect('equal', adjustable='box')

# ### Лица людей
# 
# Рассмотрим датасет с фотографиями лиц людей и применим к его признакам PCA.
# 
# Ниже изображены примеры лиц из базы, и последняя картинка - это "среднее лицо".

# In[8]:


from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces(shuffle=True, random_state=432542)
faces_images = faces.data
faces_ids = faces.target
image_shape = (64, 64)
    
mean_face = faces_images.mean(axis=0)

plt.figure(figsize=(10, 4))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples - 1):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(faces_images[i, :].reshape(image_shape), interpolation='none',
               cmap='gray')
    plt.xticks(())
    plt.yticks(())
    
plt.subplot(rows, cols, n_samples)
plt.imshow(mean_face.reshape(image_shape), interpolation='none',
           cmap='gray')
plt.xticks(())
plt.yticks(())

# Теперь найдем главные компоненты

# In[9]:


red = PCA()
faces_images -= mean_face
red.fit(faces_images)

plt.figure(figsize=(10, 4))
rows, cols = 2, 4
n_samples = rows * cols
for i in range(n_samples):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(red.components_[i, :].reshape(image_shape), interpolation='none',
               cmap='gray')
    plt.xticks(())
    plt.yticks(())

# Получилось жутковато, что уже неплохо, но есть ли от этого какая-то польза?
# 
# Во-первых, новые признаки дают более высокое качество классификации.

# In[10]:


%%time

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

gscv_rf = GridSearchCV(RandomForestClassifier(),
                       {'n_estimators': [100, 200, 500, 800], 'max_depth': [2, 3, 4, 5]},
                       cv=5)

gscv_rf.fit(faces_images, faces_ids)
print(gscv_rf.best_score_)

gscv_rf.fit(red.transform(faces_images), faces_ids)
print(gscv_rf.best_score_)

# Во-вторых, их можно использовать для компактного хранения данных. Для этого объекты трансформируются
# в новое пространство, и из него выкидываются самые незначимые признаки.
# 
# Ниже приведены результаты сжатия в 10 раз.

# In[11]:


base_size = image_shape[0] * image_shape[1]

def compress_and_show(compress_ratio):
    red = PCA(n_components=int(base_size * compress_ratio))
    red.fit(faces_images)

    faces_compressed = red.transform(faces_images)
    faces_restored = red.inverse_transform(faces_compressed) + mean_face

    plt.figure(figsize=(10, 4))
    rows, cols = 2, 4
    n_samples = rows * cols
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(faces_restored[i, :].reshape(image_shape), interpolation='none',
                   cmap='gray')
        plt.xticks(())
        plt.yticks(())
        
compress_and_show(0.1)

# И даже при сжатии в 20 раз лица остаются узнаваемыми.

# In[12]:


compress_and_show(0.05)

# ### PCA с ядрами
# 
# Так как PCA фактически работает не исходными признаками, а с матрицей их ковариаций, можно
# использовать для ее вычисления вместо скалярного произведения $\langle x_i, x_j \rangle$ произвольное
# ядро $K(x_i, x_j)$. Это будет соответствовать переходу в другое пространство, в котором
# наше предположение о линейности уже будет иметь смысл. Единственная проблема - непонятно, как
# подбирать ядро.
# 
# Ниже приведены примеры объектов в исходном пространстве (похожие группы обозначены одним цветом
# для наглядности), и результат их трансформации в новые пространства (для разных ядер). Если результаты
# получаются линейно разделимыми - значит мы выбрали подходящее ядро.

# In[13]:


from sklearn.decomposition import KernelPCA


def KPCA_show(X, y):
    reds = y == 0
    blues = y == 1
    
    plt.figure(figsize=(8, 8))
    rows, cols = 2, 2
    plt.subplot(rows, cols, 1)
    plt.scatter(X[reds, 0], X[reds, 1], alpha=0.5, c='r')
    plt.scatter(X[blues, 0], X[blues, 1], alpha=0.5, c='b')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    kernels_params = [
        dict(kernel='rbf', gamma=10),
        dict(kernel='poly', gamma=10),
        dict(kernel='cosine', gamma=10),
    ]
    
    for i, p in enumerate(kernels_params):
        dec = KernelPCA(**p)
        X_transformed = dec.fit_transform(X)
        
        plt.subplot(rows, cols, i + 2)
        plt.scatter(X_transformed[reds, 0], X_transformed[reds, 1], alpha=0.5, c='r')
        plt.scatter(X_transformed[blues, 0], X_transformed[blues, 1], alpha=0.5, c='b')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
np.random.seed(54242)
KPCA_show(*make_circles(n_samples=1000, factor=0.2, noise=0.1))

# In[14]:


np.random.seed(54242)
KPCA_show(*make_moons(n_samples=1000, noise=0.1))
