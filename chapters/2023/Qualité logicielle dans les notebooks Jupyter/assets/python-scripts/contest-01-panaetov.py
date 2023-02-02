#!/usr/bin/env python
# coding: utf-8

# Изначально повторяем все идеи из домашней работы: используем StandardScaler и PolynomialFeatures, обучаем ElasticNet на логарифме. Выкидываем столбец 'data', нам он пока не интересен. Самое первое, что очень сильно улучшает скор - это one hot encoding колонки 'zip-code'.

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def mape(y_true, y_predict): 
    return np.mean(np.abs((y_true - y_predict) / y_true)) * 100

# In[2]:


X = np.load('x_train.npy')
target = np.load('y_train.npy')
df = pd.DataFrame(X)

df.drop('date', axis=1, inplace=True)
df = pd.get_dummies(data=df, columns=['zipcode'])

Scaler = StandardScaler()
df = Scaler.fit_transform(df)

PolyFeatures = PolynomialFeatures(2)
df = pd.DataFrame(PolyFeatures.fit_transform(df))

X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8, random_state=42)

Model = ElasticNet(alpha=0.00035)
Model.fit(X_train, np.log(y_train))
mape(y_test, np.exp(Model.predict(X_test)))

# Теперь постепенно будем добавлять мелочи, которые немного улучшают скор. Поскольку мы хотим использовать PolynomialFeatures(), нам важно не сделать слишком много бесполезных столбцов. Третью степень после изменения 'zipcode' мы использовать не можем, уже слишком много. А вторая степень пока считается быстро, поэтому можно добавить какие-то признаки, которые не найдет PolynomialFeatures(2): логарифмы и экспоненты признаков, корни и прочее.
# Ещё можно добавлять признаки в другом направлении: например взять какую-нибудь точку и посчитать расстояния от всех других до неё. Посмотрим на 75% в колонках 'lat' и 'long' и посчитаем расстояния до неё.

# In[3]:


X = np.load('x_train.npy')
df = pd.DataFrame(X)
df.lat.describe(), df.long.describe()

# Ещё помогает умножать ответы на 0.99 и другие неадекватные вещи.

# In[4]:


X = np.load('x_train.npy')
target = np.load('y_train.npy')
df = pd.DataFrame(X)

df.drop('date', axis=1, inplace=True)
df = pd.get_dummies(data=df, columns=['zipcode'])

df['sqrt_living'] = np.sqrt(df.sqft_living)
df['sqrt_lot'] = np.sqrt(df.sqft_lot)
df['sqrt_above'] = np.sqrt(df.sqft_above)
df['sqrt_basement'] = np.sqrt(df.sqft_basement)

lat0 = 47.67
long0 = -122.12
df['dist'] = np.sqrt(np.square(df['lat'] - lat0) + np.square(df['long'] - long0))

Scaler = StandardScaler()
df = Scaler.fit_transform(df)

PolyFeatures = PolynomialFeatures(2)
df = pd.DataFrame(PolyFeatures.fit_transform(df))

X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8, random_state=42)

Model = ElasticNet(alpha=0.00035)
Model.fit(X_train, np.log(y_train))
mape(y_test, 0.99 * np.exp(Model.predict(X_test)))

# Можно продолжать добавлять сюда разные признаки, но сильно улучшить модель у меня больше не получалось. Но помогла другая идея - посмотреть на все признаки локально: по ближайшим N соседям (подбирая разное N, у меня лучший результат показало N=70). Сначала я просто посчитал среднюю цену и все средние площади вокруг каждого дома и сохранил это как новые данные, потому что считалось очень долго и делать это каждый раз было бы самоубийством. С новыми признаки делаем то же самое, что и раньше - извлекаем корни, берем отношения. Каждый признак отдельно проверяем на адекватность - если он улучшает скор, оставляем, если не улучшает - убираем. Так же можно заметить, что 'zipcode' нам больше не нужны и можно освободить кучу места, чтобы быстрее считалось и можно было больше экспериментивароть. Сейчас можно увеличить степень в PolynomialFeatures(), поскольку основную часть занимал 'zipcode', но это не дает прибавки к результату.

# In[5]:


target = np.load('y_train.npy')
df = pd.read_pickle('df70sq4')

df.drop('date', axis=1, inplace=True)

df['near_price_per_living'] = df['near_price'] / df['near_sqft_living']

lat0 = 47.67
long0 = -122.12
df['dist'] = np.sqrt(np.square(df['lat'] - lat0) + np.square(df['long'] - long0))

Scaler = StandardScaler()
df = Scaler.fit_transform(df)

PolyFeatures = PolynomialFeatures(2)
df = pd.DataFrame(PolyFeatures.fit_transform(df))

X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.8, random_state=42)

Model = ElasticNet(alpha=0.00035)
Model.fit(X_train, np.log(y_train))
mape(y_test, 0.99 * np.exp(Model.predict(X_test)))

# Это очень забавно, но во время написания этого отчета я заметил, что все те многочисленные признаки совершенно не помогают (и даже мешают), поэтому в ячейке выше их нет. Я оставил только один, который действительно немного улучшает результат. Полезными оказались только посчитанные изначально - локальная средняя стоимость и локальные среднии площади.
