#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# ## Семинар 2. Знакомство с NumPy и Scikit-Learn

# # NumPy
# 
# **NumPy** — библиотека языка Python, позволяющая [удобно] работать с многомерными массивами и матрицами, содержащая математические функции. Кроме того, NumPy позволяет векторизовать многие вычисления, имеющие место в машинном обучении.
# 
#  - [numpy](http://www.numpy.org)
#  - [numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
#  - [100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)

# In[13]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Основным типом данных NumPy является многомерный массив элементов одного типа — [numpy.ndarray](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.array.html). Каждый подобный массив имеет несколько *измерений* или *осей* — в частности, вектор (в классическом понимании) является одномерным массивом и имеет 1 ось, матрица является двумерным массивом и имеет 2 оси и т.д.

# In[2]:


vec = np.array([1, 2, 3])
vec.ndim # количество осей

# In[3]:


mat = np.array([[1, 2, 3], [4, 5, 6]])
mat.ndim

# Чтобы узнать длину массива по каждой из осей, можно воспользоваться атрибутом shape:

# In[4]:


vec.shape

# Чтобы узнать тип элементов и их размер в байтах:

# In[5]:


mat.dtype.name

# In[6]:


mat.itemsize

# ## Создание массивов
# 
# * Передать итерируемый объект в качестве параметра функции array (можно также явно указать тип элементов):

# In[7]:


A = np.array([1, 2, 3])
A, A.dtype

# In[8]:


A = np.array([1, 2, 3], dtype=float)
A, A.dtype

# * Создание массивов специального вида при помощи функций zeros, ones, empty, identity:

# In[9]:


np.zeros((3,))

# In[10]:


np.ones((3, 4))

# In[11]:


np.identity(3)

# In[12]:


np.empty((2, 5))

# Обратите внимание, что содержимое массива, созданного при помощи функции empty, **не инициализируется**, то есть в качестве значений он **может содержать "мусор"**.

# * Создание последовательностей при помощи функций arange (в качестве парметров принимает левую и правую границы последовательности и **шаг**) и linspace (принимает левую и правую границы и **количество элементов**):

# In[13]:


np.arange(2, 20, 3) # аналогично стандартной функции range python, правая граница не включается

# In[14]:


np.arange(2.5, 8.7, 0.9) # но может работать и с вещественными числами

# In[15]:


np.linspace(2, 18, 14) # правая граница включается (по умолчанию)

# * Для изменения размеров существующего массива можно воспользоваться функцией reshape (при этом количество элементов должно оставаться неизменным):

# In[16]:


np.arange(9).reshape(3, 3)

# Вместо значения длины массива по одному из измерений можно указать -1 — в этом случае значение будет рассчитано автоматически:

# In[17]:


np.arange(8).reshape(2, -1)

# * Транспонирование существующего массива:

# In[18]:


C = np.arange(6).reshape(2, -1)
C

# In[19]:


C.T

# * Объединение существующих массивов по заданной оси:

# In[20]:


A = np.arange(6).reshape(2, -1)
np.hstack((A, A**2))

# In[21]:


np.vstack((A, A**2))

# In[22]:


np.concatenate((A, A**2), axis=1)

# * Повторение существующего массива

# In[23]:


a = np.arange(3)
np.tile(a, (2, 2))

# In[24]:


np.tile(a, (4, 1))

# ## Базовые операции
# 
# * Базовые арифметические операции над массивами выполняются поэлементно:

# In[25]:


A = np.arange(9).reshape(3, 3)
B = np.arange(1, 10).reshape(3, 3)

# In[26]:


print(A)
print(B)

# In[27]:


A + B

# In[28]:


A * 1.0 / B

# In[29]:


A + 1

# In[30]:


3 * A

# In[31]:


A ** 2

# Отдельно обратим внимание на то, что умножение массивов также является **поэлементным**, а не матричным:

# In[32]:


A * B

# Для выполнения матричного умножения необходимо использовать функцию dot:

# In[33]:


A.dot(B)

# Поскольку операции выполняются поэлементно, операнды бинарных операций должны иметь одинаковый размер. Тем не менее, операция может быть корректно выполнена, если размеры операндов таковы, что они могут быть расширены до одинаковых размеров. Данная возможность называется [broadcasting](http://www.scipy-lectures.org/intro/numpy/operations.html#broadcasting):
# ![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/02.05-broadcasting.png)

# In[34]:


np.tile(np.arange(0, 40, 10), (3, 1)).T + np.array([0, 1, 2])

# * Универсальные функции (sin, cos, exp и т.д.) также применяются поэлементно:

# In[35]:


np.exp(A)

# * Некоторые операции над массивами (например, вычисления минимума, максимума, суммы элементов) выполняются над всеми элементами вне зависимости от формы массива, однако при указании оси выполняются вдоль нее (например, для нахождения максимума каждой строки или каждого столбца):

# In[36]:


A

# In[37]:


A.min()

# In[38]:


A.max(axis=1)

# In[39]:


A.sum(axis=1)

# ## Индексация
# 
# Для доступа к элементам может использоваться [много различных способов](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html), рассмотрим основные.
# 
# * Для индексации могут использоваться конкретные значения индексов и срезы (slice), как и в стандартных типах Python. Для многомерных массивов индексы для различных осей разделяются запятой. Если для многомерного массива указаны индексы не для всех измерений, недостающие заполняются полным срезом (:).

# In[40]:


a = np.arange(10)
a

# In[41]:


a[2:5]

# In[42]:


a[3:8:2]

# In[43]:


A = np.arange(81).reshape(9, -1)
A

# In[44]:


A[2:4]

# In[45]:


A[:, 2:4]

# In[46]:


A[2:4, 2:4]

# In[47]:


A[-1]

# * Также может использоваться индексация при помощи списков индексов (по каждой из осей):

# In[48]:


A = np.arange(81).reshape(9, -1)
A

# In[49]:


A[[2, 4, 5], [0, 1, 3]]

# * Может также применяться логическая индексация (при помощи логических массивов):

# In[50]:


A = np.arange(11)
A

# In[51]:


A[A % 5 != 3]

# In[52]:


A[np.logical_and(A != 7, A % 5 != 3)] # также можно использовать логические операции

# ## Примеры

# In[53]:


A = np.arange(120).reshape(10, -1)
A

# 1. Выбрать все четные строки матрицы A.
# 2. Составить одномерный массив из всех не делящихся на 3 элементов нечетных столбцов А.
# 3. Посчитать сумму диагональных элементов A.

# In[54]:


# Your code here

# ## Зачем?
# 
# Зачем необходимо использовать NumPy, если существуют стандартные списки/кортежи и циклы?
# 
# Причина заключается в скорости работы. Попробуем посчитать скалярное произведение 2 больших векторов:

# In[55]:


import time

A_quick_arr = np.random.normal(size = (1000000,))
B_quick_arr = np.random.normal(size = (1000000,))

A_slow_list, B_slow_list = list(A_quick_arr), list(B_quick_arr)

# In[56]:


%%time
ans = 0
for i in range(len(A_slow_list)):
    ans += A_slow_list[i] * B_slow_list[i]

# In[57]:


%%time
ans = sum([A_slow_list[i] * B_slow_list[i] for i in range(1000000)])

# In[58]:


%%time
ans = np.sum(A_quick_arr * B_quick_arr)

# In[59]:


%%time
ans = A_quick_arr.dot(B_quick_arr)

# # Scikit-Learn 

# Scikit-learn $-$ это библиотека, в которой реализованы основные алгоритмы машинного обучения. Также реализованы методы для подбора гиперпараметров (например, кросс-валидации) и обработки данных. У всех алгоритмов унифицированный интерфейс, так что вы можете легко пробовать различные методы и добавлять свои.
# 
# На семинаре мы рассмотрим стандартный сценарий работы с sklearn. В качестве датасета рассмотрим [__Ames Housing dataset__](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), в котором требуется предсказать стоимость квартиры по ряду признаков (площадь, количество комнат, год постройки и т.д., полнон описание данных можно посмотреть на kaggle).

# TL;DR:
# * Обработка и визуальный анализ данных
# * Обучение линейно регрессии в scikit-learn
# * Подбор гиперпараметров
# * Знакомство с Pipeline

# ----

# In[63]:


! wget https://www.dropbox.com/s/1ymnz6k1p0cezo7/house_prices.csv

# In[2]:


import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')


# ## 1. Загрузка данных

# In[65]:


data = pd.read_csv('./house_prices.csv', index_col='Id')

# Перед тем, как бросаться строить регрессию, почти всегда полезно посмотреть на данные и понять, что они из себя пердставляют (провести анализ).

# * Для начала посмотрим на матрицу корреляций признаков и целевой переменной

# In[66]:


corr = data.corr()
plt.figure(figsize=(15, 11))
sns.heatmap(corr, vmax=.8, square=True, cmap='magma');

# Можно увидеть, что пары `TotalBsmtSF`-`1stFlrSF` и `GarageYrBlt`-`YearBuilt` сильно скоррелированны, что неудивительно, если посмотреть на их описание:
# 
# - __TotalBsmtSF__: Total square feet of basement area
# - __1stFlrSF__: First Floor square feet
# 
# 
# - __YearBuilt__: Original construction date
# - __GarageYrBlt__: Year garage was built
# 
# Линейная зависимость признаков ([мультиколлинеарность](https://ru.wikipedia.org/wiki/Мультиколлинеарность)) приводит к существованию множества эквивалентных решений задачи регрессии, а значит, к нестабильности. Поэтому выбросим эти признаки:

# In[67]:


data.drop(['TotalBsmtSF', 'GarageYrBlt'], 1, inplace=True)

# In[68]:


data

# ## 2. Предобработка

# Для того, чтобы быстро оценить распределение признаков, удобно смотреть на [Box Plot](https://en.wikipedia.org/wiki/Box_plot), который показывает медиану, нижний и верхний квартили, а также максимальное и минимальное значение и выбросы. Кроме того, можно построить гистограммы, чтобы понять вид распределения.

# In[69]:


plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data['SalePrice'])
plt.subplot(1, 2, 2)
sns.distplot(data['SalePrice']);

# Распределение цен имеет достаточно тяжелый правый хвост, в процессе оптимизации ошибка на таких объектах будет штрафоваться сильнее, поэтому прологарифмируем значение целевой переменной, чтобы получить более симметричное распределение.

# In[70]:


data['logSalePrice'] = np.log(data['SalePrice'])
plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data['logSalePrice'])
plt.subplot(1, 2, 2)
sns.distplot(data['logSalePrice']);

# ### Пропуски в данных

# Зачастую в реальных данных не для всех объектов известно значение того или иного признака. Такие объекты нужно обрабатывать прежде чем приступать к обучению. Для каждого признака посмотрим, в какой доле объектов отсутствует значение.

# In[71]:


clmns = data.columns[data.isnull().any()]
missed = pd.DataFrame(data[clmns].isnull().sum().sort_values(ascending=False) / data.shape[0], columns=['% NULL'])
missed

# Некоторые признаки отсутствуют для большого числа объектов, поэтому имеет смысл их выкинуть.

# In[72]:


data.drop(missed[missed['% NULL'] > 0.99].index, 1, inplace=True)

# Оставшиеся пропуски заполним значением медианным значением для действительных признаков и модой для категориальных.

# In[73]:


fill = data.apply(lambda s: s.mode()[0] if s.dtype == 'object' else s.median(), axis=0)
data = data.fillna(value=fill)

# ### Категориальные признаки

# Некоторые признаки в датасете являются категориальными и принимают текстовые значения. Практически все модели в машинном обучении предполагают, что данные представлены в числовом виде, поэтому такие признаки нужно сначала обработать, чтобы обучить регрессию. Один из самых простых и распространенных способов $-$ _one-hot encoding_, создающий для каждого категориального признака, принимающего $K$ различных значений, столько же бинарных признаков, при этом для каждого объекта ровно один из них (соответствующий значению исходного категориального признака на объекте) будет принимать значение 1, остальные будут равны нулю:
# 
# ![](https://i.imgur.com/mtimFxh.png)

# In[74]:


cat_clmns = data.columns[data.dtypes == 'object']
data[cat_clmns].head()

# Для описанного выше преобразования в `scekit-learn` существует объект [`OneHotEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), однако он работает только со значениями типа `int`, поэтому необходимо сначала для каждого признака пронумеровать уникальные значения и использовать эти номера вместо исходных значений признака (в этом нам поможет [`LabelEncoder`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)).

# In[75]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# In[76]:


encoders = [LabelEncoder().fit(data[c]) for c in cat_clmns]
cat_fts = np.stack([enc.transform(data[c]) for enc, c in zip(encoders, cat_clmns)]).T

# In[77]:


cat_fts

# Теперь ко всем таким категориальным признакам применим One-Hot кодирование:

# In[78]:


ohe = OneHotEncoder()
ohe.fit(cat_fts)
cat_ohe = ohe.transform(cat_fts).toarray()
cat_ohe

# In[79]:


real_clmns = data.columns[data.dtypes != 'object']
a = pd.DataFrame(cat_ohe, index=data.index)
data = pd.concat([data[real_clmns], a], axis=1)
data.head()

# ## 3. Обучение

# В данном разделе мы разделим выборку на обучающую и тестовую и построим обычную линейную регрессию.

# При разбиениb выборки на тестовую и обучающую нужно быть внимательными. В данном случае мы решаем не просто задачу регресcии, но прогнозирования, поэтому обучаться на данных за 2010 год и предсказывать значения цен для 2009 года смысла не очень много.

# In[80]:


yrsold = data['YrSold']
sorted(yrsold.unique())

# In[81]:


train_x = data[data['YrSold'] <= 2009]
test_x = data[data['YrSold'] == 2010]

test_y = test_x['SalePrice']
train_y = train_x['SalePrice']

test_x, train_x = test_x.drop(['SalePrice', 'logSalePrice'], 1), train_x.drop(['SalePrice', 'logSalePrice'], 1)
train_idxs, val_idxs = np.where(train_x['YrSold'] < 2009)[0], np.where(train_x['YrSold'] == 2009)[0]

# In[82]:


print('train_size = %.2f' % (train_idxs.shape[0] * 1.0 / data.shape[0]))
print('val_size = %.2f' % (val_idxs.shape[0] * 1.0 / data.shape[0]))
print('test_size = %.2f' % (test_x.shape[0] * 1.0 / data.shape[0]))

# т.к. метрика RMSE штрафует по-разному занижение и завышение цены, в качестве целевой метрики мы будем использовать MAPE, которая показывает ошибку в процентах от истинного значения.
# 
# $$ \text{MAPE} = \dfrac{1}{l} \sum_{i=1}^l \dfrac{|y_i - a(x_i) |}{y_i} $$

# In[10]:


def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Обучим обычную линейную регрессию (обратим вниманием, что в процессе обучения оптимизируется функционал MSE, однако качество на тесте мы будем считать при помощи функционала MAPE) и сравним качество моделей, обученных с логарифмированием целевой переменной и без. При этом качество нужно, очевидно, считать в исходных значениях для обеих моделей.

# In[84]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

lr = LinearRegression()
lr.fit(train_x, train_y)

test_p = lr.predict(test_x)
print('Using Y:')
print('Test MAPE %.2f%%' % mape(test_y, test_p))
print('Test RMSE %.3f' % mean_squared_error(test_y, test_p)**0.5)
print('Test RMSLE %.3f' % mean_squared_log_error(test_y, test_p)**0.5)
print('Test R2 %.3f\n' % r2_score(test_y, test_p))


lr.fit(train_x, np.log(train_y))
test_p = np.exp(lr.predict(test_x))
print('Using logY:')
print('Test MAPE %.2f%%' % mape(test_y, test_p))
print('Test RMSE %.3f' % mean_squared_error(test_y, test_p)**0.5)
print('Test RMSLE %.3f' % mean_squared_log_error(test_y, test_p)**0.5)
print('Test R2 %.3f' % r2_score(test_y, test_p))

# ## 5. Подбор гиперпараметров

# Чтобы избежать переобучения, к функционалу обычной линейной регрессии добавляют регуляризаторы, например $L_2$:
# 
# $$ Q(w) = \text{MSE}(X, w) + \alpha \cdot L_2(w)$$
# 
# $$ L_2(w) = \sum_{i=1}^{d} w_{i}^2 $$
# $$ \text{MSE}(X, w) = \sum_{i=1}^l (w^Tx_i - y_i)^2 $$
# 
# Но встает вопрос о выборе коэффициента регуляризации $\alpha$. Если настраивать этот параметр по обучающей выборке, то лучшим вариантом будет $\alpha = 0$, т.к. именно в этом случае функционал принимает минимальное значение, однако качество на тесте скорее всего упадет, поскольку модель начнет настраиваться под особенности обучающей выборки вместо выявления общих закономерностей в данных. Поэтому нужно пользоваться отложенной выборкой или кросс-валидацией для подбора гиперпараметров.
# 
# 
# ![](images/K-fold_cross_validation.jpeg)
# 

# In[11]:


def mape_scorer(estimator, X_test, y_test):
    return -mape(np.exp(y_test), np.exp(estimator.predict(X_test)))

# In[86]:


from sklearn.linear_model import RidgeCV, Ridge, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

params = {
    'alpha': np.logspace(-6, 3, 20)
}

cv = GridSearchCV(Ridge(), params, cv=[[train_idxs, val_idxs]], scoring=mape_scorer)
cv.fit(train_x, np.log(train_y));

# In[87]:


cv_grid = pd.DataFrame(cv.cv_results_['params'])
cv_grid['val MAPE'] = -cv.cv_results_['mean_test_score']
cv_grid

# In[88]:


plt.title(r'Validation MAPE')
plt.semilogx(cv_grid['alpha'], cv_grid['val MAPE'], label='Validation')

best_l2 = cv.best_params_['alpha']
plt.vlines(best_l2, 8, 11, linestyles='--', color='gray', label='min')
plt.legend()
plt.xlabel('alpha')
plt.ylabel('MAPE, %');

# In[89]:


print('Test MAPE %.3f' % -mape_scorer(cv, test_x, np.log(test_y)))

# Посмотрим на 10 самых значимых признаков и их корреляции с целевой переменной:

# In[90]:


plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('Coef')
w = cv.best_estimator_.coef_
arg = np.argsort(np.abs(w))[-10:]
clmns = train_x.columns.values[arg]
w = w[arg]

x = np.arange(len(w))
plt.barh(x, w)
plt.yticks(x, clmns);

plt.subplot(1, 2, 2)
plt.title('Correlation')
corr = data.corr()
x = np.arange(len(w))
plt.barh(x, corr['logSalePrice'][clmns])
plt.yticks(x, clmns);

# ## 6. Pipeline

# Как мы увидели, процесс построения алгоритма машинного обучения от сырых данных до предсказания можно разделить на несколько этапов:
# 
# * Обработка/фильтрация данных
# * Добавление новых признаков
# * Обучение алгоритма
# 
# Для удобства все эти этапы можно объединить в один [`Pipeline`](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

# In[14]:


data = pd.read_csv('house_prices.csv', index_col='Id')

# In[15]:


from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin,BaseEstimator
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

UNIQUE = {c: data[c][~data[c].isnull()].unique() for c in data.columns[data.dtypes == 'object']}

class MyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, null_threshold):
        self.null_threshold = null_threshold
        self.drop_clmns = []
        self.encoder = None

    def transform(self, x, y=None):
        data = x.copy()
        
        data.drop(['TotalBsmtSF', 'GarageYrBlt'], 1, inplace=True, errors='ignore')
        data.drop(self.drop_clmns, 1, inplace=True)

        data = data.fillna(value=self.fill)

        cat_fts = np.stack([enc.transform(data[c]) for enc, c in zip(self.encoders, self.cat_clmns)]).T
        cat_ohe = self.ohe.transform(cat_fts).toarray()

        real_clmns = data.columns[data.dtypes != 'object']
        cat_ohe = pd.DataFrame(data=cat_ohe, index=data.index)
        data = pd.concat([data[real_clmns], cat_ohe], axis=1)
        
        return data
    
    def fit(self, x, y=None):
        data = x.copy()
        data.drop(['TotalBsmtSF', 'GarageYrBlt'], 1, inplace=True, errors='ignore')
        
        # drop and fill NULLs
        clmns = data.columns[data.isnull().any()]
        missed = pd.DataFrame(data[clmns].isnull().sum().sort_values(ascending=False) / data.shape[0], columns=['% NULL'])
        self.drop_clmns = missed[missed['% NULL'] > self.null_threshold].index
        data.drop(self.drop_clmns, 1, inplace=True)
        
        # Fill remaining NULLs
        self.fill = data.apply(lambda s: s.mode()[0] if s.dtype == 'object' else s.median(), axis=0)
        data = data.fillna(value=self.fill)
        
        # Encode categorical features (str -> int)
        self.cat_clmns = data.columns[data.dtypes == 'object']
        self.encoders = [LabelEncoder().fit(vals) for c, vals in UNIQUE.items() if c in self.cat_clmns]
        
        cat_fts = np.stack([enc.transform(data[c]) for enc, c in zip(self.encoders, self.cat_clmns)]).T

        self.ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe.fit(cat_fts)

        return self
    
    
class MyScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def transform(self, x, y=None):
        x = x.copy()
        x.loc[:, self.clmns] = self.scaler.transform(x[self.clmns])
        return x
    
    def fit(self, x, y=None):
        cond = x.dtypes == 'float'
        self.clmns = cond[cond].index
        self.scaler.fit(x[self.clmns])
        return self

# In[16]:


train_x = data[data['YrSold'] <= 2009]
test_x = data[data['YrSold'] == 2010]

test_y = test_x['SalePrice']
train_y = train_x['SalePrice']

test_x, train_x = test_x.drop('SalePrice', 1), train_x.drop('SalePrice', 1)

# In[17]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



pipe = Pipeline([
    ('process', MyTransformer(0.)),
    ('scale', MyScaler()),
    ('reg', LinearRegression())
])

# ## Зачем?
# 
# * Снижает риск забыть совершить какое-то из преобразований, например, с новыми данными
# * Позволяет довольно просто добавлять новое преобразование в середину процесса
# * Позволяет просто перебирать комбинации методов и их гиперпараметры и сравнивать друг с другом

# Выше мы перебирали значение коэффициента регуляризации при помощи перебора по сетке. Заметим, что тип регуляризации (L1, L2 или без регуляризации), а также порог на долю отсутствующих значений признака, по которому мы удаляли признаки из данных, также являются гиперпараметрами нашей модели и могут быть подобраны по валидационной выборке с помощью уже знакомого `GridSearchCV`.

# In[18]:


# %debug

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

params = [{
    'process__null_threshold': [0., 0.005, 0.1],
    'reg': [LinearRegression()]
}, {
    'process__null_threshold': [0., 0.005, 0.1, 0.5, 0.9],
    'reg': [Ridge(), Lasso()],
    'reg__alpha': np.logspace(-4, 4, 9)
}]
train_idxs, val_idxs = np.where(train_x['YrSold'] < 2009)[0], np.where(train_x['YrSold'] == 2009)[0]
cv = GridSearchCV(pipe, params, cv=[[train_idxs, val_idxs]], scoring=mape_scorer)
cv.fit(train_x, np.log(train_y))

# Соберем итоги валидации в датафрейм, где каждая строчка это один запуск, а в колонках указаны параметры а также скор на валидации.

# In[19]:


cv_grid = pd.DataFrame(cv.cv_results_['params'])
cv_grid['val MAPE'] = -cv.cv_results_['mean_test_score']
cv_grid['reg__alpha'] = cv_grid['reg__alpha'].fillna('0.')
cv_grid['reg'] = cv_grid['reg'].apply(lambda x: x.__class__.__name__)
cv_grid.head()

# С помощью [seaborn](https://seaborn.pydata.org) отобразим полученные результаты.

# In[20]:


sns.catplot(x='reg__alpha', y='val MAPE', data=cv_grid[cv_grid['val MAPE'] < 12], col='reg',
            hue='process__null_threshold', kind='point')

# In[21]:


cv.best_params_

# In[22]:


print('Test MAPE %.3f' % -mape_scorer(cv, test_x, np.log(test_y)))
