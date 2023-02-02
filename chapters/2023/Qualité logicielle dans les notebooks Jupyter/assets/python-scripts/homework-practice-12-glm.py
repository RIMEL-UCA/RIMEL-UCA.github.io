#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# ## Практическое задание 12
# 
# ### Общая информация
# 
# Дата выдачи: 9.05.2020
# 
# Жёсткий дедлайн: 9.06.2020 23:59MSK

# ### О задании
# 
# Мы будем решать задачу предсказания опасных событий для страховой компании: [Liberty Mutual Group: Property Inspection Prediction](https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction). Обучающая выборка состоит из засекреченных признаков целого и строкового типов. Целевой переменной являются счётчики $y \in \mathbb{Z}_+$.
# 
# Работа состоит из следующих пунктов:
# * Предобработать данные [1 балл]
# * Написать свой алгоритм прогнозирования событий [3 балла]
# * Настроить линейные методы из библиотеки StatsModels для решения задачи [1 балл]
# * Настроить бустинг из библиотеки lightgbm для решения задачи [2 балла]
# 
# Задания является дополнительным, то есть само по себе не учитывается в накопленной оценке. Все полученные за задание баллы являются бонусными, то есть их можно прибавить к оценке за любое теоретическое или практическое домашнее задание из курса.
# 
# 
# ### Оценивание и штрафы
# Каждая из задач имеет определенную «стоимость» (указана в скобках около задачи). Максимально допустимая оценка за работу — 7 баллов.
# 
# Сдавать задание после указанного срока сдачи нельзя. При выставлении неполного балла за задание в связи с наличием ошибок на усмотрение проверяющего предусмотрена возможность исправить работу на указанных в ответном письме условиях.
# 
# Задание выполняется самостоятельно. «Похожие» решения считаются плагиатом и все задействованные студенты (в том числе те, у кого списали) не могут получить за него больше 0 баллов (подробнее о плагиате см. на странице курса). Если вы нашли решение какого-то из заданий (или его часть) в открытом источнике, необходимо указать ссылку на этот источник в отдельном блоке в конце вашей работы (скорее всего вы будете не единственным, кто это нашел, поэтому чтобы исключить подозрение в плагиате, необходима ссылка на источник).
# 
# Неэффективная реализация кода может негативно отразиться на оценке.
# 
# 
# ### Формат сдачи
# Для сдачи задания переименуйте получившийся файл *.ipynb в соответствии со следующим форматом: homework-practice-12-Username.ipynb, где Username — ваша фамилия и имя на латинице именно в таком порядке (например, homework-practice-12-IvanovIvan.ipynb). 
# 
# Далее отправьте этот файл в AnyTask.
# 
# Для удобства проверки самостоятельно посчитайте свою максимальную оценку (исходя из набора решенных задач) и укажите ниже.

# ** Оценка:** ...

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.optimize import minimize
from lightgbm import LGBMModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

pd.set_option('max_rows', 10)
pd.set_option('max_columns', None)
plt.style.use('bmh')


# __Задание 1 (1 балл).__ Загрузка и предобработка данных.

# In[ ]:


data = pd.read_csv('data.csv', index_col='Id')
data.head()

# In[ ]:


data.shape

# Выделим категориальные и числовые признаки:

# In[ ]:


categorical, numerical = list(), list()
for col in data.columns[1:]:
    if isinstance(data.loc[1, col], str):
        categorical.append(col)
    if isinstance(data.loc[1, col], np.int64):
        numerical.append(col)

# In[ ]:


data[categorical].nunique().values

# In[ ]:


data[numerical].nunique().values

# Нарисуем априорное распределение ответов $p(y)$:

# In[ ]:


data['Hazard'].plot(kind='hist', figsize=(10, 4), bins=40)
plt.show()

# В обобщённых линейных моделях, как и в машинном обучении в целом, мы проводим основную работу с апостериорным распределением $p(y|x)$, ведь именно в нём заключается информация о конкретной задаче. Здесь же мы знаем, что количество несчастных случаев во многом подчиняется распределению Пуассона, поэтому будем стараться моделировать именно его.

# В выборке могут присутствовать шумовые признаки. Попробуем простейшим способом избавиться от них.
# 
# Исследуйте абсолютное значение корреляции:
# * Признаков и отклика
# * Признаков и логарифма отклика

# In[ ]:


corrs = [data[col].corr(data['Hazard']) for col in numerical]
corrs = pd.DataFrame(np.abs(corrs), numerical, 
    ['abs corr with target']).sort_values('abs corr with target')
corrs.plot(kind='bar', figsize=(10, 4))
plt.show()

# In[ ]:


# corr with log(target)
# place your code here

# Уберите несколько наиболее неинформативных признаков. Лучше сделать это число гиперпараметром и потом настраивать его по функционалу качества. Может быть разумно также исследовать взаимосвязь признаков с логарифмом целевой переменной, поскольку мы предполагаем, что она неотрицательна и имеет распределение Пуассона.

# In[ ]:


# place your code here

# Выделим столбец целевой переменной из наших данных. Множество значений случайной величины с распределением Пуассона начинается с нуля $\{0, 1, 2...\}$, поэтому вычтем единицу из целевой переменной. На самом деле, помимо прочего это приводит к существенному росту качества на валидации.

# In[ ]:


objects = data.loc[:, 'T1_V1':]
labels = data['Hazard'] - 1

# Сделаем бинарное кодирование категориальных признаков:

# In[ ]:


transformer = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical),
    ('cat', OneHotEncoder(sparse=False), categorical)
])
objects_encoded = transformer.fit_transform(objects)

# Разделим выборку на обучающую и тестовую:

# In[ ]:


train_objects, test_objects, train_labels, test_labels = train_test_split(
    objects_encoded, labels.values, random_state=1, test_size=0.2
)

# __Задание 2 (3 балла).__ Обучение регрессии с распределением Пуассона.
# 
# Будем считать, что апостериорное распределение — это распределение Пуассона:
# 
# $$p(y|\lambda(x)) = \frac{e^{-\lambda(x)}\lambda(x)^y}{y!}.$$
# 
# Реализуйте функции для вычисления функционала качества (через метод максимального правдоподобия) и его градиентов — они были выведены на [семинаре](https://github.com/esokolov/ml-course-hse/blob/master/2017-spring/seminars/sem22-glm.pdf).
# 
# Численные алгоритмы должны работать по возможности быстро, поэтому циклов быть не должно, и все операции должны быть векторными. Дальше мы будем использовать эту функцию в качестве аргумента другой функции. Можете попробовать добавить в модель регуляризатор.

# In[ ]:


def oracle(w, X, y):
    """
    :param w: weights
    :param X: features
    :param y: target
    :yield: loss, grad
    """
    
    # place your code here
    return loss, grad

# Добавьте к признакам столбец единиц, чтобы учесть вектор сдвига. Это важно. Библиотечные алгоритмы уже учитывают это внутри себя, поэтому им на вход нужно подавать исходную выборку.

# In[ ]:


train_objects_bias = None
test_objects_bias = None

# Дальше воспользуйтесь функцией _scipy.optimize.minimize_, в ней реализовано множество градиентных методов оптимизации. Рекомендуется взять [L-BFGS-B](http://www.machinelearning.ru/wiki/images/6/65/MOMO17_Seminar6.pdf). Начальное приближение весов $w^{(0)}$ возьмите из стандартного нормального распределения как _np.random.randn_. Возможно, придётся запустить функцию несколько раз, прежде чем начальное приближение окажется удачным, и алгоритм покажет хороший результат. 
# 
# Сделайте прогноз для тестовых объектов. В качестве прогноза мы будем брать матожидание распределения $p(y | \lambda(x))$ в данной точке, поэтому не забудьте взять экспоненту от выхода линейной модели, предсказывающей значение натурального параметра. Измерьте качество с помощью коэффициента Джини (чем выше – тем лучше) — именно он является целевой метрикой в соревновании.

# In[ ]:


def gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

# In[ ]:


# place your code here
poisson_results = minimize(oracle, None, args=(None, None), method=None, jac=True) #change parameters
pred_labels = None # w = poisson_results.x
gini(test_labels, pred_labels)

# __Задание 3 (1 балл).__ Линейные модели из коробки.
# 
# Запустите базовую линейную регрессию *sklearn.linear_model.LinearRegression* для предсказания логарифма целевой переменной и измерьте качество. Сравните полученный результат с работой вашего алгоритма.

# In[ ]:


# place your code here

# Познакомимся теперь с библиотекой [StatsModels](http://www.statsmodels.org/dev/index.html). Она предназначена скорее для описательной статистики, проверки гипотез и построения доверительных интервалов, чем для прогнозирования, — в отличие от scikit-learn, который создан специально для решения задач машинного обучения. В то же время в StatsModels реализован очень сильный алгоритм прогнозирования временных рядов – [SARIMAX](http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html), который при правильной настройке работает очень хорошо и занимает первые места в конкурсах ([Запись трансляции ML тренировки 03.02.18 | TrainMyData Ascott](https://www.youtube.com/watch?v=9MQEEyYDCQc&t=1101s)). 
# 
# Мы же попробуем обучить обобщённые линейные модели (модуль [GLM](http://www.statsmodels.org/dev/glm.html)) с различными вероятностными распределениями. Запустите алгоритм _sm.GLM_ на нескольких распределениях family, посмотрите на качество и проинтерпретируйте результаты. Синтаксис StatsModels немного отличается от scikit-learn тем, что здесь объекты и метки задаются в конструкторе модели, метод _fit()_ идёт без аргументов, и после обучения сохраняется новая модель с результатами и методом _predict_.

# __Gaussian__

# In[ ]:


#place your code here
gaussian_sm_model = None
gaussian_sm_results = None
pred_labels = gaussian_sm_results.predict(test_objects)
gini(test_labels, pred_labels)

# __Poisson__

# In[ ]:


# place your code here

# __NegativeBinomial (Pascal)__

# In[ ]:


# place your code here

# Как видно, качество в последнем случае получилось лучше. На практике чаще используют именно отрицательное Биномиальное распределение. Оно является обобщением геометрического распределения и даёт некоторую свободу в выборе среднего и дисперсии для $p(y|x)$, тогда как в распределении Пуассона среднее и дисперсия совпадают. Если вам будет интересно, можете прочитать подробнее на вики [NegativeBinomial](https://en.m.wikipedia.org/wiki/Negative_binomial_distribution).

# __Задание 4. (2 балла)__ Прогнозирование с помощью бустинга.

# Что бы мы ни делали, бустинг по-прежнему остаётся лучшим подходом для широкого круга задач — особенно если мы не работаем со структурированными данными вроде последовательностей или картинок. Бустинг хоть и сложный по своей структуре алгоритм, но вероятностные распределения не чужды и ему. Запустите LGBMModel, используя классическую регрессию и регрессию Пуассона. Настройте параметры, чтобы добиться наилучшего качества. В особенности обратите внимание на *objective*, *n_estimators*, *num_leaves* и *colsample_bytree*.

# In[ ]:


# objective='regression'

# In[ ]:


# objective='poisson'

# In[ ]:



