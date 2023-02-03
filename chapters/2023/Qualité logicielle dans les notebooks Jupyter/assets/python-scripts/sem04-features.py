#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение 1, ПМИ ФКН ВШЭ
# 
# ## Семинар 4
# 
# ## Подготовка данных

# In[1]:


%pylab inline

import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.datasets import fetch_20newsgroups

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

import warnings
warnings.simplefilter("ignore")

# ## Работа с текстовыми данными
# 
# Как правило, модели машинного обучения действуют в предположении, что матрица "объект-признак" является вещественнозначной, поэтому при работе с текстами сперва для каждого из них необходимо составить его признаковое описание. Для этого широко используются техники векторизации, tf-idf и пр. Рассмотрим их на примере датасета новостей о разных топиках.
# 
# Сперва загрузим данные:

# In[2]:


data = fetch_20newsgroups(subset='all', categories=['comp.graphics', 'sci.med'])

# Данные содержат тексты новостей, которые надо классифицировать на два раздела: компьютерные науки и медицинские.

# In[3]:


data['target_names']

# In[4]:


texts = data['data']
target = data['target']

# Например:

# In[5]:


texts[0]

# In[6]:


data['target_names'][target[0]]

# In[7]:


texts_train, texts_test, y_train, y_test = train_test_split(
    texts, target, test_size=0.2, random_state=10
)

# ### Bag-of-words
# 
# Самый очевидный способ формирования признакового описания текстов — векторизация. Пусть у нас имеется коллекция текстов $D = \{d_i\}_{i=1}^l$ и словарь всех слов, встречающихся в выборке $V = \{v_j\}_{j=1}^d.$ В этом случае некоторый текст $d_i$ описывается вектором $(x_{ij})_{j=1}^d,$ где
# $$x_{ij} = \sum_{v \in d_i} [v = v_j].$$
# 
# Таким образом, текст $d_i$ описывается вектором количества вхождений каждого слова из словаря в данный текст.

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(encoding='utf8')
_ = vectorizer.fit(texts_train)
len(vectorizer.vocabulary_)

# Результатом является разреженная матрица.

# In[9]:


vectorizer.transform(texts_train[:1])

# In[10]:


print(vectorizer.transform(texts_train[:1]).indices)
print(vectorizer.transform(texts_train[:1]).data)

# Подберем оптимальные гиперпараметры по сетке и обучим модель. Учить будем логистическую регрессию, так как мы решаем задачу бинарной классификации, а для оценки качества будем использовать точность 
# $$
# Accuracy(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N [\hat{y_i} = y_i]
# $$

# In[11]:


def train_model(X_train, y_train):
    alphas = np.logspace(-1, 3, 10)
    searcher = GridSearchCV(LogisticRegression(), [{'C': alphas, 'max_iter': [500]}],
                            scoring='accuracy', cv=5, n_jobs=-1)
    searcher.fit(X_train, y_train)

    best_alpha = searcher.best_params_["C"]
    print("Best alpha = %.4f" % best_alpha)
    
    model = LogisticRegression(C=best_alpha, max_iter=500)
    model.fit(X_train, y_train)
    
    return model

# In[12]:


X_train = vectorizer.transform(texts_train)
X_test = vectorizer.transform(texts_test)

# In[13]:


model = train_model(X_train, y_train)

print("Train accuracy = %.4f" % accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy = %.4f" % accuracy_score(y_test, model.predict(X_test)))

# ### TF-IDF
# 
# Ещё один способ работы с текстовыми данными — [TF-IDF](https://en.wikipedia.org/wiki/Tf–idf) (**T**erm **F**requency–**I**nverse **D**ocument **F**requency). Рассмотрим коллекцию текстов $D$.  Для каждого уникального слова $t$ из документа $d \in D$ вычислим следующие величины:
# 
# 1. Term Frequency – количество вхождений слова в отношении к общему числу слов в тексте:
# $$\text{tf}(t, d) = \frac{n_{td}}{\sum_{t \in d} n_{td}},$$
# где $n_{td}$ — количество вхождений слова $t$ в текст $d$.
# 1. Inverse Document Frequency
# $$\text{idf}(t, D) = \log \frac{\left| D \right|}{\left| \{d\in D: t \in d\} \right|},$$
# где $\left| \{d\in D: t \in d\} \right|$ – количество текстов в коллекции, содержащих слово $t$.
# 
# Тогда для каждой пары (слово, текст) $(t, d)$ вычислим величину:
# 
# $$\text{tf-idf}(t,d, D) = \text{tf}(t, d)\cdot \text{idf}(t, D).$$
# 
# Отметим, что значение $\text{tf}(t, d)$ корректируется для часто встречающихся общеупотребимых слов при помощи значения $\text{idf}(t, D)$.

# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(encoding='utf8')
_ = vectorizer.fit(texts_train)
len(vectorizer.vocabulary_)

# На выходе получаем разреженную матрицу.

# In[15]:


vectorizer.transform(texts_train[:1])

# In[16]:


print(vectorizer.transform(texts[:1]).indices)
print(vectorizer.transform(texts[:1]).data)

# Подберем оптимальные гиперпараметры по сетке и обучим модель.

# In[17]:


X_train = vectorizer.transform(texts_train)
X_test = vectorizer.transform(texts_test)

# In[18]:


model = train_model(X_train, y_train)

print("Train accuracy = %.4f" % accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy = %.4f" % accuracy_score(y_test, model.predict(X_test)))

# ## Стемминг и лемматизация
# 
# Заметим, что одно и то же слово может встречаться в различных формах (например, "сотрудник" и "сотрудника"), но описанные выше методы интерпретируют их как различные слова, что делает признаковое описание избыточным. Устранить эту проблему можно при помощи **лемматизации** и **стемминга**.
# 
# ### Стемминг
# 
# [**Stemming**](https://en.wikipedia.org/wiki/Stemming) –  это процесс нахождения основы слова. В результате применения данной процедуры однокоренные слова, как правило, преобразуются к одинаковому виду.
# 
# **Примеры стемминга:**
# 
# | Word        | Stem           |
# | ----------- |:-------------:|
# | вагон | вагон |
# | вагона | вагон |
# | вагоне | вагон |
# | вагонов | вагон |
# | вагоном | вагон |
# | вагоны | вагон |
# | важная | важн |
# | важнее | важн |
# | важнейшие | важн |
# | важнейшими | важн |
# | важничал | важнича |
# | важно | важн |
# 
# [Snowball](http://snowball.tartarus.org/) – фрэймворк для написания алгоритмов стемминга. Алгоритмы стемминга отличаются для разных языков и используют знания о конкретном языке – списки окончаний для разных чистей речи, разных склонений и т.д. Пример алгоритма для русского языка – [Russian stemming](http://snowballstem.org/algorithms/russian/stemmer.html).

# In[19]:


import nltk
stemmer = nltk.stem.snowball.RussianStemmer()

# In[20]:


print(stemmer.stem(u'машинное'), stemmer.stem(u'обучение'))

# Попробуем применить **стемминг** для предобработки текста перед векторизацией. Векторизовывать будем с помощью **tf-idf**, так как такой метод показал лучшее качество.

# In[21]:


stemmer = nltk.stem.snowball.EnglishStemmer()

def stem_text(text, stemmer):
    tokens = text.split()
    return ' '.join(map(lambda w: stemmer.stem(w), tokens))

stemmed_texts_train = []
for t in tqdm(texts_train):
    stemmed_texts_train.append(stem_text(t, stemmer))
    
stemmed_texts_test = []
for t in tqdm(texts_test):
    stemmed_texts_test.append(stem_text(t, stemmer))

# In[22]:


print(texts_train[1])

# In[23]:


print(stemmed_texts_train[1])

# In[24]:


vectorizer = TfidfVectorizer(encoding='utf8')
_ = vectorizer.fit(stemmed_texts_train)
len(vectorizer.vocabulary_)

# In[25]:


X_train = vectorizer.transform(stemmed_texts_train)
X_test = vectorizer.transform(stemmed_texts_test)

# In[26]:


model = train_model(X_train, y_train)

print("Train accuracy = %.4f" % accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy = %.4f" % accuracy_score(y_test, model.predict(X_test)))

# ### Лемматизация
# 
# [Лемматизация](https://en.wikipedia.org/wiki/Lemmatisation) — процесс приведения слова к его нормальной форме (**лемме**):
# - для существительных — именительный падеж, единственное число;
# - для прилагательных — именительный падеж, единственное число, мужской род;
# - для глаголов, причастий, деепричастий — глагол в инфинитиве.

# Например, для русского языка есть библиотека pymorphy2.

# In[27]:


import pymorphy2
morph = pymorphy2.MorphAnalyzer()

# In[28]:


morph.parse('играющих')[0]

# Сравним работу стеммера и лемматизатора на примере:

# In[29]:


stemmer = nltk.stem.snowball.RussianStemmer()
print(stemmer.stem('играющих'))

# In[30]:


print(morph.parse('играющих')[0].normal_form)

# Для английского языка будем пользоваться лемматизатором из библиотеки **nltk**.

# In[31]:


from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text, stemmer):
    tokens = text.split()
    return ' '.join(map(lambda w: lemmatizer.lemmatize(w), tokens))

lemmatized_texts_train = []
for t in tqdm(texts_train):
    lemmatized_texts_train.append(lemmatize_text(t, stemmer))
    
lemmatized_texts_test = []
for t in tqdm(texts_test):
    lemmatized_texts_test.append(lemmatize_text(t, stemmer))

# In[32]:


print(lemmatized_texts_train[1])

# In[33]:


print(stemmed_texts_train[1])

# Лемматизируем наш корпус применим tf-idf векторизацию и обучим модель.

# In[34]:


vectorizer = TfidfVectorizer(encoding='utf8')
_ = vectorizer.fit(lemmatized_texts_train)
len(vectorizer.vocabulary_)

# In[35]:


X_train = vectorizer.transform(lemmatized_texts_train)
X_test = vectorizer.transform(lemmatized_texts_test)

# In[36]:


model = train_model(X_train, y_train)

print("Train accuracy = %.4f" % accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy = %.4f" % accuracy_score(y_test, model.predict(X_test)))

# In[ ]:




# ## Трансформация признаков и целевой переменной

# Разберёмся, как может влиять трансформация признаков или целевой переменной на качество модели. 

# ### Логарифмирование 
# 
# Воспользуется датасетом с ценами на дома, с которым мы уже сталкивались ранее ([House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)).

# In[37]:


!wget  -O 'train_sem4.csv' -q 'https://www.dropbox.com/s/syfy4lb6xb7wdlx/_train_sem4.csv?dl=0'

# In[2]:


data = pd.read_csv('train_sem4.csv')

data = data.drop(columns=["Id"])
y = data["SalePrice"]
X = data.drop(columns=["SalePrice"])

# Оставим только числовые признаки, пропуски заменим средним значением.

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)

numeric_data = X_train.select_dtypes([np.number])
numeric_data_mean = numeric_data.mean()
numeric_features = numeric_data.columns

X_train = X_train.fillna(numeric_data_mean)[numeric_features]
X_test = X_test.fillna(numeric_data_mean)[numeric_features]

# Посмотрим на распределение целевой переменной 

# In[4]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.distplot(y, label='target')
plt.title('target')

plt.show()

# Видим, что распределения несимметричные с тяжёлыми правыми хвостами.
# 
# Если разбирать линейную регрессию c MSE ошибкой с [вероятностной](https://github.com/esokolov/ml-course-hse/blob/master/2018-fall/seminars/sem04-linregr.pdf) точки зрения, то можно получить, что шум должен быть распределён нормально. Поэтому лучше, когда целевая переменная распределена также нормально.
# 
# Если прологарифмировать целевую переменную, то её распределение станет больше похоже на нормальное:

# In[5]:


sns.distplot(np.log(y+1), label='target')
plt.show()

# Сравним качество линейной регрессии в двух случаях:
# 1. Целевая переменная без изменений.
# 2. Целевая переменная прологарифмирована.
# 
# Не забудем вернуть во втором случае взять экспоненту от предсказаний!

# In[6]:


def train_model(X_train, y_train):
    alphas = np.logspace(-2, 3, 10)
    searcher = GridSearchCV(Ridge(), [{'alpha': alphas}],
                            scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
    searcher.fit(X_train, np.log(y_train+1))

    best_alpha = searcher.best_params_["alpha"]
    print("Best alpha = %.4f" % best_alpha)

    return searcher.best_estimator_.fit(X_train, y_train)

# In[7]:


model = train_model(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# In[8]:


model = train_model(X_train, np.log(y_train+1))

y_pred_train = np.exp(model.predict(X_train)) - 1
y_pred_test = np.exp(model.predict(X_test)) - 1

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# Попробуем аналогично логарифмировать один из признаков, имеющих также смещённое распределение (этот признак был вторым по важности!)

# In[45]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.distplot(y, label='GrLivArea')
plt.title('GrLivArea')

plt.show()

# In[14]:


X_train.GrLivArea = np.log(X_train.GrLivArea + 1)
X_test.GrLivArea = np.log(X_test.GrLivArea + 1)

# In[15]:


model = train_model(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# In[16]:


model = train_model(X_train, np.log(y_train+1))

y_pred_train = np.exp(model.predict(X_train)) - 1
y_pred_test = np.exp(model.predict(X_test)) - 1

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# Как видим, логарифмирование признака уменьшило ошибку на тренировочной выборке, но на тестовой выборке ошибка увеличилась.

# ## Категориальные признаки

# In[74]:


# ! pip install category_encoders

# In[17]:


from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)

# In[19]:


numeric = list(X_train.select_dtypes(include=np.number).columns)
categorical = list(X_train.dtypes[X_train.dtypes == "object"].index)

# In[20]:


X_train[categorical] = X_train[categorical].fillna("NotGiven")
X_test[categorical] = X_test[categorical].fillna("NotGiven")

numeric_data_mean = X_train[numeric_features].mean()
X_train[numeric] = X_train[numeric].fillna(numeric_data_mean)
X_test[numeric] = X_test[numeric].fillna(numeric_data_mean)

# ### One Hot Encoder

# In[26]:


column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('ohe', column_transformer),
    ('regression', Ridge())
])

# In[27]:


alphas = np.logspace(-2, 5, 10)
searcher = GridSearchCV(pipeline, [{'regression__alpha': alphas}],
                        scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
searcher.fit(X_train, np.log(y_train+1))

best_alpha = searcher.best_params_["regression__alpha"]
print("Best alpha = %.4f" % best_alpha)

# In[28]:


model = searcher.best_estimator_

y_pred_train = np.exp(model.predict(X_train)) - 1
y_pred_test = np.exp(model.predict(X_test)) - 1

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# In[31]:


print('Features before OHE:', len(numeric) + len(categorical))
print('Features after OHE:', len(model['regression'].coef_))

# Видим, что OHE кодирование признаков привело к колоссальному переобучению, попробуем что-нибудь умнее.

# ### Счетчики (mean target encoding)

# При кодировании категориального признака каждое его значение будем заменять на среднее значение целевой переменной для всех объектов с такой категорией.
# 
# $$
# g_j(x, X) = \frac{\sum_{i=1}^{\ell}\left[f_j(x)=f_j\left(x_i\right)\right] y_i}{\sum_{i=1}^{\ell}\left[f_j(x)=f_j\left(x_i\right)\right]}
# $$

# In[32]:


column_transformer = ColumnTransformer([
    ('te', TargetEncoder(smoothing=1.0), categorical)
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('scale', column_transformer),
    ('regression', Ridge())
])

# In[33]:


alphas = np.logspace(-2, 3, 10)
searcher = GridSearchCV(pipeline, [{'regression__alpha': alphas}],
                        scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)
searcher.fit(X_train, np.log(y_train+1))

best_alpha = searcher.best_params_["regression__alpha"]
print("Best alpha = %.4f" % best_alpha)

# In[35]:


model = searcher.best_estimator_

y_pred_train = np.exp(model.predict(X_train)) - 1
y_pred_test = np.exp(model.predict(X_test)) - 1

print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train) ** 0.5)
print("Test RMSE = %.4f" % mean_squared_error(y_test, y_pred_test) ** 0.5)

# Гораздо лучше!

# ## Транзакционные данные

# Напоследок посмотрим, как можно извлекать признаки из транзакционных данных. 
# 
# Транзакционные данные характеризуются тем, что есть много строк, характеризующихся моментов времени и некоторым числом (суммой денег, например). При этом если это банк, то каждому человеку принадлежит не одна транзакция, а чаще всего надо предсказывать некоторые сущности для клиентов. Таким образом, надо получить признаки для пользователей из множества их транзакций. Этим мы и займёмся.
# 
# Для примера возьмём данные [отсюда](https://www.kaggle.com/regivm/retailtransactiondata/). Задача детектирования фродовых клиентов.

# In[60]:


!wget  -O 'Retail_Data_Response.csv' -q 'https://www.dropbox.com/s/le9icl9syo22thh/Retail_Data_Response.csv?dl=0'
!wget  -O 'Retail_Data_Transactions.csv' -q 'https://www.dropbox.com/s/obsxryxpfsdz3ut/Retail_Data_Transactions.csv?dl=0'

# In[61]:


customers = pd.read_csv('Retail_Data_Response.csv')
transactions = pd.read_csv('Retail_Data_Transactions.csv')

# In[62]:


customers.head()

# In[63]:


transactions.head()

# In[64]:


transactions.trans_date = transactions.trans_date.apply(
    lambda x: datetime.datetime.strptime(x, '%d-%b-%y'))

# Посмотрим на распределение целевой переменной:

# In[65]:


customers.response.mean()

# Получаем примерно 1 к 9 положительных примеров. Если такие данные разбивать на части для кросс валидации, то может получиться так, что в одну из частей попадёт слишком мало положительных примеров, а в другую — наоборот. На случай такого неравномерного баланса классов есть StratifiedKFold, который бьёт данные так, чтобы баланс классов во всех частях был одинаковым.

# In[66]:


from sklearn.model_selection import StratifiedKFold

# Когда строк на каждый объект много, можно считать различные статистики. Например, средние, минимальные и максимальные суммы, потраченные клиентом, количество транзакий, ...

# In[67]:


agg_transactions = transactions.groupby('customer_id').tran_amount.agg(
    ['mean', 'std', 'count', 'min', 'max']).reset_index()

data = pd.merge(customers, agg_transactions, how='left', on='customer_id')

data.head()

# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(
    LogisticRegression(solver='newton-cg'),
    X=data.drop(['customer_id', 'response'], axis=1),
    y=data.response,
    cv=StratifiedKFold(n_splits=3),
    scoring='roc_auc'))

# Но каждая транзакция снабжена датой! Можно посчитать статистики только по свежим транзакциям. Добавим их.

# In[70]:


transactions.trans_date.min(), transactions.trans_date.max()

# In[71]:


agg_transactions = transactions.loc[transactions.trans_date.apply(
    lambda x: x.year == 2014)].groupby('customer_id').tran_amount.agg(
    ['mean', 'std', 'count', 'min', 'max']).reset_index()

# In[72]:


data = pd.merge(data, agg_transactions, how='left', on='customer_id', suffixes=('', '_2014'))
data = data.fillna(0)

# In[73]:


np.mean(cross_val_score(
    LogisticRegression(solver='newton-cg'),
    X=data.drop(['customer_id', 'response'], axis=1),
    y=data.response,
    cv=StratifiedKFold(n_splits=3),
    scoring='roc_auc'))

# Можно также считать дату первой и последней транзакциями пользователей, среднее время между транзакциями и прочее.
