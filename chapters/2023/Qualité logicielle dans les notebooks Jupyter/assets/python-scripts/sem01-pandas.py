#!/usr/bin/env python
# coding: utf-8

# ## Машинное обучение 1, ПМИ ФКН ВШЭ
# 
# ## Семинар 1
# 
# ## Работа с табличными данными

# В машинном обучении, как правило, всё сводится к анализу табличных данных. Начинать мы можем с большого количества сложных таблиц, изображений, текстов или ещё чего-то непростого, но в итоге всё это обычно сводится к одной таблице, где каждый объект описывается набором признаков. Поэтому важно уметь работать с таблицами.
# 
# А ещё есть некоторые исследования, показывающие, что в решении задачи интеллектуального анализа данных обычно 20% времени уходит на построение моделей и прочую интересную работу, связанную с тем, что рассказывается у нас на лекциях, а 80% времени специалисты тратят на подготовку и обработку данных. Сюда входит формирование признаков, устранение выбросов и пропусков и т.д. И это тоже, по сути дела, манипуляции с таблицами.
# 
# Вывод: важно уметь работать с табличными данными. В Python для этого есть библиотека pandas, которую мы и будем сегодня изучать.

# Чаще всего название библиотеки при импорте сокращают до "pd":

# In[1]:


import pandas as pd

# ### Распределение студентов по элективам

# Разумно тренироваться на реальных сложных данных. А что может быть более сложным, чем данные, сгенерированные студентами?
# 
# Сегодня мы будем работать с анкетами студентов ПМИ 2017 и 2018 годов набора о том, на какие курсы по выбору они хотят попасть. Данные были анонимизированы: ФИО захешированы с солью, к рейтингам добавлен случайный шум.
# 
# *Вопрос: как можно деанонимизировать данные после манипуляций, которые мы проделали? А как бы вы предложили провести анонимизацию?*

# У нас есть 2 таблицы (для 3 и 4 курса):
# 
#     – 'Timestamp': время получения ответов
#     – 'ID': ID студента (может повторяться, если студент больше одного раза заполнял анкету) 
#     – 'Рейтинг': Кредитно-рейтинговая сумма студента (грубо говоря, сумма оценок студента по всем его дисциплинам с весами — чем дольша шла дисциплина, тем больше вес; подробности тут: https://www.hse.ru/studyspravka/rate/)
#     – 'Группа (в формате 182)': Номер группы
#     – 'МИ?': 1, если студент распределился на специализацию МИ, или NaN в противном случае (признак важен, поскольку студенты МИ берут осенью два курса по выбору, а студенты остальных специализаций только один)
#     – 'Осенний курс по выбору, приоритет 1'
#     – 'Осенний курс по выбору, приоритет 2' 
#     – 'Осенний курс по выбору, приоритет 3'
#     – 'Весенний курс по выбору, приоритет 1'
#     – 'Весенний курс по выбору, приоритет 2'
#     – 'Весенний курс по выбору, приоритет 3'
#     – 'Вы заполняете анкету в первый раз?': "Да" или "Нет"
#    
# Дополнительные столбцы для 4ого курса:
#     
#     – 'Группа (в формате 173)': Номер группы
#     – 'blended-курс': Выбор blended-курса (кол-во мест неограничено)

# Загрузим данные (обратите внимание, что мы легко читаем xlsx-файлы):

# In[2]:


!wget  -O 'data_3_course.xlsx' -q 'https://www.dropbox.com/s/ysxs5srafoyxknb/_data_3_course.xlsx?dl=1'
!wget  -O 'data_4_course.xlsx' -q 'https://www.dropbox.com/s/hfg2mzmvcivtxqk/_data_4_course.xlsx?dl=1'

# In[3]:


data3 = pd.read_excel('data_3_course.xlsx')
data4 = pd.read_excel('data_4_course.xlsx')

# In[4]:


data3

# Посмотрим размер таблицы:

# In[5]:


data3.shape

# Для начала будем работать с одной таблицей для 3 курса. Теперь данные хранятся в переменной ```data3```, которая имеет тип [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html):

# In[6]:


type(data3)

# DataFrame можно создать и вручную:

# In[7]:


df = pd.DataFrame({
    'AAA': [4, 5, 6, 7], 
    'BBB': [10, 20, 30, 40], 
    'CCC': [100, 50, 'E', -50]
})
df

# DataFrame можно частично отобразить в jupyter-ноутбуке с помощью методов ```head```(первые строки) и ```sample```(случайные строки):

# In[8]:


data3.head(2)

# In[9]:


data3.sample(5)

# Можно немного залезть во внутренности Jupyter, чтобы отобразить сразу несколько таблиц:

# In[10]:


from IPython.display import display
display(data3.sample(3)), display(data3.sample(3))

# Если вам очень хочется отобразить все строки таблицы, то можно сделать так:

# In[11]:


# pd.options.display.max_rows = 999
# data3

# In[12]:


# pd.options.display.max_rows = 20

# DataFrame, по сути, является двумерной таблицей с набором полезных методов. Давайте рассмотрим некоторые из них.
# 
# ```columns``` — возвращает названия колонок
# 
# ```dtypes``` — типы колонок
# 
# 

# In[13]:


data3.columns

# In[14]:


data3.dtypes

# В DataFrame есть несколько способов для обращения к строкам, столбцам и отдельным элементам таблицы: квадратные скобки и методы ```loc```, ```iloc```.
# 
# Как обычно, лучший источник знаний об этом — [документация](https://pandas.pydata.org/docs/user_guide/indexing.html).
# Ниже краткое содержание.

# В метод ```loc``` можно передать значение индекса (число, которое стоит в колонке index) строки, чтобы получить эту строку:

# In[15]:


data3.loc[2]

# Получили отдельную строчку в виде объекта класса [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html):

# In[16]:


type(data3.loc[2])

# А с помощью срезов можно выбрать часть таблицы:

# In[17]:


data3.loc[2:4]

# Срез в ```loc``` производится по index и включает в себя последний элемент.
# 
# Метод ```iloc``` действует похожим образом, но он индексирует элементы не по index, а по порядку в таблице (который может отличаться от index). Например:

# In[18]:


subset = data3.sample(5)
subset

# Если же вызвать просто ```loc[2]```, то получим ошибку:
# 

# In[19]:


# subset.loc[2]

# С помощью ```iloc``` тоже можно делать срезы, но в них последний элемент не включается (как и в обычных срезах в Python, **в отличие от loc**):

# In[20]:


subset.iloc[2:4]

# Срезы можно брать не только по строкам, но и по столбцам. Обратите внимание на различия индексации столбцов в ```loc``` и ```iloc```:

# In[21]:


data3.iloc[2:4, 2:6]

# In[22]:


data3.loc[2:4, 'Рейтинг':'Осенний курс по выбору, приоритет 1']

# Через квадратные скобки можно обращаться к одной или нескольким колонкам:

# In[23]:


data3[['Рейтинг', 'Осенний курс по выбору, приоритет 1']].head(3)

# Есть и более интересные способы индексации. Например, давайте выберем студентов из группы 182:

# In[24]:


data3[data3['Группа (в формате 182)'] == 182].sample(3)

# С DataFrame'ами и Series'ами одинаковой структуры можно производить математические операции:

# In[25]:


# strange_feature = data3['Рейтинг'] ** 2 + data3['Группа (в формате 182)']
# strange_feature.head()

# Видимо, 'Рейтинг' представлен в виде строки. Исправим это:

# In[26]:


# data3['Рейтинг'].apply(lambda x: float(str(x).replace(',', '.')))

# strange_feature = data3['Рейтинг'] ** 2 + data3['Группа (в формате 182)']
# strange_feature.head()

# Получаем ту же ошибку, ведь метод apply не модифицирует таблицу, а просто возвращает новый столбец. Обходят это обычно так:

# In[27]:


data3['Рейтинг'] = data3['Рейтинг'].apply(lambda x: float(x.replace(',', '.')))

strange_feature = data3['Рейтинг'] ** 2 + data3['Группа (в формате 182)']
strange_feature.head()

# В дальнейшем нам понадобится работать с перцентилями студентов. Чтобы сделать такой столбец, в pandas уже есть подходящий метод:

# In[28]:


data3['percentile'] = 1 - data3['Рейтинг'].rank() / data3.shape[0]

# добавим также наш странный признак
data3['new'] = strange_feature

# При желании можно удалить любой признак при помоши метода ```drop```:

# In[29]:


data3 = data3.drop(columns=['new'])
data3.head()

# ### Разведочный анализ
# 
# Теперь изучим наши данные. Вашим домашним заданием будет распределение студентов по курсам, с учётом их предпочтений, рейтинга и ограничений. Начнём к этому готовиться.

# Для начала посмотрим еще раз на типы данных и подумаем, надо ли их менять:

# In[30]:


data3.dtypes

# Вроде бы нет... 
# 
# А что с таблицей для 4ого курса? Как вы знаете, на ряд курсов студенты 3 и 4 годов обучения отбираются совместно, поэтому надо собрать данные в одну таблицу. *Можно ли это сделать без подготовки?*

# In[31]:


data3.head()

# In[32]:


data4.head()

# Кажется, рейтинги имеют разные распределения. Проверим это:

# In[33]:


data3['Рейтинг'].describe()

# Да, рейтинг для 4 курса тоже надо привести к числовому типу.

# In[34]:


data4['Рейтинг'] = data4['Рейтинг'].apply(lambda x: float(str(x).replace(',', '.')))
data4['Рейтинг'].describe()

# Видно, что квантили в самом деле отличаются — поэтому сами рейтинги не стоит использовать после объединения таблиц, надо работать только с перцентилями. Вычислим их и объединим таблицы с помощью метода ```concat```:

# In[35]:


data4['percentile'] = 1 - data4['Рейтинг'].rank() / data4.shape[0]

data = pd.concat([data3, data4])
data.head()

# Теперь для удобства переименуем столбцы (обратите внимание на ```inplace=True```):

# In[36]:


data.rename(columns={'Timestamp':'timestamp',
                     'ID':'id',
                     'Рейтинг':'rating',
                     'МИ?':'is_mi',
                     'Группа (в формате 182)':'18_group',
                     'Группа (в формате 173)':'17_group',
                     'Осенний курс по выбору, приоритет 1':'fall_1',
                     'Осенний курс по выбору, приоритет 2':'fall_2',
                     'Осенний курс по выбору, приоритет 3':'fall_3',
                     'Весенний курс по выбору, приоритет 1':'spring_1',
                     'Весенний курс по выбору, приоритет 2':'spring_2',
                     'Весенний курс по выбору, приоритет 3':'spring_3',
                     'Вы заполняете анкету в первый раз?':'is_first_time',
                     'blended-курс':'blended'},
           inplace=True)

# Поскольку у (почти всех) столбцов теперь названия являются корректными именами переменных в Python, мы можем использовать ещё один способ обращения к столбцам таблицы как к полям класса:

# In[37]:


data

# Нередко работы с данными начинают с поиска пропущенных значений (NaN и др.) и их заполнения. Для начала посмотрим на их наличие:

# In[38]:


data.isna().sum()

# Видно, что тут содержательных пропусков нет — есть только проблемы с колонками, специфичными для одного из курсов.
# 
# Заполнять пропуски необходимо в соответствии со смыслом колонки. Можно заполнять с помощью среднего, медианного, константного или других значений. Для этого обычно используется метод ```fillna()``` с которым вы познакомитесь в домашнем задании.

# Также для разведочного анализа может помочь метод ```groupby(column)```. 
# 
# Он группирует объекты по указанной(-ым) колонке(-ам). Необходимо также указать какую статистику для группировки выводить. Это может быть количество (count), среднее (mean) или другие. Из огромной функциональности этого метода разберем только несколько базовых приемов:

# In[39]:


data.groupby('fall_1').count()[['id', 'is_mi']]

# *Какие выводы вы можете сделать отсюда?*

# Сделаем ```groupby``` с усреднением:

# In[40]:


data.groupby(by='fall_1').mean()

# Отсюда мы узнаём среднюю перцентиль для того или иного курса по выбору.
# Обратите внимание, что средний рейтинг тут не очень показателен из-за разных его распределений у разных годов обучения.
# 
# Что выводится в следующей строке?

# In[41]:


data.groupby(by='fall_1').count()[['17_group', '18_group']].sum(axis=1)

# Полезным бывает посмотреть на основные статистики по каждому *числовому*  признаку (столбцу). Метод ```describe``` позволяет быстро сделать это: 

# In[42]:


data.describe()

# *Какие элементы таблицы выше могут быть полезны? Для чего?*

# Студентам специализации МОП (машинное обучение и приложения) нельзя выбирать курс "Машинное обучение 2" в качестве весеннего курса по выбору. Давайте проверим, есть ли те, кто попытался:

# In[43]:


!wget  -O 'ml_students_anon.xlsx' -q 'https://www.dropbox.com/s/izc21kik0b8iw10/_ml_students_anon.xlsx?dl=0'

ml_students = pd.read_excel('ml_students_anon.xlsx')
ml_students.head()

# Если вы знакомы с SQL, то знаете, что там крайне часто используется операция JOIN для соединения нескольких таблиц по тому или иному значению. В pandas такое тоже есть, функция называется ```merge```.
# 
# У нас есть две таблицы: (1) приоритеты студентов по элективам и (2) специализации, на которые распределены студенты. Эти таблицы содержат разную информацию про студентов, но в обеих конкретный студент имеет один и тот же ID. Допустим, мы теперь хотим соединить эти таблицы — то есть получить новую таблицу, в которой для каждого студента есть информация и о приоритетах по элективам, и о его специализации. Как раз для этого и понадобится операция ```merge```.
# 
# Идея соединения таблиц также отражена на картинке ниже.

# <img src="https://i.imgur.com/WYyBFTE.png" style="width: 400px">

# In[44]:


data = data.merge(ml_students, on='id', how='left')
data.head()

# In[45]:


data[(data['is_ml_student'] == True) & 
     (
         (data['spring_1'] == 'Машинное обучение 2') | \
         (data['spring_2'] == 'Машинное обучение 2') | \
         (data['spring_3'] == 'Машинное обучение 2')
     )
]

# In[ ]:




# Попробуем понять, есть ли явная зависимость между рейтингом и номером группы. Для начала посмотрим на корреляции (функция corr считает по умолчанию корреляцию Пирсона):

# In[46]:


corrmat = data[['rating', '18_group', '17_group']].corr()
corrmat

# *Проинтерпретируйте результаты. Можно ли им доверять, разумно ли смотреть на корреляции?* 

# In[ ]:




# Здесь числовых признаков не так много, но на практике их бывают десятки, а то и сотни. В таком случае бывает полезно посмотреть на эту матрицу корреляций в виде heatmap:

# In[47]:


# импорт библиотек для графиков
import matplotlib.pyplot as plt
import seaborn as sns

# In[48]:


plt.figure(figsize=(8, 8))
sns.heatmap(corrmat, square=True)
plt.show()

# К графикам надо относиться серьёзно, они должны быть понятными и информативными. Рассмотрим несколько примеров.
# 
# *Прокомментируйте что вам кажется хорошим и плохим на данных графиках.* 

# In[49]:


plt.hist(data['timestamp'])
plt.title('Гистограмма распределения ответов по времени')
plt.show()

# In[50]:


sns.set()

fig, ax = plt.subplots()
ax = sns.countplot(y='fall_1', data=data)
ax.set_title('Осенний курс по выбору, приоритет 1')
ax.set_ylabel('')

ax.set(xlabel='Количество заявок')


plt.show()

# In[51]:


crs3 = data[data['17_group'].isna()]
gr_raiting_med = crs3.groupby('18_group').median()['rating']
gr_raiting_sum = crs3.groupby('18_group').sum()['rating']

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Зависимость номера группы от рейтинга')
#fig.text('Рейтинг')


axs[0].plot(gr_raiting_sum)
axs[0].set_ylabel('Суммарный рейтинг')

axs[1].plot(gr_raiting_med)
axs[1].set_ylabel('Медианный рейтинг')


plt.xlabel('Номер группы')

plt.show()

# Если вы будете делать графики без подписанных осей, с налезающими друг на друга метками, неаккуратными линиями и т.д., то имеете все шансы попасть сюда: https://t.me/funny_homeworks

# Сохраним полученную таблицу, чтобы вы могли продолжить с ней работу дома: 

# In[52]:


data.to_excel('end_seminar.xlsx', index=False)

# ### Разведочный анализ данных

# Решение любой задачи, связанной с машинным обучением, начинается с разведочного анализа данных. Перед тем, как строить модели, надо понять, сколько у нас данных и какая информация есть о каждом объекте, а также:
# * выяснить, нет ли пропусков в данных (т.е. отсутствующих значений признаков у некоторых объектов)
# * выяснить, нет ли выбросов (т.е. объектов, которые очень сильно отличаются от большинства, имеют неадекватные значения признаков)
# * выяснить, нет ли повторов в данных
# * выяснить, нет ли нелогичной информации (например, если мы анализируем данные по кредитам, и видим запись, где кредит выдали пятилетнему ребёнку, то это странно)
# 
# И это лишь небольшой список проблем, которые можно выявить. Помимо этого с данными нужно в целом познакомиться, чтобы понять, какие признаки там можно сделать, какие из них будут наиболее полезны.
# 
# Попробуем провести такой анализ на реальной задаче предсказания продолжительности поездки на такси в Нью-Йорке: https://www.kaggle.com/c/nyc-taxi-trip-duration/overview

# Рассказ во многом взят из ноутбука https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367/notebook

# In[53]:


import numpy as np
import pandas as pd
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
import seaborn as sns

# Загрузим данные и изучим доступные нам признаки.

# In[54]:


!wget  -O 'train.csv' -q 'https://www.dropbox.com/s/en5f9nhn915cnkf/_train.csv?dl=0'

# In[55]:


train = pd.read_csv('train.csv')
train.head()

# Смысл столбцов:
# 
# * id - идентификатор поездки
# * vendor_id - код провайдера, от которого пришла информация о поездке
# * pickup_datetime - время старта поездки
# * dropoff_datetime - время окончания поездки
# * passenger_count - число пассажиров (вводится водителем)
# * pickup_longitude - долгота точки посадки
# * pickup_latitude - широта точки посадки
# * dropoff_longitude - долгота точки высадки
# * dropoff_latitude - долгота точки высадки
# * store_and_fwd_flag - равно Y, если информация о поездке какое-то время хранилась в памяти таксометра из-за отсутствия связи; иначе принимает значение N
# * trip_duration - продолжительность поездки в секундах

# In[56]:


train.shape

# Данных довольно много, поэтому нужно задумываться об эффективности всех вычислений. Ниже мы увидим на примере, что правильное использование возможностей pandas позволит существенно ускорить вычисления.

# Посмотрим внимательно на столбцы из нашей таблицы и попробуем проверить, нет ли там противоречий и проблем. Например, в голову приходят следующие вопросы:
# * можно ли считать ID уникальным идентификатором поездки, или же есть записи с одинаковыми ID?
# * есть ли где-то пропуски?
# * действительно ли столбец store_and_fwd_flag принимает только значения Y и N?

# In[57]:


print('Id is unique.') if train.id.nunique() == train.shape[0] else print('oops')

# In[58]:


print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] else print('oops')

# In[59]:


print('The store_and_fwd_flag has only two values {}.'.format(str(set(train.store_and_fwd_flag.unique()))))

# Посмотрим на типы данных в каждом столбце.

# In[60]:


train.dtypes

# Видно, что pandas отнёс столбцы с датами к общему типу object. Будет лучше привести их к типу datetime, чтобы использовать удобные функции для манипуляций с датами. Заодно сделаем проверку — действительно ли столбец check_trip_duration показывает продолжительность поездки, или же он входит в противоречие со столбцами pickup_datetime и dropoff_datetime.

# In[61]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())
duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]
print('Trip_duration and datetimes are ok.') if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0 else print('Ooops.')

# In[62]:


train.dtypes

# In[63]:


train.head()

# Ещё имеет смысл проверить целевую переменную trip_duration. Нет ли там выбросов? Какими по продолжительности бывают поездки? Выглядит ли распределение этой переменной осмысленно?

# In[64]:


plt.hist(train['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()

# Когда в каком-то столбце распределение имеет тяжёлые хвосты или есть выбросы, обычные гистограммы не очень информативны. В этом случае может быть полезно нарисовать распределение в логарифмической шкале.

# In[65]:


train['log_trip_duration'] = np.log1p(train['trip_duration'].values)
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()

# В целом распределение разумное, продолжительность поездки находится примерно в следующем интервале:

# In[66]:


print("В секундах:", np.exp(5), np.exp(9))
print("В минутах:", np.exp(5) // 60, np.exp(9) // 60)

# Самая длинная поездка (в часах):

# In[67]:


train['trip_duration'].max() // 3600

# In[68]:


train[train.trip_duration >= 979 * 3600]

# Это выброс. Мог сломаться таксометр, водитель мог забыть остановить поездку по каким-то причинам и т.д. В любом случае, будет странно обучаться на таких данных, обычно их выкидывают из выборки. Кандидатов на выбрасывание не так уж много — например, если взять поездки длиной 10 часа и больше, то их окажется около двух тысяч, и определённо надо выяснять, что это такое.

# In[69]:


train[train.trip_duration >= 10 * 3600]

# Попробуем нарисовать, откуда обычно стартуют поездки.

# In[70]:


N = 100000
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.figure(figsize=(15,15))
plt.scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N],
              color='blue', s=1, label='train', alpha=0.1)
plt.ylabel('latitude')
plt.xlabel('longitude')
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()

# Теперь нарисуем как следует, на карте.

# In[71]:


from ipyleaflet import Map, Circle, LayerGroup, basemaps

# In[72]:


def show_circles_on_map(data, latitude_column, longitude_column, color):
    """
    The function draws map with circles on it.
    The center of the map is the mean of coordinates passed in data.
    
    data: DataFrame that contains columns latitude_column and longitude_column
    latitude_column: string, the name of column for latitude coordinates
    longitude_column: string, the name of column for longitude coordinates
    color: string, the color of circles to be drawn
    """

    center = (data[latitude_column].mean(), data[longitude_column].mean())
    result_map = Map(center=center, zoom=10, basemap=basemaps.Esri.NatGeoWorldMap)

    circles = []
    for _, row in data.iterrows():
        circles.append(Circle(
            location=(row[latitude_column], row[longitude_column]),
            fill_color=color,
            fill_opacity=0.2,
            radius=300,
            stroke=False
        ))
    circles_layer = LayerGroup(layers=circles)
    result_map.add_layer(circles_layer)

    return result_map

# In[73]:


show_circles_on_map(train.sample(1000), "pickup_latitude", "pickup_longitude", "blue")

# Кстати, случайный ли порядок записей в таблице? Это важно понимать, например, для разбиения выборки на обучающую и тестовую части. Если порядок не является случайным, а мы отнесём к обучающей части первую половину таблицы, то есть риск, что данные в обучении и тесте будут иметь разные распределения, а значит, модель сможет хорошо работать только на одной из частей.

# In[74]:


plt.figure(figsize=(15,5))
days_since_min_ride = (train['pickup_datetime'] - train['pickup_datetime'].min()).apply(lambda x: x.total_seconds() // (60*60*24))
plt.plot(days_since_min_ride[::1000], 'o-')
plt.title('Связь номера строки и времени поездки')
plt.xlabel('Номер записи в таблице')
plt.ylabel('Дней с момента ранней поездки')

# Вроде бы всё довольно случайно.

# Посчитаем какие-нибудь признаки. Скорее всего продолжительность поездки неплохо зависит от расстояния — посчитаем его. Кстати, важно заметить, что само расстояние не может быть признаком, поскольку в тестовой выборке нет информации о пункте назначения. Но мы пока исследуем данные и, может, увидим в расстояниях что-то интересное.
# 
# Можно выбрать научный подход и посчитать честное расстояние на сфере между двумя точками. Это называется [haversine distance](https://en.wikipedia.org/wiki/Haversine_formula).
# 
# Можно решить, что Земля плоская, и считать стандартные расстояния. В этом случае очень неплохо подойдёт [манхэттенское расстояние](https://en.wikipedia.org/wiki/Taxicab_geometry) — оно учитывает, что машины всё-таки не летают.

# In[75]:


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

# Обсудим немного скорость вычислений в pandas. Есть несколько способов применить функцию к каждой строке в таблице. Если вы до этого изучали преимущественно C/C++, то первое, что должно прийти вам в голову, — написать цикл.

# In[76]:


train1 = train.copy()

# In[77]:


%%timeit

for i in range(100):
    train1[i, 'distance_haversine'] = haversine_array(train1['pickup_latitude'].iloc[i], 
                                                      train1['pickup_longitude'].iloc[i], 
                                                      train1['dropoff_latitude'].iloc[i], 
                                                      train1['dropoff_longitude'].iloc[i])

# Обратите внимание, что мы посчитали расстояния всего лишь для первых 100 записей, и это уже было не очень быстро.

# Можно воспользоваться функцией ```apply```.

# In[78]:


%%timeit

train1.iloc[:5000, :].apply(lambda x: 
                            haversine_array(x['pickup_latitude'], 
                                            x['pickup_longitude'], 
                                            x['dropoff_latitude'], 
                                            x['dropoff_longitude']),
                            axis=1)

# За то же самое время мы успеваем обработать уже 5000 записей, а не 100.

# Но лучший способ — это векторизовать вычисления. Подробнее об этом мы будем говорить на следующих семинарах.

# In[79]:


%%timeit
train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

# Чуть подробнее об ускорении вычислений можно почитать здесь: https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2c6a4d6

# Посмотрим на распределения расстояний.

# In[80]:


plt.hist(np.log1p(train.distance_haversine), bins=100)
plt.show()

# In[81]:


plt.hist(np.log1p(train.distance_dummy_manhattan), bins=100)
plt.show()

# Выглядят достаточно одинаково.
# 
# Поисследуем теперь связь между расстоянием и продолжительностью поездки.

# In[82]:


plt.figure(figsize=(10,10))
plt.scatter(train.distance_haversine[:100000], train.trip_duration[:100000], marker='o')
plt.xlabel('haversine distance')
plt.ylabel('trip duration')

# В нижней части графика точки неплохо выстраиваются в линию, но ещё есть некоторое количество выбросов. Нарисуем тот же график без них.

# In[83]:


plt.figure(figsize=(10,10))
plt.scatter(train[train.trip_duration < 20000].distance_haversine[:100000],
            train[train.trip_duration < 20000].trip_duration[:100000], marker='o')
plt.xlabel('haversine distance')
plt.ylabel('trip duration')

# Кажется, тут вполне может сработать линейная регрессия!
# 
# Ещё раз напомним, что расстояние не может быть признаком, его не посчитать для тестовой выборки. Но мы пока просто играемся.

# In[84]:


from sklearn import linear_model, metrics

# In[85]:


train_filtered = train[train.trip_duration < 20000]
X = train_filtered.distance_haversine.values[:, np.newaxis]
y = train_filtered.trip_duration.values

regr = linear_model.Ridge()
regr.fit(X, y)
metrics.mean_absolute_error(regr.predict(X), y)

# In[86]:


train_filtered = train[train.trip_duration < 20000]
X = train_filtered.distance_dummy_manhattan.values[:, np.newaxis]
y = train_filtered.trip_duration.values

regr = linear_model.Ridge()
regr.fit(X, y)
metrics.mean_absolute_error(regr.predict(X), y)

# То есть в среднем модель ошибается где-то на 300 секунд при предсказании продолжительности поездки. Как понять, хорошо это или плохо? Например, сравнить с качеством константной модели — например, которая всегда в качестве прогноза выдаёт медианное или среднее время поездки.

# In[87]:


metrics.mean_absolute_error(np.median(y) * np.ones(y.shape), y)

# In[88]:


metrics.mean_absolute_error(np.mean(y) * np.ones(y.shape), y)

# ### Почему важно исследовать данные?

# Иногда бывает, что задача сложная, но при этом хорошего качества можно добиться с помощью простых правил. Причины могут быть разные:
# * Разметка собрана по простому правилу. Например, для задачи предсказания тональности твитов могли сделать разметку через эмодзи — тогда достаточно, скажем, добавить признак "наличие в тексте подстроки ':)'".
# * Задача действительно простая и не требует поиска закономерностей методами машинного обучения.
# * В данных есть утечка (leak) — то есть в признаках содержится информация, которая на самом деле не должна быть доступна при построении прогноза.
# 
# Про некоторые истории с утечками можно почитать и посмотреть здесь:
# * https://dyakonov.org/2018/06/28/простые-методы-анализа-данных/
# * https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/discussion/4865
# * https://www.youtube.com/watch?v=UOxf2P9WnK8
