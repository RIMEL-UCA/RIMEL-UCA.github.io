#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# # Семинар 4

# In[9]:


%pylab inline
import matplotlib.pylab as plt
import math
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# ## Линейная классификация
# 
# Ранее мы изучили задачу восстановления регрессии и способы её решения. В частности, для линейной модели с функционалом качества MSE решение данной задачи выписывается аналитически через матрицу "объект-признак" и вектор ответов обучающей выборки.
# 
# Давайте приступим к рассмотрению задачи классификации. Начнём с того, что вспомним её постановку.
# 
# ### Постановка задачи классификации
# 
# Пусть задана обучающая выборка $X = \left\{ \left( x_i, y_i \right) \right\}_{i=1}^l, x_i \in \mathbb{X}, y_i \in \mathbb{Y},$ — $l$ пар объект-ответ, где
# $\mathbb{X}$ — пространство объектов,
# $\mathbb{Y}$ — пространство ответов.
# 
# Задача классификации отличается о задачи восстановления регрессии тем, что в данном случае пространство ответов конечно:
# $$\left| \mathbb{Y} \right| < \infty.$$
# 
# 
# **Примеры**:
# 
# 1) Кредитный скоринг.
# * Объект — заявка на кредит от клиента банка.
# * Ответ — решение о выдаче или невыдаче кредита.
# 
# 2) Медицинская диагностика.
# * Объект — характеристики пациента (результаты анализов, общая информация о пациенте, история болезни).
# * Ответ — диагноз (одно из ограниченного множества заболеваний).
# 
# В дальнейшем будем рассматривать задачу бинарной классификации (то есть разделения объектов на 2 группы) и полагать $\mathbb{Y} = \{-1, +1\}.$
# 
# ### Линейная модель
# 
# Сегодня мы будем рассматривать линейную модель. В случае классификации формула для ответа на объекте $x$ видоизменяется по сравнению с линейной регрессией (при этом мы предполагаем, что нулевой признак в выборке — константный):
# $$a(x) = sign(w_0 \cdot 1 + w_1 x_1 + \dots w_d x_d) = sign(\sum_{j = 0}^d w_j x_j) = sign(\langle w, x \rangle).$$
# 
# Множество точек $\langle w, x\rangle = 0$ является гиперплоскостью и делит пространство на 2 части по разные стороны от этой самой гиперплоскости. При этом объекты, расположенные по разные стороны от неё, полученный классификатор относит к различным классам в зависимости от знака скалярного произведения.
# 
# <tr>
# <td> <img src="images/plane.png" alt="Drawing" style="width: 350px;"/> </td>
# <td> <img src="images/cube.png" alt="Drawing" style="width: 350px;"/> </td>
# </tr>
# 
# Отметим, что для некоторого объекта $x$ расстояние до гиперплоскости для фиксированного вектора весов $w$ можно записать как $\frac{\left| \langle w, x \rangle \right|}{\|w \|}$.
# 
# Поэтому чем больше $\left| \langle w, x \rangle \right|$, тем дальше объект отстоит от гиперплоскости, тем увереннее алгоритм в его классификации.
# 
# ### Оптимизируемый функционал
# 
# Как и в случае линейной регрессии, поиск оптимального алгоритма среди множества всех линейных моделей в случае классификации производится путем оптимизации некоторого функционала. Можно было бы рассматривать MAE или MSE, как и в случае регрессии, однако в случае классификации этот подход кажется немного странным, как минимум, потому что классы могут задаваться своими названиями, а не номерами. Их можно было бы занумеровать, но тогда результат будет сильно зависеть от способа нумерации.
# 
# Будем минимизировать долю неверных классификаций, которую можно записать в следующем виде:
# 
# $$Q(w, X) = \frac{1}{l} \sum_{i = 1}^l [a(x_i) \ne y_i] \to \min_w.$$
# 
# В этом функционале слагаемое $[a(x_i) \ne y_i]$ отвечает за то, была ли допущена алгоритмом $a(\cdot)$ ошибка на объекте $x_i$.
# 
# Заметим, что $[a(x_i) \ne y_i] = [sign \, \langle w, x_i \rangle \ne sign \, y_i] = [y_i \langle w, x_i \rangle < 0],$ поэтому функционал можно переписать в следующем виде:
# $$Q(w, X) = \frac{1}{l} \sum_{i = 1}^l [y_i \langle w, x_i \rangle  < 0] \to \min_w.$$
# 
# Величина $M_i = y_i \langle w, x_i \rangle$ называется **отступом** объекта $x_i$. Как было показано ранее:
# * объект верно классифицируется тогда и только тогда, когда отступ положительный;
# * чем больше значение отступа, тем надёжнее классифицируется объект.
# 
# Тем не менее, получившийся функционал не получится оптимизировать градиентными методами. Поэтому рассмотрим пороговую функцию потерь для одного объекта $L(M) = [M < 0],$ участвующую в формуле функционала, оценим её сверху некоторой гладкой функцией потерь и будем минимизировать её:
# 
# $$Q(w, X) = \frac{1}{l} \sum_{i = 1}^l [M_i < 0] = \frac{1}{l} \sum_{i = 1}^l L(M_i) \le \frac{1}{l} \sum_{i = 1}^l \widetilde{L}(M_i) = \widetilde{Q}(w, X) \to \min_{w}$$
# 
# Примеры оценок сверху для одного объекта:
# <img src="images/est.png" alt="Drawing" style="width: 650px;">
# 
# **ВАЖНО**: мы минимизируем верхнюю оценку и надеемся, что тогда будет минимизироваться и исходный функционал $Q(w, X)$, однако нет никаких гарантий, что так и будет!
# 
# В зависимости от выбранной верхней оценки будут меняться свойства итогового алгоритма, положение разделяющей гиперплоскости и т.п.
# 
# ### Итог
# 
# Таким образом, процесс обучения линейного классификатора складывается из следующих шагов:
# 
# * Получение обучающей выборки $X = \{ \left( x_i, y_i \right) \}_{i=1}^l$
# * Выбор верхней оценки для пороговой функции потерь — отсюда получаем общий вид оптимизируемого функционала $\widetilde{Q}(w, X^l)$
# * Проводим оптимизацию при помощи некоторого метода оптимизации и получаем оптимальное значение $w^*$
# * Предсказываем ответы для новых объектов по формуле $a(x) = \langle w^*, x\rangle$
# 
# ### Логистическая регрессия
# 
# Рассмотрим в качестве верхней оценки пороговой функции потерь логистическую функцию:
# 
# $$\widetilde{L}(M) = \log_2(1 + \exp(-M)).$$
# 
# Таким образом, необходимо решить следующую оптимизационную задачу:
# $$\frac{1}{l} \sum_{i=1}^l \widetilde{L} (M_i) = \frac{1}{l} \sum_{i=1}^l \log_2 (1 + \exp (-y_i \langle w, x_i \rangle)) \to \min_w$$
# 
# Получившийся метод обучения называется **логистической регрессией**.
# 
# Одно из полезных свойств логистической регрессии, которое будет изучено нами несколько позднее, — тот факт, что она позволяет предсказывать помимо метки класса ещё и вероятность принадлежности каждому из них, что может быть полезным в некоторых задачах.
# 
# **Пример**: Вы работаете в банке и хотите выдавать кредиты только тем клиентам, которые вернут его с вероятностью не меньше 0.9.
# 
# ### Пример обучения логистической регрессии
# #### Определение спама по тексту электронного письма
# 
# Попробуем при помощи моделей линейной классификации построить алгоритм, отделяющий спам от нормальной почты. Для экспериментов воспользуемся небольшим набором данных с [UCI](https://archive.ics.uci.edu/ml/datasets.html). Объекты в датасете соответствуют письмам, которые описаны признаками на основе текста письма, спам — положительный пример для классификации, хорошее письмо — отрицательный пример.
# 
# ![Spam](./images/spam.jpg)

# In[10]:


import pandas
import urllib2

SPAMBASE_NAMES_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names'
SPAMBASE_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'

feature_names = [
    line.strip().split(':')[0] 
    for line in urllib2.urlopen(SPAMBASE_NAMES_URL).readlines()[33:]
]
spam_data = pandas.read_csv(SPAMBASE_DATA_URL, header=None, names=(feature_names + ['spam']))
 
X, y = spam_data.ix[:, :-1].values, spam_data.ix[:, -1].values
 
spam_data.head()

# ### Обучение логистической регрессии

# Разделим выборку на обучающую и тестовую в отношении 80/20 и обучим логистическую регрессию при помощи объекта [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

# In[11]:


from sklearn.linear_model import LogisticRegression

# splitting data
train_part = 0.8
n_train = int(train_part * X.shape[0])
X_tr = X[:n_train]
X_test = X[n_train:]
y_tr = y[:n_train]
y_test = y[n_train:]

# training
lr = LogisticRegression()
lr = lr.fit(X_tr, y_tr)

# Оценим долю верных прогнозов полученной модели (accuracy) при помощи соответствующей функции из модуля [sklearn.metrics](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).

# In[12]:


from sklearn import metrics

preds = lr.predict(X_test)

print 'Accuracy =', metrics.accuracy_score(y_test, preds)

# Сгенерируем двумерную искуственную выборку из 2 различных нормальных распределений:

# In[13]:


plt.figure(figsize=(20,10))
mean0 = [10, 5]
cov0 = [[1, 0], [0, 5]]  # diagonal covariance
data0 = np.random.multivariate_normal(mean0, cov0, 1000)
mean1 = [0, 0]
cov1 = [[3, 1], [0, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, 1000)
data = np.vstack((data0, data1))
y_vis = np.hstack((-np.ones(1000), np.ones(1000)))


plt.scatter(data0[:, 0], data0[:, 1], c='red')
plt.scatter(data1[:, 0], data1[:, 1], c='green')
plt.legend(['y = -1', 'y = 1'])
axes = plt.gca()
axes.set_xlim([-5,15])
axes.set_ylim([-5,10])
plt.show()

# Обучим логистическую регрессию:

# In[14]:


X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(data, y_vis, test_size=0.2)

from sklearn import linear_model

logreg = linear_model.LogisticRegression(penalty = 'l2')
logreg.fit(X_train_vis, y_train_vis)

# Полученные в результате оптимизации коэффициенты линейной модели содержатся в атрибутах coef\_ и intercept\_ соответствующего объекта. Визуализируем разделяющую гиперплоскость алгоритма и рассмотрим значения предсказанных моделью вероятностей принадлежности нового объекта каждому из классов в зависимости от его координат.

# In[15]:


print(logreg.coef_, logreg.intercept_)

# In[16]:


w_1 = logreg.coef_[0][0]
w_2 = logreg.coef_[0][1]
w_0 = logreg.intercept_[0]


plt.figure(figsize=(20,10))
plt.scatter(data0[:, 0], data0[:, 1], c='red')
plt.scatter(data1[:, 0], data1[:, 1], c='green')
plt.legend(['y = -1', 'y = 1'])
x_arr = np.linspace(-10, 15, 3000)
plt.plot(x_arr, -(w_0 + w_1 * x_arr) / w_2)
axes = plt.gca()
axes.set_xlim([-5,15])
axes.set_ylim([-5,10])
plt.show()

# In[17]:


point = np.array([[10, 2]]) # изменяем только координаты объекта

plt.figure(figsize=(20,10))
plt.scatter(data0[:, 0], data0[:, 1], c='red')
plt.scatter(data1[:, 0], data1[:, 1], c='green')
plt.scatter(point[:, 0], point[:, 1], marker = '*', s = 300, color = 'magenta')
plt.legend(['y = -1', 'y = 1'])
x_arr = np.linspace(-10, 15, 3000)
plt.plot(x_arr, -(w_0 + w_1 * x_arr) / w_2)
axes = plt.gca()
axes.set_xlim([-5,15])
axes.set_ylim([-5,10])
prob = logreg.predict_proba(point)
print 'P(y = -1|x) =', prob[0][0]
print 'P(y = 1|x) =', prob[0][1]
plt.show()

# ### SGDClassifier
# 
# Объект [SGDClissifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) позволяет обучать линейные модели классификации и регрессии с помощью стохастического градиентного спуска.
# 
# Полезные параметры:
# * loss - функция потерь (по факту то, какую модель обучаем): **hinge** (SVM), **log** (логистическая регрессия), **perceptron** (персептрон) и другие;
# * penalty - тип регуляризации: **l1**, **l2**, **elasticnet** (смесь l1 и l2 регуляризации);
# * alpha - коэффициент регуляризации;
# * fit_intercept - необходимо ли добавлять в модель свободный член (True/False);
# * n_iter - число эпох (полных проходов по выборке) при обучении;
# * learning_rate - шаг градиентного спуска (оптимизируется по умолчанию).

# In[18]:


from sklearn.linear_model import SGDClassifier
lr_sgd = SGDClassifier(loss="log", alpha=0.05, n_iter=200, fit_intercept=True)
lr_sgd.fit(X_tr, y_tr)
preds_sgd = lr_sgd.predict(X_test)
print 'Accuracy =', metrics.accuracy_score(y_test, preds_sgd)

# ## Оценка качества работы классификатора
# ### Бинарные метрики
# Обучение и оценка качества модели производится на независимых множествах примеров. Как правило, имеющующиеся примеры разбивают на два подмножества: обучение (`train`) и контроль (`test`). Выбор пропорции разбиения — компромисс: большой размер обучения ведет к более богатым информацией и качественным алгоритмам, большой размер контрольной выборки ведет к менее шумной оценке качества.
# 
# Для оценки качества классификации рассматривают [матрицу ошибок](http://en.wikipedia.org/wiki/Confusion_matrix): строчки соответствуют прогнозу модели, столбцы — истинным ответам, ячейки содержат число примеров тестовой выборки. Если для некоторого алгоритма $a(\cdot)$ и объекта $x$ выполняется $a(x) = 1$, то говорят, что алгоритм $a$ _выделяет_ объект $x$.
# 
# | | y = +1 | y = -1 |
# |------|------|
# |   **a(x) = +1**  | TP| FP |
# | **a(x) = -1**| FN | TN |
# 
# Диагональ матрицы ошибок содержит правильно классифицированные положительные (TP) и отрицательные (TN) примеры. False Positive (FP) — ошибки I рода (ложное срабатывание, положили в спам хорошее письмо), False Negative (FN) — ошибки II рода (не отфильтровали спам). Ошибки I и II рода могут иметь различную стоимость.
# 
# Часто рассматриваются следующие метрики качества бинарной классификации:
#   - Доля правильных ответов (Accuracy):
#   $$accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
#   - Точность/precision (доля действительно положительных объектов среди объектов, выделенных алгоритмом):
#   $$precision = \frac{TP}{TP + FP}$$
#   - Полнота/recall (доля выделенных алгоритмом объектов среди всех положительных объектов выборки):
#   $$recall = \frac{TP}{TP + FN}$$
#   - $F_1$-мера (среднее гармоническое между точностью и полнотой)
#   $$F_1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$
#   
# Подробнее про метрики качества бинарной классификации на Википедии: [Precision and Recall](http://en.wikipedia.org/wiki/Precision_and_recall).
# 
# ### Выбор порога классификации
# Многие модели классификации получают оценку принадлежности положительному классу $b(x) \in \mathbb{R}$, после чего принимается решение о классификации объекта путем сравнения оценки с некоторым порогом:
# 
# $a(x) = \begin{cases}
#         +1, \, b(x) \ge t,\\
#         -1, \, b(x) < t.
#         \end{cases}$
#         
# Матрица ошибок и все производные от нее метрики (Accuracy, Precision, Recall, etc.) зависят от порога $t$:
# ![Порог классификации](./images/binary_threshold.png) 
# 
# Поскольку иногда необходимо оценить качество модели $b(x),$ предсказывающей не конкретную метку класса, а степень принадлежности классу (т.е. вне зависимости от значения порога), рассматривают координатную плоскость, по осям которой отложены значения метрик, а качество работы классификатора представлено в виде кривой — траектории изменения соответствующих метрик при варьировании порога:
# ![Принцип построения ROC-кривой](./images/roc_trajectory.png)
# 
# Наиболее распространены ROC-кривые и Precision/Recall кривые.
#  - По осям Ox и Oy ROC-кривой отложены соответственно False Positive Rate (FPR) и True Positive Rate (TPR):
#  $$FPR = \frac{FP}{FP + TN},$$
#  $$TPR = \frac{TP}{FN + TP}.$$
#  - По осям Ox и Oy PR-кривой отложены соответственно Recall и Precision.
# 
# 
# Аббревиатура ROC ([Receiver Operating Characteristic](en.wikipedia.org/wiki/Receiver_operating_characteristic)) была унаследована из инженерного дела.
# 
# ![Примеры ROC-кривых](./images/roc_and_pr_curves.png)
# 
# В случае, если необходимо сравнить качество классификаторов вне зависимости от порога, применяют интегральные числовые метрики, например AUC-ROC (**A**rea **U**nder RO**C**) — площадь под ROC-кривой классификатора. AUC-ROC идеально работающего классификатора равно 1. Идеальный случайный классификатор в среднем имеет AUC-ROC=0.5.
# 
# Построим описанные кривые для логистической регрессии, обученной на описанном выше датасете.

# * PR-кривая проходит через точку (0,0).
# 
# * Если при каком-то значении порога $t$ алгоритм $a(x)$ идеально разделяет объекты 2 классов, то PR-кривая проходит через точку (1,1).
# 

# In[19]:


from sklearn.cross_validation import train_test_split

X_train_curve, X_test_curve, y_train_curve, y_test_curve = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
lr = lr.fit(X_train_curve, y_train_curve)

# In[20]:


from sklearn.metrics import precision_recall_curve

# In[21]:


precision, recall, thresholds = precision_recall_curve(y_test_curve, lr.predict_proba(X_test_curve)[:, 1])
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()

# In[22]:


from sklearn.metrics import auc

auc(recall, precision)

# * ROC проходит через точки (0,0) и (1,1)
# * Если при каком-то значении порога $t$ алгоритм $a(x)$ идеально разделяет объекты 2 классов, то ROC проходит через точку (0,1).
# 

# In[23]:


from sklearn.metrics import roc_curve

# In[24]:


fpr, tpr, thresholds = roc_curve(y_test_curve, lr.predict_proba(X_test_curve)[:, 1])
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# In[25]:


auc(fpr, tpr)

# ## Валидация
# Чтобы оценить качество работы алгоритма, необходимо провести валидацию. Это один из самых важных шагов в процессе решения задачи. Оценим accuracy для модели логистической регрессии в задаче про спам-письма на тестовой выборке. 

# In[26]:


spam_data = pandas.read_csv(SPAMBASE_DATA_URL, header=None, names=(feature_names + ['spam']))
 
X, y = spam_data.ix[:, :-1].values, spam_data.ix[:, -1].values

# обучающая выборка
X_tr = X[:n_train]
y_tr = y[:n_train]

# валидационная выборка
X_test = X[n_train:]
y_test = y[n_train:]

# обучим ещё раз логистическую регрессию
lr = LogisticRegression()
lr = lr.fit(X_tr, y_tr)

#посмотрим на точность классификации

preds = lr.predict(X_test)
print 'Accuracy =', metrics.accuracy_score(y_test, preds)

# А теперь попробуем перемешать объекты и повторим действия:

# In[27]:


from sklearn import cross_validation
X_tr, X_test, y_tr, y_test = cross_validation.train_test_split(X, y, test_size=train_part, random_state=123)

# обучим ещё раз логистическую регрессию
lr = LogisticRegression()
lr = lr.fit(X_tr, y_tr)

#посмотрим на точность классификации

preds = lr.predict(X_test)
print 'Accuracy =', metrics.accuracy_score(y_test, preds)

# Как мы видим, качество классификации новых данных резко возросло. С чем это может быть связано? Рассмотрим вектор целевой переменной:

# In[28]:


plt.plot(y, 'ro')
plt.xlabel('Object number')
plt.ylabel('Target')

# Проблема заключалась в том, что в выборке примеры были упорядочены: сначала шли примеры положительного класса, а потом отрицательного. Поэтому нельзя забывать **перемешивать классы**.
# 
# Чтобы повысить устойчивость оценки качества, можно проводить разбиение выборки на обучающую и тестовую не один, $N$ раз, после чего усреднять результаты, полученные на $N$ контрольных выборках. Для этого можно использовать функцию [`sklearn.cross_validation.ShuffleSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html#sklearn.cross_validation.ShuffleSplit)

# In[29]:


cv = cross_validation.ShuffleSplit(n=X.shape[0], n_iter = 5, train_size=0.9, test_size=0.1, random_state=123)
quals = []
lr = LogisticRegression()
for tr_ind, test_ind in cv:
    # using new set split for training and evaluating
    lr.fit(X[tr_ind, :], y[tr_ind])
    quals.append(metrics.roc_auc_score(y[test_ind], lr.predict_proba(X[test_ind,:])[:,1]))

# In[30]:


print 'Mean AUC-ROC =', np.mean(quals)
print 'AUC-ROC standart deviation =', np.std(quals)

# Увеличим $N$:

# In[31]:


cv = cross_validation.ShuffleSplit(n=X.shape[0], n_iter = 10, train_size=0.9, test_size=0.1, random_state=124)
quals = []
lr = LogisticRegression()
for tr_ind, test_ind in cv:
    lr.fit(X[tr_ind, :], y[tr_ind])
    quals.append(metrics.roc_auc_score(y[test_ind], lr.predict_proba(X[test_ind,:])[:,1]))

# In[32]:


print 'Mean AUC-ROC =', np.mean(quals)
print 'AUC-ROC standart deviation =', np.std(quals)

# Также рассмотрим зависимость оценки качества от пропорции, в которой выборка разбивается на обучающую и тестовую:

# In[33]:


means = []
stds = []
xs = []
quals = []
for tr_size in np.arange(0.5,0.9,0.01):
    # set train size
    cv = cross_validation.ShuffleSplit(n=X.shape[0], n_iter = 5, train_size=tr_size, random_state=123)
    fold_quals = []
    lr = LogisticRegression()
    for tr_ind, test_ind in cv:
        # evaluate quality for each split
        lr.fit(X[tr_ind, :], y[tr_ind])
        qual = metrics.roc_auc_score(y[test_ind], lr.predict_proba(X[test_ind,:])[:,1])
        fold_quals.append(qual)
        quals.append(qual)
        xs.append(tr_size)
    # evaluation for current train set size
    means.append(np.mean(fold_quals))
    stds.append(np.std(fold_quals))

# In[34]:


figsize(6,4)
plt.plot(np.arange(0.5,0.9,0.01), means, 'ro')
plt.xlabel('train size')
plt.ylabel('AUC-ROC')

# In[35]:


plt.plot(np.arange(0.5,0.9,0.01), stds)
plt.xlabel('train size')
plt.ylabel('AUC-ROC std')

# ### Несбалансированные классы
# 
# Если объём выборки невелик, а объектов одного класса значительно меньше, чем другого, то может сложиться ситуация, когда при случайном разбиении объектов меньшего класса не окажется в тестовой выборке, в связи с чем результаты оценки качества будут неустойчивы.

# **Пример**: задача кредитного скоринга. Поскольку случаи невозврата кредита довольно редки, количество объектов отрицательного класса будет значительно меньше, чем положительного.

# In[36]:


df = pd.read_csv('data/givemesomecredit')
X = df.drop('SeriousDlqin2yrs', axis=1)
X = X.fillna(X.mean()).as_matrix()
y = df['SeriousDlqin2yrs']
print "Доля заемщиков, не вернувших кредит:", y.mean()

# - Всего 0.7% выборки составляют объекты положительного класса

# - В таком случае необходимо производить стратификацию, то есть разбивать отдельно объекты каждого класса на обучение и тест (сохраняя их доли).
# 
# Оценим влияние стратификации на оценку качества путем разбиения выборки $N=10$ раз на обучение и тест и последующего усреднения AUC-ROC на тестовой выборке:

# In[37]:


cv = cross_validation.ShuffleSplit(n=X.shape[0], n_iter = 10, train_size=0.8, random_state=124)
quals = []

lr = LogisticRegression()
for tr_ind, test_ind in cv:
    lr.fit(X[tr_ind, :], y[tr_ind])
    quals.append(metrics.roc_auc_score(y[test_ind], lr.predict_proba(X[test_ind,:])[:,1]))

print "AUC-ROC w/o stratification = ", np.mean(quals)
print "AUC-ROC std w/o stratification = ", np.std(quals)

# In[38]:


cv = cross_validation.StratifiedShuffleSplit(y, n_iter = 10, train_size=0.8, random_state=124)
quals = []
lr = LogisticRegression()
for tr_ind, test_ind in cv:
    lr.fit(X[tr_ind, :], y[tr_ind])
    quals.append(metrics.roc_auc_score(y[test_ind], lr.predict_proba(X[test_ind,:])[:,1]))

print "AUC-ROC with stratification = ", np.mean(quals)
print "AUC-ROC std with stratification = ", np.std(quals)

# Как мы видим, стратификация позволяет уменьшить дисперсию и более точно оценить качество

# #### Кросс-валидация
# Рассмотрим датасет о пациентах, больных бесплодием, содержащий около 100 объектов. Выборка небольшая, но из специфичной области, поэтому каждый объект может нести в себе важную информацию, влияющую на значение целевой переменной. В связи с этим при оценивании качества модели хотелось бы сделать обучающую выборку как можно больше. При этом из-за маленького объема всей выборки мы можем позволить себе обучать модель многократно. В данной ситуации для оценки качества можно использовать схему leave-one-out, реализованную в качестве объекта [`LeaveOneOut`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut).
# 
# Оценка качества производится следующим образом: каждый объект поочередно исключается из обучающей выборки, после чего модель обучается на всех остальных объектах, а качество измеряется на исключенном объекте (этот объект играет роль тестовой выборки); после того, как каждый объект был выкинут ровно по одному разу, итоговая оценка качества получается как среднее по всем полученным значениям функционала:
# 
# $$Q_{LOO}(X) = \frac{1}{l} \sum_{i=1}^l Q(\{x_i \}; a_{X \backslash \{x_i\} }),$$
# где
# * $X = \left\{ (x_i, y_i)\right\}_{i=1}^l$ — обучающая выборка;
# * $Q(X^{test}; a)$ — значение функционала алгоритма $a$ на контрольной выборке $X^{test}$;
# * $a_{X^{train}}(\cdot)$ — алгоритм, обученный на обучающей выборке $X^{train}.$

# In[39]:


df = pd.read_csv('data/fertility', sep = ',', header = None)
X = df.ix[:,:8]
y = np.zeros(df.shape[0])
y[df[9].as_matrix() == 'N'] = 1

cv_LOO = cross_validation.LeaveOneOut(n = len(y))
qual = []
for tr_ids, test_ids in cv_LOO:
    lr.fit(X.ix[tr_ids,:], y[tr_ids])
    qual.append(lr.predict(X.ix[test_ids, :]) == y[test_ids])
print "LOO accuracy =", np.mean(qual)


# С ростом размера выборки возрастают также и затраты на многократное обучение модели. Тем не менее, хотелось бы, чтобы каждый объект побывал и в обучающей, и в тестовой выборках, причём одинаковое количество раз. Чтобы удовлетворить этому условию, можно использовать схему K-fold кросс-валидации, реализованную в качестве объекта [`sklearn.cross_validation.KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold).
# 
# ![](http://scott.fortmann-roe.com/docs/docs/MeasuringError/crossvalidation.png)
# 
# В данном случае выборка разбивается на $K$ (примерно) одинаковых блоков, каждый из которых поочередно выступает в качестве контрольной выборки, а оставшиеся $K-1$ — в качестве обучающей, после чего оценки качества усредняются:
# 
# $$Q_{K-fold}(X) = \frac{1}{K} \sum_{k=1}^K Q(X^k; a_{X \backslash X^k }),$$
# где $X^k, k = \overline{1, K}$ — непересекающиеся блоки, на которые разбивается выборка $X$: $X = \sqcup_{k=1}^K X^k.$
# 

# In[40]:


cv = cross_validation.KFold(n=len(y), n_folds=5, random_state=123, shuffle=True)
lr = LogisticRegression()
quals = []
for tr_ids, test_ids in cv:
    lr.fit(X.ix[tr_ids,:], y[tr_ids])
    quals.append(metrics.accuracy_score(y[test_ids], lr.predict(X.ix[test_ids, :])))

print "K-fold accuracy =", np.mean(quals)

# ### Вопросы:
# 1. Как соотносятся кросс-валидация по схеме K-fold и схема leave-one-out?
# 2. Что общего у k-fold и ShuffleSplit?

# ### Настройка параметров модели
# 
# Настройка наилучших параметров модели производится с использованием техник кросс-валидации (для оценки качества) и соответственно перебора параметров. 
# 
# Для перебора параметров можно использовать: 
# - Модуль для перебора параметров модели по сетке [`sklearn.grid_search`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.grid_search)
#   - Класс [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) — полный перебор параметров модели по сетке
#   - Класс [`RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV) — перебирает случайные комбинации параметров

# ## Краткий справочник по SciKit-Learn
# 
# Обратите внимание на следующую функциональность библиотеки SciKit-Learn:
# 
# - Модуль для оценки качества моделей и кросс валидации [`sklearn.cross_validation`](http://scikit-learn.org/stable/modules/cross_validation.html):
#   - функция разбиения выборки [`train_test_split()`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html#sklearn.cross_validation.train_test_split)
#   - итераторы разбиений для кросс-валидации: [`LeaveOneOut`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut), [`KFold`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold), [`ShuffleSplit`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html#sklearn.cross_validation.ShuffleSplit)
#   - всегда указывайте параметр `random_state`, это сделает код вашего эксперимента воспроизводимым: при повторном запуске разбиения будут одинаковыми
# - Модуль с метриками качества моделей [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
#   - [`accuracy_score(y_true, y_pred)`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
#   - [`precision_recall_fscore_support(y_true, y_pred)`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)
#   - [`roc_curve(y_true, y_score)`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)
#   - [`precision_recall_curve(y_true, y_score)`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
#   - [`roc_auc_score(y_true, y_score)`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
# 
# - Модуль для перебора параметров модели по сетке [`sklearn.grid_search`](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.grid_search)
#   - Класс [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) — полный перебор параметров модели по сетке
#   - Класс [`RandomizedSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.RandomizedSearchCV.html#sklearn.grid_search.RandomizedSearchCV) — перебирает случайные комбинации параметров
#   
# ### Руководства и примеры из документации
# 
# - [Оценка качества предсказаний](http://scikit-learn.org/stable/modules/model_evaluation.html)
# - [Оптимизация параметров по сетке с использованием кросс-валидации](http://scikit-learn.org/stable/auto_examples/grid_search_digits.html)
# - [ROC-кривые](http://scikit-learn.org/stable/auto_examples/plot_roc.html), [Precision-Recall кривые](http://scikit-learn.org/stable/auto_examples/plot_precision_recall.html)
