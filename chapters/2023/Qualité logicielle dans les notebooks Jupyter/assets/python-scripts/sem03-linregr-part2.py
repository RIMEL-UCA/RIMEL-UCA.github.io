#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение, ФКН ВШЭ
# 
# # Семинар 3

# В данном семинаре мы обсудим: 
# * Библиотеку для численных вычислений Numpy
# * Оптимальные константы и веса для функций потерь
# * Градиентный спуск &mdash; преимущества и некоторые модификации

# ## NumPy
# 
# **NumPy** — библиотека языка Python, позволяющая [удобно] работать с многомерными массивами и матрицами, содержащая математические функции. Кроме того, NumPy позволяет векторизовать многие вычисления, имеющие место в машинном обучении.
# 
#  - [numpy](http://www.numpy.org)
#  - [numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/)
#  - [100 numpy exercises](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')

%pylab inline

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

# #### Создание массивов
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

# * Создание последовательностей при помощи функций arange (в качестве параметров принимает левую и правую границы последовательности и **шаг**) и linspace (принимает левую и правую границы и **количество элементов**):

# In[12]:


np.arange(2, 20, 3) # аналогично стандартной функции range python, правая граница не включается

# In[13]:


np.arange(2.5, 8.7, 0.9) # но может работать и с вещественными числами

# In[14]:


np.linspace(2, 18, 14) # правая граница включается (по умолчанию)

# * Для изменения размеров существующего массива можно воспользоваться функцией reshape (при этом количество элементов должно оставаться неизменным):

# In[15]:


np.arange(9).reshape(3, 3)

# Вместо значения длины массива по одному из измерений можно указать -1 — в этом случае значение будет рассчитано автоматически:

# In[16]:


np.arange(8).reshape(2, -1)

# * Транспонирование существующего массива:

# In[17]:


C = np.arange(6).reshape(2, -1)
C

# In[18]:


C.T

# * Повторение существующего массива

# In[19]:


a = np.arange(3)
np.tile(a, (2, 2))

# In[20]:


np.tile(a, (4, 1))

# #### Базовые операции
# 
# * Базовые арифметические операции над массивами выполняются поэлементно:

# In[21]:


A = np.arange(9).reshape(3, 3)
B = np.arange(1, 10).reshape(3, 3)

# In[22]:


print(A)
print(B)

# In[23]:


A + B

# In[24]:


A * 1.0 / B

# In[25]:


A + 1

# In[26]:


3 * A

# In[27]:


A ** 2

# Отдельно обратим внимание на то, что умножение массивов также является **поэлементным**, а не матричным:

# In[28]:


A * B

# Для выполнения матричного умножения необходимо использовать функцию dot:

# In[29]:


A.dot(B)

# Поскольку операции выполняются поэлементно, операнды бинарных операций должны иметь одинаковый размер. Тем не менее, операция может быть корректно выполнена, если размеры операндов таковы, что они могут быть расширены до одинаковых размеров. Данная возможность называется [broadcasting](http://www.scipy-lectures.org/intro/numpy/operations.html#broadcasting):
# ![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/02.05-broadcasting.png)

# In[30]:


np.tile(np.arange(0, 40, 10), (3, 1)).T + np.array([0, 1, 2])

# * Некоторые операции над массивами (например, вычисления минимума, максимума, суммы элементов) выполняются над всеми элементами вне зависимости от формы массива, однако при указании оси выполняются вдоль нее (например, для нахождения максимума каждой строки или каждого столбца):

# In[31]:


A

# In[32]:


A.min()

# In[33]:


A.max(axis=1)

# In[34]:


A.sum(axis=1)

# #### Индексация
# 
# Для доступа к элементам может использоваться [много различных способов](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html), рассмотрим основные.
# 
# * Для индексации могут использоваться конкретные значения индексов и срезы (slice), как и в стандартных типах Python. Для многомерных массивов индексы для различных осей разделяются запятой. Если для многомерного массива указаны индексы не для всех измерений, недостающие заполняются полным срезом (:).

# In[35]:


a = np.arange(10)
a

# In[36]:


a[2:5]

# In[37]:


a[3:8:2]

# In[38]:


A = np.arange(81).reshape(9, -1)
A

# In[39]:


A[2:4]

# In[40]:


A[:, 2:4]

# In[41]:


A[2:4, 2:4]

# In[42]:


A[-1]

# * Также может использоваться индексация при помощи списков индексов (по каждой из осей):

# In[43]:


A = np.arange(81).reshape(9, -1)
A

# In[44]:


A[[2, 4, 5], [0, 1, 3]]

# * Может также применяться логическая индексация (при помощи логических массивов):

# In[45]:


A = np.arange(11)
A

# In[46]:


A[A % 5 != 3]

# In[47]:


A[np.logical_and(A != 7, A % 5 != 3)] # также можно использовать логические операции

# #### Зачем?
# 
# Зачем необходимо использовать NumPy, если существуют стандартные списки/кортежи и циклы?
# 
# Причина заключается в скорости работы. Попробуем посчитать скалярное произведение 2 больших векторов:

# In[48]:


SIZE = 10000000

A_quick_arr = np.random.normal(size = (SIZE,))
B_quick_arr = np.random.normal(size = (SIZE,))

A_slow_list, B_slow_list = list(A_quick_arr), list(B_quick_arr)

# In[49]:


%%time
ans = 0
for i in range(len(A_slow_list)):
    ans += A_slow_list[i] * B_slow_list[i]

# In[50]:


%%time
ans = sum([A_slow_list[i] * B_slow_list[i] for i in range(SIZE)])

# In[51]:


%%time
ans = np.sum(A_quick_arr * B_quick_arr)

# In[52]:


%%time
ans = A_quick_arr.dot(B_quick_arr)

# ## Градиентный спуск
# 
# Напомним, что в градиентном спуске значения параметров на следующем шаге получаются из значений параметров на текущем шаге смещением в сторону антиградиента функционала: 
# 
# $$w^{(t+1)} = w^{(t)} - \eta_t \nabla Q(w^{(t)}),$$
# где $\eta_t$ — длина шага градиентного спуска.

# ### Асимптотическая сложность 
# 
# Явная формула параметров линейной модели записывается как $w = (X^TX)^{-1}X^Ty$, и в ней присутствует обращение матрицы $X^TX$ — очень трудоёмкая операция при большом количестве признаков. Нетрудно подсчитать, что сложность вычислений $O(d^3 + d^2l)$. При решении задач такая трудоёмкость часто оказывается непозволительной, поэтому параметры ищут итерационными методами, стоимость которых меньше. Один из них — градиентный спуск.
# 
# Формула градиента функции ошибки в случае MSE, как мы вывели ранее, выглядит следующим образом:
# 
# $$\nabla Q(w) = 2X^T(Xw - y).$$
#  
# Сложность вычислений в данном случае $O(dl)$. Стохастический градиентный спуск отличается от обычного заменой градиента на несмещённую оценку по одному или нескольким объектам. В этом случае сложность становится $O(kd)$, где $k$ — количество объектов, по которым оценивается градиент, $k << l$. Это отчасти объясняет популярность стохастических методов оптимизации.

# ### Визуализация траекторий GD и SGD
# На простом примере разберём основные тонкости, связанные со стохастической оптимизацией.

# Сгенерируем матрицу объекты—признаки $X$ и вектор весов $w_{true}$, вектор целевых переменных $y$ вычислим как $Xw_{true}$ и добавим нормальный шум:

# In[66]:


n_features = 2
n_objects = 300
batch_size = 10
num_steps = 43

w_true = np.random.normal(size=(n_features, ))

X = np.random.uniform(-5, 5, (n_objects, n_features))
X *= (np.arange(n_features) * 2 + 1)[np.newaxis, :]  # for different scales
Y = X.dot(w_true) + np.random.normal(0, 1, (n_objects))
w_0 = np.random.uniform(-2, 2, (n_features))

# Обучим на полученных данных линейную регрессию для MSE при помощи полного градиентного спуска — тем самым получим вектор параметров.

# In[67]:


w = w_0.copy()
w_list = [w.copy()]
step_size = 1e-2

for i in range(num_steps):
    w -= 2 * step_size * np.dot(X.T, np.dot(X, w) - Y) / Y.shape[0]
    w_list.append(w.copy())
w_list = np.array(w_list)

# Покажем последовательность оценок параметров $w^{(t)}$, получаемых в ходе итераций. Красная точка — $w_{true}$.

# In[68]:


# compute level set
A, B = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

levels = np.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        w_tmp = np.array([A[i, j], B[i, j]])
        levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))


plt.figure(figsize=(15, 12))
plt.title('GD trajectory')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')
plt.xlim((w_list[:, 0].min() - 0.1, w_list[:, 0].max() + 0.1))
plt.ylim((w_list[:, 1].min() - 0.1, w_list[:, 1].max() + 0.1))
plt.gca().set_aspect('equal')

# visualize the level set
CS = plt.contour(A, B, levels, levels=np.logspace(0, 1, num=20), cmap=plt.cm.rainbow_r)
CB = plt.colorbar(CS, shrink=0.8, extend='both')

# visualize trajectory
plt.scatter(w_true[0], w_true[1], c='r')
plt.scatter(w_list[:, 0], w_list[:, 1])
plt.plot(w_list[:, 0], w_list[:, 1])

plt.show()

# На лекции обсуждалось, что градиент перпендикулярен линиям уровня. Это объясняет такие зигзагообразные траектории градиентного спуска. Для большей наглядности в каждой точке пространства посчитаем градиент функционала и покажем его направление.

# In[69]:


# compute level set
A, B = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
A_mini, B_mini = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 27))

levels = np.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        w_tmp = np.array([A[i, j], B[i, j]])
        levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))
        
# visualize the level set
plt.figure(figsize=(13, 9))
CS = plt.contour(A, B, levels, levels=np.logspace(-1, 1.5, num=30), cmap=plt.cm.rainbow_r)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
        
# visualize the gradients
gradients = np.empty_like(A_mini)
for i in range(A_mini.shape[0]):
    for j in range(A_mini.shape[1]):
        w_tmp = np.array([A_mini[i, j], B_mini[i, j]])
        antigrad = - 2*1e-3 * np.dot(X.T, np.dot(X, w_tmp) - Y) / Y.shape[0]
        plt.arrow(A_mini[i, j], B_mini[i, j], antigrad[0], antigrad[1], head_width=0.02)

plt.title('Antigradients demonstration')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')
plt.xlim((w_true[0] - 1.5, w_true[0] + 1.5))
plt.ylim((w_true[1] - .5, w_true[1] + .7))
plt.gca().set_aspect('equal')
plt.show()

# Визуализируем теперь траектории стохастического градиентного спуска, повторив те же самые действия, оценивая при этом градиент по подвыборке.

# In[70]:


w = w_0.copy()
w_list = [w.copy()]
step_size = 0.2

for i in range(num_steps):
    sample = np.random.randint(n_objects, size=batch_size)
    w -= 2 * step_size * np.dot(X[sample].T, np.dot(X[sample], w) - Y[sample]) / batch_size
    w_list.append(w.copy())
w_list = np.array(w_list)

# In[72]:


# compute level set
A, B = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

levels = np.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        w_tmp = np.array([A[i, j], B[i, j]])
        levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))


plt.figure(figsize=(13, 12))
plt.title('SGD trajectory')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')
plt.xlim((w_list[:, 0].min() - 0.1, w_list[:, 0].max() + 0.1))
plt.ylim((w_list[:, 1].min() - 0.1, w_list[:, 1].max() + 0.1))
plt.gca().set_aspect('equal')

# visualize the level set
CS = plt.contour(A, B, levels, levels=np.logspace(0, 1, num=20), cmap=plt.cm.rainbow_r)
CB = plt.colorbar(CS, shrink=0.8, extend='both')

# visualize trajectory
plt.scatter(w_true[0], w_true[1], c='r')
plt.scatter(w_list[:, 0], w_list[:, 1])
plt.plot(w_list[:, 0], w_list[:, 1])

plt.show()

# Как видно, метод стохастического градиента «бродит» вокруг оптимума. Это объясняется подбором шага градиентного спуска $\eta_k$. Дело в том, что для сходимости стохастического градиентного спуска для последовательности шагов $\eta_k$ должны выполняться [условия Роббинса-Монро](https://projecteuclid.org/download/pdf_1/euclid.aoms/1177729586):
# $$
# \sum_{k = 1}^\infty \eta_k = \infty, \qquad \sum_{k = 1}^\infty \eta_k^2 < \infty.
# $$
# Интуитивно это означает следующее: 
# 1. последовательность должна расходиться, чтобы метод оптимизации мог добраться до любой точки пространства, 
# 2. но расходиться не слишком быстро.

# Попробуем посмотреть на траектории SGD, последовательность шагов которой удовлетворяет условиям Роббинса-Монро:

# In[73]:


w = w_0.copy()
w_list = [w.copy()]
step_size_0 = 0.45

for i in range(num_steps):
    step_size = step_size_0 / ((i+1)**0.51)
    sample = np.random.randint(n_objects, size=batch_size)
    w -= 2 * step_size * np.dot(X[sample].T, np.dot(X[sample], w) - Y[sample]) / batch_size
    w_list.append(w.copy())
w_list = np.array(w_list)

# In[74]:


# compute level set
A, B = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))

levels = np.empty_like(A)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        w_tmp = np.array([A[i, j], B[i, j]])
        levels[i, j] = np.mean(np.power(np.dot(X, w_tmp) - Y, 2))


plt.figure(figsize=(13, 9))
plt.title('SGD trajectory')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')
plt.xlim((w_list[:, 0].min() - 0.1, w_list[:, 0].max() + 0.1))
plt.ylim((w_list[:, 1].min() - 0.1, w_list[:, 1].max() + 0.1))
#plt.gca().set_aspect('equal')

# visualize the level set
CS = plt.contour(A, B, levels, levels=np.logspace(0, 1, num=20), cmap=plt.cm.rainbow_r)
CB = plt.colorbar(CS, shrink=0.8, extend='both')

# visualize trajectory
plt.scatter(w_true[0], w_true[1], c='r')
plt.scatter(w_list[:, 0], w_list[:, 1])
plt.plot(w_list[:, 0], w_list[:, 1])

plt.show()

# ### Сравнение скоростей сходимости

# Последнее, что хотелось бы продемонстрировать — сравнение, насколько быстро достигают оптимума метод полного и стохастического градиентного спуска. Сгенерируем выборку и построим график зависимости функционала от итерации.

# In[79]:


# data generation
n_features = 50
n_objects = 1000
num_steps = 200
batch_size = 2

w_true = np.random.uniform(-2, 2, n_features)

X = np.random.uniform(-10, 10, (n_objects, n_features))
Y = X.dot(w_true) + np.random.normal(0, 5, n_objects)

# In[104]:


from scipy.linalg import norm

step_size_sgd = 1
step_size_gd = 1e-2
w_sgd = np.random.uniform(-4, 4, n_features)
w_gd = w_sgd.copy()
residuals_sgd = [np.mean(np.power(np.dot(X, w_sgd) - Y, 2))]
residuals_gd = [np.mean(np.power(np.dot(X, w_gd) - Y, 2))]

norm_sgd = []
norm_gd = []


for i in range(num_steps):
    step_size = step_size_sgd / ((i+1) ** 0.51)
    sample = np.random.randint(n_objects, size=batch_size)
    
    w_sgd -= 2 * step_size * np.dot(X[sample].T, np.dot(X[sample], w_sgd) - Y[sample]) / batch_size
    residuals_sgd.append(np.mean(np.power(np.dot(X, w_sgd) - Y, 2)))
    norm_sgd.append(norm(np.dot(X[sample].T, np.dot(X[sample], w_sgd) - Y[sample])))
    
    w_gd -= 2 * step_size_gd * np.dot(X.T, np.dot(X, w_gd) - Y) / Y.shape[0]
    residuals_gd.append(np.mean(np.power(np.dot(X, w_gd) - Y, 2)))
    norm_gd.append(norm(np.dot(X.T, np.dot(X, w_gd) - Y)))

# In[105]:


plt.figure(figsize=(13, 6))
plt.plot(range(num_steps+1), residuals_gd, label='Full GD')
plt.plot(range(num_steps+1), residuals_sgd, label='SGD')

plt.title('Empirial risk over iterations')
plt.xlim((-1, num_steps+1))
plt.legend()
plt.xlabel('Iter num')
plt.ylabel(r'Q($w$)')
plt.grid()
plt.show()

# In[110]:


plt.figure(figsize=(13, 6))
plt.plot(range(num_steps), norm_gd, label='Full GD')
plt.plot(range(num_steps), norm_sgd, label='SGD')

plt.title('Gradient norm over iterations')
plt.xlim((-1, num_steps+1))
plt.legend()
plt.xlabel('Iter num')
plt.ylabel(r'$||\nabla Q$($w$)||')
plt.grid()
plt.show()

# Как видно, GD буквально за несколько итераций оказывается вблизи оптимума, в то время как поведение SGD может быть весьма нестабильным. Как правило, для более сложных моделей наблюдаются ещё большие флуктуации в зависимости качества функционала от итерации при использовании стохастических градиентных методов. Путём подбора величины шага можно добиться лучшей скорости сходимости, и существуют методы, адаптивно подбирающие величину шага (AdaGrad, Adam, RMSProp).
