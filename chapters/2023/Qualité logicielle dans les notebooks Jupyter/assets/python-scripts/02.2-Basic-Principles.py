#!/usr/bin/env python
# coding: utf-8

# <small><i>This notebook was put together by [Jake Vanderplas](http://www.vanderplas.com). Source and license info is on [GitHub](https://github.com/jakevdp/sklearn_tutorial/).</i></small>

# # Basic Principles of Machine Learning

# Here we'll dive into the basic principles of machine learning, and how to
# utilize them via the Scikit-Learn API.
# 
# After briefly introducing scikit-learn's *Estimator* object, we'll cover **supervised learning**, including *classification* and *regression* problems, and **unsupervised learning**, including *dimensinoality reduction* and *clustering* problems.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# ## The Scikit-learn Estimator Object
# 
# Every algorithm is exposed in scikit-learn via an ''Estimator'' object. For instance a linear regression is implemented as so:

# In[ ]:


from sklearn.linear_model import LinearRegression

# **Estimator parameters**: All the parameters of an estimator can be set when it is instantiated, and have suitable default values:

# In[ ]:


model = LinearRegression(normalize=True)
print(model.normalize)

# In[ ]:


print(model)

# **Estimated Model parameters**: When data is *fit* with an estimator, parameters are estimated from the data at hand. All the estimated parameters are attributes of the estimator object ending by an underscore:

# In[ ]:


x = np.arange(10)
y = 2 * x + 1

# In[ ]:


print(x)
print(y)

# In[ ]:


plt.plot(x, y, 'o');

# In[ ]:


# The input data for sklearn is 2D: (samples == 10 x features == 1)
X = x[:, np.newaxis]
print(X)
print(y)

# In[ ]:


# fit the model on our data
model.fit(X, y)

# In[ ]:


# underscore at the end indicates a fit parameter
print(model.coef_)
print(model.intercept_)

# The model found a line with a slope 2 and intercept 1, as we'd expect.

# ## Supervised Learning: Classification and Regression
# 
# In **Supervised Learning**, we have a dataset consisting of both features and labels.
# The task is to construct an estimator which is able to predict the label of an object
# given the set of features. A relatively simple example is predicting the species of 
# iris given a set of measurements of its flower. This is a relatively simple task. 
# Some more complicated examples are:
# 
# - given a multicolor image of an object through a telescope, determine
#   whether that object is a star, a quasar, or a galaxy.
# - given a photograph of a person, identify the person in the photo.
# - given a list of movies a person has watched and their personal rating
#   of the movie, recommend a list of movies they would like
#   (So-called *recommender systems*: a famous example is the [Netflix Prize](http://en.wikipedia.org/wiki/Netflix_prize)).
# 
# What these tasks have in common is that there is one or more unknown
# quantities associated with the object which needs to be determined from other
# observed quantities.
# 
# Supervised learning is further broken down into two categories, **classification** and **regression**.
# In classification, the label is discrete, while in regression, the label is continuous. For example,
# in astronomy, the task of determining whether an object is a star, a galaxy, or a quasar is a
# classification problem: the label is from three distinct categories. On the other hand, we might
# wish to estimate the age of an object based on such observations: this would be a regression problem,
# because the label (age) is a continuous quantity.

# ### Classification Example
# K nearest neighbors (kNN) is one of the simplest learning strategies: given a new, unknown observation, look up in your reference database which ones have the closest features and assign the predominant class.
# 
# Let's try it out on our iris classification problem:

# In[ ]:


from sklearn import neighbors, datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

# create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# fit the model
knn.fit(X, y)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
# call the "predict" method:
result = knn.predict([[3, 5, 4, 2],])

print(iris.target_names[result])

# You can also do probabilistic predictions:

# In[ ]:


knn.predict_proba([[3, 5, 4, 2],])

# In[ ]:


from fig_code import plot_iris_knn
plot_iris_knn()

# ---
# 
# #### Exercise
# 
# Use a different estimator on the same problem: ``sklearn.svm.SVC``.
# 
# *Note that you don't have to know what it is do use it. We're simply trying out the interface here*
# 
# *If you finish early, try to create a similar plot as above with the SVC estimator.*

# In[ ]:


from sklearn.svm import SVC

# In[ ]:




# ---

# ### Regression Example
# 
# One of the simplest regression problems is fitting a line to data, which we saw above.
# Scikit-learn also contains more sophisticated regression algorithms

# In[ ]:


# Create some simple data
import numpy as np
np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.randn(20)

plt.plot(X.squeeze(), y, 'o');

# As above, we can plot a line of best fit:

# In[ ]:


model = LinearRegression()
model.fit(X, y)

# Plot the data and the model prediction
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);

# Scikit-learn also has some more sophisticated models, which can respond to finer features in the data:

# In[ ]:


# Fit a Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)

# Plot the data and the model prediction
X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)

plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit);

# Whether either of these is a "good" fit or not depends on a number of things; we'll discuss details of how to choose a model later in the tutorial.

# ---
# 
# #### Exercise
# 
# Explore the ``RandomForestRegressor`` object using IPython's help features (i.e. put a question mark after the object).
# What arguments are available to ``RandomForestRegressor``?
# How does the above plot change if you change these arguments?
# 
# These class-level arguments are known as *hyperparameters*, and we will discuss later how you to select hyperparameters in the model validation section.
# 
# ---

# ## Unsupervised Learning: Dimensionality Reduction and Clustering
# 
# **Unsupervised Learning** addresses a different sort of problem. Here the data has no labels,
# and we are interested in finding similarities between the objects in question. In a sense,
# you can think of unsupervised learning as a means of discovering labels from the data itself.
# Unsupervised learning comprises tasks such as *dimensionality reduction*, *clustering*, and
# *density estimation*. For example, in the iris data discussed above, we can used unsupervised
# methods to determine combinations of the measurements which best display the structure of the
# data. As we'll see below, such a projection of the data can be used to visualize the
# four-dimensional dataset in two dimensions. Some more involved unsupervised learning problems are:
# 
# - given detailed observations of distant galaxies, determine which features or combinations of
#   features best summarize the information.
# - given a mixture of two sound sources (for example, a person talking over some music),
#   separate the two (this is called the [blind source separation](http://en.wikipedia.org/wiki/Blind_signal_separation) problem).
# - given a video, isolate a moving object and categorize in relation to other moving objects which have been seen.
# 
# Sometimes the two may even be combined: e.g. Unsupervised learning can be used to find useful
# features in heterogeneous data, and then these features can be used within a supervised
# framework.

# ### Dimensionality Reduction: PCA
# 
# Principle Component Analysis (PCA) is a dimension reduction technique that can find the combinations of variables that explain the most variance.
# 
# Consider the iris dataset. It cannot be visualized in a single 2D plot, as it has 4 features. We are going to extract 2 combinations of sepal and petal dimensions to visualize it:

# In[ ]:


X, y = iris.data, iris.target

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
pca.fit(X)
X_reduced = pca.transform(X)
print("Reduced dataset shape:", X_reduced.shape)

# In[ ]:


import pylab as plt
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
           cmap='RdYlBu')

print("Meaning of the 2 components:")
for component in pca.components_:
    print(" + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component,
                                            iris.feature_names)))

# #### Clustering: K-means
# 
# Clustering groups together observations that are homogeneous with respect to a given criterion, finding ''clusters'' in the data.
# 
# Note that these clusters will uncover relevent hidden structure of the data only if the criterion used highlights it.

# In[ ]:


from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0) # Fixing the RNG in kmeans
k_means.fit(X)
y_pred = k_means.predict(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred,
           cmap='RdYlBu');

# ### Recap: Scikit-learn's estimator interface
# 
# Scikit-learn strives to have a uniform interface across all methods,
# and we'll see examples of these below. Given a scikit-learn *estimator*
# object named `model`, the following methods are available:
# 
# - Available in **all Estimators**
#   + `model.fit()` : fit training data. For supervised learning applications,
#     this accepts two arguments: the data `X` and the labels `y` (e.g. `model.fit(X, y)`).
#     For unsupervised learning applications, this accepts only a single argument,
#     the data `X` (e.g. `model.fit(X)`).
# - Available in **supervised estimators**
#   + `model.predict()` : given a trained model, predict the label of a new set of data.
#     This method accepts one argument, the new data `X_new` (e.g. `model.predict(X_new)`),
#     and returns the learned label for each object in the array.
#   + `model.predict_proba()` : For classification problems, some estimators also provide
#     this method, which returns the probability that a new observation has each categorical label.
#     In this case, the label with the highest probability is returned by `model.predict()`.
#   + `model.score()` : for classification or regression problems, most (all?) estimators implement
#     a score method.  Scores are between 0 and 1, with a larger score indicating a better fit.
# - Available in **unsupervised estimators**
#   + `model.predict()` : predict labels in clustering algorithms.
#   + `model.transform()` : given an unsupervised model, transform new data into the new basis.
#     This also accepts one argument `X_new`, and returns the new representation of the data based
#     on the unsupervised model.
#   + `model.fit_transform()` : some estimators implement this method,
#     which more efficiently performs a fit and a transform on the same input data.

# ## Model Validation
# 
# An important piece of machine learning is **model validation**: that is, determining how well your model will generalize from the training data to future unlabeled data. Let's look at an example using the *nearest neighbor classifier*. This is a very simple classifier: it simply stores all training data, and for any unknown quantity, simply returns the label of the closest training point.
# 
# With the iris data, it very easily returns the correct prediction for each of the input points:

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
X, y = iris.data, iris.target
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)
y_pred = clf.predict(X)
print(np.all(y == y_pred))

# A more useful way to look at the results is to view the **confusion matrix**, or the matrix showing the frequency of inputs and outputs:

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))

# For each class, all 50 training samples are correctly identified. But this **does not mean that our model is perfect!** In particular, such a model generalizes extremely poorly to new data. We can simulate this by splitting our data into a *training set* and a *testing set*. Scikit-learn contains some convenient routines to do this:

# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)
print(confusion_matrix(ytest, ypred))

# This paints a better picture of the true performance of our classifier: apparently there is some confusion between the second and third species, which we might anticipate given what we've seen of the data above.
# 
# This is why it's **extremely important** to use a train/test split when evaluating your models.  We'll go into more depth on model evaluation later in this tutorial.

# ## Flow Chart: How to Choose your Estimator
# 
# This is a flow chart created by scikit-learn super-contributor [Andreas Mueller](https://github.com/amueller) which gives a nice summary of which algorithms to choose in various situations. Keep it around as a handy reference!

# In[ ]:


from IPython.display import Image
Image("http://scikit-learn.org/dev/_static/ml_map.png")

# Original source on the [scikit-learn website](http://scikit-learn.org/stable/tutorial/machine_learning_map/)

# ## Quick Application: Optical Character Recognition

# To demonstrate the above principles on a more interesting problem, let's consider OCR (Optical Character Recognition) – that is, recognizing hand-written digits.
# In the wild, this problem involves both locating and identifying characters in an image. Here we'll take a shortcut and use scikit-learn's set of pre-formatted digits, which is built-in to the library.

# ### Loading and visualizing the digits data
# 
# We'll use scikit-learn's data access interface and take a look at this data:

# In[ ]:


from sklearn import datasets
digits = datasets.load_digits()
digits.images.shape

# Let's plot a few of these:

# In[ ]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])

# Here the data is simply each pixel value within an 8x8 grid:

# In[ ]:


# The images themselves
print(digits.images.shape)
print(digits.images[0])

# In[ ]:


# The data for use in our algorithms
print(digits.data.shape)
print(digits.data[0])

# In[ ]:


# The target label
print(digits.target)

# So our data have 1797 samples in 64 dimensions.

# ### Unsupervised Learning: Dimensionality Reduction
# 
# We'd like to visualize our points within the 64-dimensional parameter space, but it's difficult to plot points in 64 dimensions!
# Instead we'll reduce the dimensions to 2, using an unsupervised method.
# Here, we'll make use of a manifold learning algorithm called *Isomap*, and transform the data to two dimensions.

# In[ ]:


from sklearn.manifold import Isomap

# In[ ]:


iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)

# In[ ]:


data_projected.shape

# In[ ]:


plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10));
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

# We see here that the digits are fairly well-separated in the parameter space; this tells us that a supervised classification algorithm should perform fairly well. Let's give it a try.

# ### Classification on Digits
# 
# Let's try a classification task on the digits. The first thing we'll want to do is split the digits into a training and testing sample:

# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=2)
print(Xtrain.shape, Xtest.shape)

# Let's use a simple logistic regression which (despite its confusing name) is a classification algorithm:

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

# We can check our classification accuracy by comparing the true values of the test set to the predictions:

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest, ypred)

# This single number doesn't tell us **where** we've gone wrong: one nice way to do this is to use the *confusion matrix*

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, ypred))

# In[ ]:


plt.imshow(np.log(confusion_matrix(ytest, ypred)),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted');

# We might also take a look at some of the outputs along with their predicted labels. We'll make the bad labels red:

# In[ ]:


fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(ypred[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == ypred[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])

# The interesting thing is that even with this simple logistic regression algorithm, many of the mislabeled cases are ones that we ourselves might get wrong!
# 
# There are many ways to improve this classifier, but we're out of time here. To go further, we could use a more sophisticated model, use cross validation, or apply other techniques.
# We'll cover some of these topics later in the tutorial.
