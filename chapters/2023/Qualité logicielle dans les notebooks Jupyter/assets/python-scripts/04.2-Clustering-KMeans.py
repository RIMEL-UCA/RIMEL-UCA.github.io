#!/usr/bin/env python
# coding: utf-8

# <small><i>This notebook was put together by [Jake Vanderplas](http://www.vanderplas.com). Source and license info is on [GitHub](https://github.com/jakevdp/sklearn_tutorial/).</i></small>

# # Clustering: K-Means In-Depth

# Here we'll explore **K Means Clustering**, which is an unsupervised clustering technique.
# 
# We'll start with our standard set of initial imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn')

# ## Introducing K-Means

# K Means is an algorithm for **unsupervised clustering**: that is, finding clusters in data based on the data attributes alone (not the labels).
# 
# K Means is a relatively easy-to-understand algorithm.  It searches for cluster centers which are the mean of the points within them, such that every point is closest to the cluster center it is assigned to.
# 
# Let's look at how KMeans operates on the simple clusters we looked at previously. To emphasize that this is unsupervised, we'll not plot the colors of the clusters:

# In[ ]:


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50);

# By eye, it is relatively easy to pick out the four clusters. If you were to perform an exhaustive search for the different segmentations of the data, however, the search space would be exponential in the number of points. Fortunately, there is a well-known *Expectation Maximization (EM)* procedure which scikit-learn implements, so that KMeans can be solved relatively quickly.

# In[ ]:


from sklearn.cluster import KMeans
est = KMeans(4)  # 4 clusters
est.fit(X)
y_kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow');

# The algorithm identifies the four clusters of points in a manner very similar to what we would do by eye!

# ## The K-Means Algorithm: Expectation Maximization
# 
# K-Means is an example of an algorithm which uses an *Expectation-Maximization* approach to arrive at the solution.
# *Expectation-Maximization* is a two-step approach which works as follows:
# 
# 1. Guess some cluster centers
# 2. Repeat until converged
#    A. Assign points to the nearest cluster center
#    B. Set the cluster centers to the mean 
#    
# Let's quickly visualize this process:

# In[ ]:


from fig_code import plot_kmeans_interactive
plot_kmeans_interactive();

# This algorithm will (often) converge to the optimal cluster centers.

# ### KMeans Caveats
# 
# The convergence of this algorithm is not guaranteed; for that reason, scikit-learn by default uses a large number of random initializations and finds the best results.
# 
# Also, the number of clusters must be set beforehand... there are other clustering algorithms for which this requirement may be lifted.

# ## Application of KMeans to Digits
# 
# For a closer-to-real-world example, let's again take a look at the digits data. Here we'll use KMeans to automatically cluster the data in 64 dimensions, and then look at the cluster centers to see what the algorithm has found.

# In[ ]:


from sklearn.datasets import load_digits
digits = load_digits()

# In[ ]:


est = KMeans(n_clusters=10)
clusters = est.fit_predict(digits.data)
est.cluster_centers_.shape

# We see ten clusters in 64 dimensions. Let's visualize each of these cluster centers to see what they represent:

# In[ ]:


fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

# We see that *even without the labels*, KMeans is able to find clusters whose means are recognizable digits (with apologies to the number 8)!
# 
# The cluster labels are permuted; let's fix this:

# In[ ]:


from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# For good measure, let's use our PCA visualization and look at the true cluster labels and K-means cluster labels:

# In[ ]:


from sklearn.decomposition import PCA

X = PCA(2).fit_transform(digits.data)

kwargs = dict(cmap = plt.cm.get_cmap('rainbow', 10),
              edgecolor='none', alpha=0.6)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels, **kwargs)
ax[0].set_title('learned cluster labels')

ax[1].scatter(X[:, 0], X[:, 1], c=digits.target, **kwargs)
ax[1].set_title('true labels');

# Just for kicks, let's see how accurate our K-Means classifier is **with no label information:**

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(digits.target, labels)

# 80% – not bad! Let's check-out the confusion matrix for this:

# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(digits.target, labels))

plt.imshow(confusion_matrix(digits.target, labels),
           cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted');

# Again, this is an 80% classification accuracy for an **entirely unsupervised estimator** which knew nothing about the labels.

# ## Example: KMeans for Color Compression
# 
# One interesting application of clustering is in color image compression. For example, imagine you have an image with millions of colors. In most images, a large number of the colors will be unused, and conversely a large number of pixels will have similar or identical colors.
# 
# Scikit-learn has a number of images that you can play with, accessed through the datasets module. For example:

# In[ ]:


from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
plt.imshow(china)
plt.grid(False);

# The image itself is stored in a 3-dimensional array, of size ``(height, width, RGB)``:

# In[ ]:


china.shape

# We can envision this image as a cloud of points in a 3-dimensional color space. We'll rescale the colors so they lie between 0 and 1, then reshape the array to be a typical scikit-learn input:

# In[ ]:


X = (china / 255.0).reshape(-1, 3)
print(X.shape)

# We now have 273,280 points in 3 dimensions.
# 
# Our task is to use KMeans to compress the $256^3$ colors into a smaller number (say, 64 colors). Basically, we want to find $N_{color}$ clusters in the data, and create a new image where the true input color is replaced by the color of the closest cluster.
# 
# Here we'll use ``MiniBatchKMeans``, a more sophisticated estimator that performs better for larger datasets:

# In[ ]:


from sklearn.cluster import MiniBatchKMeans

# In[ ]:


# reduce the size of the image for speed
n_colors = 64

X = (china / 255.0).reshape(-1, 3)
    
model = MiniBatchKMeans(n_colors)
labels = model.fit_predict(X)
colors = model.cluster_centers_
new_image = colors[labels].reshape(china.shape)
new_image = (255 * new_image).astype(np.uint8)

# create and plot the new image
with plt.style.context('seaborn-white'):
    plt.figure()
    plt.imshow(china)
    plt.title('input: 16 million colors')

    plt.figure()
    plt.imshow(new_image)
    plt.title('{0} colors'.format(n_colors))

# Compare the input and output image: we've reduced the $256^3$ colors to just 64.
