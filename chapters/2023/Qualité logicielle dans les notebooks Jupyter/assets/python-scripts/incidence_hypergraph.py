#!/usr/bin/env python
# coding: utf-8

# ## incidence matrix of hypergraph
# 
# credits: [DHNE](https://github.com/tadpole/DHNE)

# each triple is a hyperedge

# each component type is a node
# 
# e.g.: for a knowledge base, each subject is a node, each relation is a node and each object is a node

# ### use [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) to store data

# In[18]:


'''
compute incident matrix for each node type
e.g.: for a knowledge base, there are three node types - subject type, relation type, object type
'''
from scipy.sparse import csr_matrix

H = [] #incidence matrix
E = nums_examples #each example is a hyperedge

for i in range(3):
    
    data = np.ones(E)
    row = edge[:, i]
    col = range(E)
    V = nums_type[i]
    
    csr_H = csr_matrix((data, (row, col)), shape=(V,E))
    H.append(csr_H)
    

'''
H = [csr_matrix((np.ones(nums_examples), 
                 (edge[:, i], range(nums_examples))), 
                shape=(nums_type[i], num_hyperedges)) for i in range(3)]
'''
