#!/usr/bin/env python
# coding: utf-8

# ## embeddings (adjacency)

# compute HH^T by [stacking H \[j\] 's vertically](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.vstack.html)

# In[19]:


'''
compute embeddings (adjacency) for each node considering each node type (as before)
'''

from scipy.sparse import vstack as s_vstack

embeddings = []
for i in range(3):
        '''
        there cannot be relationships between nodes of the same type
        e.g.: in a knowledge base, two subject nodes cannot be joined
        '''
        H_other = [H[j] for j in range(3) if j != i]
        H_stack = s_vstack(H_other)
        H_T = H_stack.T
        
        embeddings.append(H[i].dot(H_T))

#embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
