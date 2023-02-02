#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
a=np.loadtxt("whlt.txt", dtype={'names':('w','h','l','t'), 'formats':('f4','f4','f4','S10')})

# In[13]:


a['t']
606   b'Truck'
14385 b'Car'
893   b'Cyclist'
2280  b'Pedestrian'
1617  b'Van'
636   b'Misc'
287   b'Tram'
0     b'Person_sitting'

# In[12]:


f, axes = plt.subplots(7, 3, figsize=(10, 30), sharex=False)
i=0
c=('r','g','b')
for t in (b'Truck', b'Car', b'Cyclist',b'Pedestrian', b'Van', b'Misc', b'Tram', b'Person_sitting'):
    ind=np.where(a['t']==t)[0]
    print(len(ind),t)
    if len(ind)==0:
        continue
    j=0
    for e in ('w','h','l'):
        axes[i,j].hist(a[e][ind],bins=4*(j+1),color=c[j])
        axes[i,j].set_title(str(t)+' '+e)
        j+=1
    i+=1
f.savefig("dist.pdf")

# In[20]:


len(a['w'][ind])

# In[ ]:



