#!/usr/bin/env python
# coding: utf-8

# # 10-4-1 ImageFolder (1)

# In[ ]:


import torchvision
from torchvision import transforms

from torch.utils.data import DataLoader

# In[ ]:


from matplotlib.pyplot import imshow

# In[ ]:


trans = transforms.Compose([
    transforms.Resize((64,128))
])

train_data = torchvision.datasets.ImageFolder(root='custom_data/origin_data', transform=trans)

# In[ ]:


for num, value in enumerate(train_data):
    data, label = value
    print(num, data, label)
    
    if(label == 0):
        data.save('custom_data/train_data/gray/%d_%d.jpeg'%(num, label))
    else:
        data.save('custom_data/train_data/red/%d_%d.jpeg'%(num, label))
