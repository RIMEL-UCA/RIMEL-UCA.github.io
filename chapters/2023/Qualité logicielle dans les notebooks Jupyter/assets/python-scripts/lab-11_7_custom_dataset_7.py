#!/usr/bin/env python
# coding: utf-8

# #    모두를 위한 딥러닝 시즌2 pytorch
#     *** Custom Dataset은 어떻게 쓰나요? (7)***
# 
#         6번째 파일에서는 학습된 모델을 저장하고 불러오는 방법까지 배워봤습니다. 
#         7번째로 완성된 모델의 성능을 평가하는 것을 진행해볼겁니다. 
#         모델의 성능평가를 위해서는 성능평가를 진행할 테스트 데이터 셋이 필요합니다. 
#         
#         데이터셋에는 3가지 종류가 있는데 
#         
#         Training dataset : 학습을 위한 Dataset
#         Validation dataset : 학습 중에 학습이 잘 되고 있는지 확인 하는 Dataset, Training dataset의 일부를 사용
#         Test dataset : 학습이 완료된 이후에 학습이 잘 되었는지 확인하는 Dataset
#         
#         위와 같습니다. 
#         
#         제가 작성한 코드에는 아쉽게도 Validation dataset은 없습니다. 
#         기존에 가지고 있는 dataset이 매우 작기 때문입니다. 
#         
#         아래에서는 testset을 이용한 평가를 진행하기 때문에 기존에 사용했던 train_data 및 train_dataset은 사용하지 않습니다. 

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision

from CNN_ksg.model import model

# In[ ]:


trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,128)),
    torchvision.transforms.ToTensor()
])

# 위에서 말씀드린대로 아래에서는 test dataset을 사용합니다. 

# In[ ]:


test_data = torchvision.datasets.ImageFolder(root='./CNN_ksg/test_data', transform=trans)
testloader = DataLoader(dataset=test_data, batch_size=8, shuffle=True, num_workers=4)

# In[ ]:


length = len(testloader)
print(length)

# In[ ]:


pre_train_net = model.NN()
pre_train_net.load_state_dict(torch.load('./CNN_ksg/model/model.pth'))

# In[ ]:


device='cuda'
pre_train_net = pre_train_net.to(device)

# In[ ]:


# 몇개 맞았는지 저장할 변수
correct = 0
# 전체 개수를 저장할 변수
total = 0

# In[ ]:


for num, data in enumerate(testloader):
    inputs, labels = data
    inputs = inputs.to(device)

    out = pre_train_net(inputs)
    _, predicted = torch.max(out,1)

    #torch.Tensor.cuda()하고 torch.Tensor()는 비교가 안됩니다.
    #따라서 .cpu() 를 이용해서 바꿔주세요
    predicted = predicted.cpu()
    total += labels.size(0)

    #잘 맞추고 있는지 궁금하면 아래 print를 출력해 보세요
    #print(predicted, labels)

    correct += (predicted == labels).sum().item()

# In[ ]:


print('Accuracy of the network on the 50 test images : %d %%'%(100* correct /total))

# In[ ]:



