#!/usr/bin/env python
# coding: utf-8

# In[1]:


%config InlineBackend.figure_format = 'retina'
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
# import seaborn as sns

# In[2]:


os.chdir('./')
save_model_path = './'

# save all train test results
A = np.load(os.path.join(save_model_path, 'CRNN_varlen_epoch_training_loss.npy'))
B = np.load(os.path.join(save_model_path, 'CRNN_varlen_epoch_training_score.npy'))
C = np.load(os.path.join(save_model_path, 'CRNN_varlen_epoch_test_loss.npy'))
D = np.load(os.path.join(save_model_path, 'CRNN_varlen_epoch_test_score.npy'))
epochs = len(A)

# plot
fig = plt.figure(figsize=(16, 7))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)  # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")

# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B)  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)  # test loss (on epoch end)
plt.title("ResNet CRNN scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
save_file = "./loss_UCF101_CRNN.png"
plt.savefig(save_file, dpi=600)
# plt.close(fig)
plt.show()

# In[3]:


best_epoch = np.where(D==np.max(D))[0].item()
print('Best epoch: {}, validation accuracy: {:.2f}%'.format(best_epoch + 1, 100 * np.max(D)))

# In[ ]:



