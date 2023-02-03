#!/usr/bin/env python
# coding: utf-8

# In[1]:


%config InlineBackend.figure_format = 'retina'
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# In[2]:


os.chdir('./')

# save all train test results
A = np.load('./CRNN_epoch_training_losses.npy')
B = np.load('./CRNN_epoch_training_scores.npy')
C = np.load('./CRNN_epoch_test_loss.npy')
D = np.load('./CRNN_epoch_test_score.npy')

epochs = len(A)

# plot
fig = plt.figure(figsize=(16, 7))
plt.subplot(121)
plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), D)         # test loss (on epoch end)
# plt.plot(histories.losses_val)
plt.title("model scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
save_file = "./loss_UCF101_CRNN.png"
plt.savefig(save_file, dpi=600)
# plt.close(fig)
plt.show()

# In[3]:


best_epoch = np.where(D==np.max(D))[0].item()
print('Best epoch: {}, validation accuracy: {:.2f}%'.format(best_epoch, 100 * np.max(D)))

# In[ ]:



