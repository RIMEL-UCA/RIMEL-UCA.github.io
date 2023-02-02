#!/usr/bin/env python
# coding: utf-8

# # Visual Original Spectrogram

# In[10]:


from nlpaug.util.audio.loader import AudioLoader
from nlpaug.util.audio.visualizer import AudioVisualizer

path = 'Yamaha-V50-Rock-Beat-120bpm.wav'

data = AudioLoader.load_mel_spectrogram(path, n_mels=128)
AudioVisualizer.spectrogram('Original', data)

# In[2]:


import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

# # Frequency Masking

# In[3]:


aug = nas.FrequencyMaskingAug(zone=(0, 1))

aug_data = aug.substitute(data)
AudioVisualizer.spectrogram('Frequency Masking', aug_data)

# # Time Masking

# In[8]:


aug = nas.TimeMaskingAug(zone=(0, 1))

aug_data = aug.substitute(data)
AudioVisualizer.spectrogram('Time Masking', aug_data)

# # Loudness

# In[5]:


aug = nas.LoudnessAug()

aug_data = aug.substitute(data)
AudioVisualizer.spectrogram('Loudness', aug_data)

# # Combine Frequency Masking, Time Masking

# In[9]:


import nlpaug.flow as naf

flow = naf.Sequential([
    nas.FrequencyMaskingAug(), 
    nas.TimeMaskingAug(), 
])
aug_data = flow.augment(data)
AudioVisualizer.spectrogram('Combine Frequency Masking and Time Masking', aug_data)

# In[ ]:



