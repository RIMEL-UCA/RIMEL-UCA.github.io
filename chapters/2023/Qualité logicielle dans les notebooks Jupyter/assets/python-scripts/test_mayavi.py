#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xvfbwrapper import Xvfb
vdisplay = Xvfb(width=1920, height=1080)
vdisplay.start()

from mayavi import mlab
mlab.init_notebook('x3d', 800, 450)
s = mlab.test_plot3d()
s
