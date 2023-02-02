import os
from google.colab import drive 

drive.mount('/content/gdrive')

Working_Directory = 'Shared drives/GitHub FONDOCYT/Yolov3_DeepSort/CPU' #@param {type:"string"}
wd="/content/gdrive/"+Working_Directory
os.chdir(wd)


dirpath = os.getcwd()
print("current directory is : " + dirpath)




#installing requirements
!pip install -r requirements.txt
from IPython.display import clear_output
#clear_output()
print("All Requirements were installed!")
#Downloading the weigts
!wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
# yolov3
!python load_weights.py
#clear_output()
print("Weights were loaded!")
# yolov3 on video
!python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.avi
#clear_output()
print('''
        Video object tracking is done, go to /data/video
        to see the resulting video...
''')
