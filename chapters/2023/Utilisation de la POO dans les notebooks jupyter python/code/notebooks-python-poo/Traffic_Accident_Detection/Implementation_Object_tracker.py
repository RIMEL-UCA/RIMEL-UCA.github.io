
Use_Google_Drive= False #@param {type:"boolean"}

if Use_Google_Drive:
    import os
    from google.colab import drive 

    drive.mount('/content/gdrive',force_remount=True)

    Working_Directory = 'My Drive' #@param {type:"string"}
    wd="/content/gdrive/"+Working_Directory
    os.chdir(wd)

    !git clone https://github.com/lopezbec/Traffic_Accident_Detection

    
    %cd Traffic_Accident_Detection/Yolov3_DeepSort/GPU 
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    
else:
    !git clone https://github.com/lopezbec/Traffic_Accident_Detection
    %cd Traffic_Accident_Detection/Yolov3_DeepSort/GPU 



#installing requirements
!pip install -r requirements-gpu.txt
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
!python object_tracker.py --video ./data/video/test.mp4 --output ./data/video/results.mp4
#clear_output()
print('''
        Video object tracking is done, go to /data/video
        to see the resulting video...
''')
%cd data/video

Download_video_and_json= True #@param {type:"boolean"}

if Download_video_and_json:
    from google.colab import files
    files.download('results.mp4') 
    %cd ..
    %cd ..
    files.download('info.json') 
else:
    from google.colab.patches import cv2_imshow
    import cv2

    # Download sample video


    cap = cv2.VideoCapture('results.mp4')
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        cv2_imshow(image) # Note cv2_imshow, not cv2.imshow

        cv2.waitKey(1) & 0xff

    cv2.destroyAllWindows()
    cap.release()

