%tensorflow_version 1.x
from IPython.display import clear_output
!pip3 uninstall -y keras
!pip3 install keras==2.1.0
#clear_output()
print("Installed!")
!git clone https://github.com/matterport/Mask_RCNN
%cd Mask_RCNN
!pip3 install -r requirements.txt
!python3 setup.py install
#clear_output()
print("Done!")
!git clone https://github.com/cocodataset/cocoapi.git
%cd cocoapi/PythonAPI
!make
%cd ../../../
#clear_output()
print("Done!")
Use_Google_Drive= False #@param {type:"boolean"}

if Use_Google_Drive:
    import os
    from google.colab import drive 

    drive.mount('/content/gdrive',force_remount=True)

    Working_Directory = 'My Drive' #@param {type:"string"}
    wd="/content/gdrive/"+Working_Directory
    os.chdir(wd)

    !git clone https://github.com/lopezbec/Traffic_Accident_Detection

    
    %cd Traffic_Accident_Detection/MaskRCNN-ODS 
    dirpath = os.getcwd()
    print("current directory is : " + dirpath)

    
else:
    !git clone https://github.com/lopezbec/Traffic_Accident_Detection
    %cd Traffic_Accident_Detection/MaskRCNN-ODS
!python video_detection.py -v images/test.mp4 -sp images/result.mp4
%cd images/

Download_video_and_json= True #@param {type:"boolean"}

if Download_video_and_json:
    from google.colab import files
    files.download('result.mp4') 
    %cd ..
    files.download('info.json') 
else:
    from google.colab.patches import cv2_imshow
    import cv2

    # Download sample video


    cap = cv2.VideoCapture('result.mp4')
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        cv2_imshow(image) # Note cv2_imshow, not cv2.imshow

        cv2.waitKey(1) & 0xff

    cv2.destroyAllWindows()
    cap.release()
