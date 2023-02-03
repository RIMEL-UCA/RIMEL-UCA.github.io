!git clone https://github.com/lopezbec/AICity-2020-CETCVLAB_ADO.git
%mv AICity-2020-CETCVLAB_ADO Anomaly_detection
%cd Anomaly_detection
%ls
!wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
!pip install imageai --upgrade
!pip install scipy
!pip install matplotlib
!pip install opencv-python
!python normdetect.py
!python create_bg.py
!python bgnormdetect.py
!python CombinedExtractor.py normal
!python zoomdetect.py
!python bgcropdetect.py
!python CombinedExtractor.py
try:
    import Image
except ImportError:
    from PIL import Image

import cv2
from google.colab.patches import cv2_imshow

print("Minute Mask")
cv2_imshow(cv2.imread('./MinuteMask/1/1.png', cv2.IMREAD_UNCHANGED))
