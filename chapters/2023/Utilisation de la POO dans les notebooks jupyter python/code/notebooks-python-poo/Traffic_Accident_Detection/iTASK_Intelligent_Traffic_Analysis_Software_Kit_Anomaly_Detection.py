!git clone https://github.com/lopezbec/AI_City_2020_iTaskT4.git
%mv AI_City_2020_iTaskT4 AI_City_2020
%cd AI_City_2020/track4-traffic-anomaly-detection

!pip install colour
!sudo apt-get install poppler-utils
!sudo apt autoremove

import os
from pathlib import Path

def setup_preprocessed_data():
  %cd preprocessed_data/
  !gdown  --id 1sPRjyGdU1rBI3a75EatsMPx5SWyCT7GM
  !unzip -qq -o Data.zip
  %rm Data.zip
  %cd ..


#@markdown **By default, this comes with a small dataset of videos to detect
#@markdown anomalies**


!sed -n "4 c\dataset_path = r'Datasets/AI city challenge/AIC20_track4/test-data'" "Config.py"
!sed -n "5 c\data_path = os.path.dirname(os.path.abspath(__file__)) + '/preprocessed_data'" "Config.py"
!sed -n "6 c\output_path = data_path + '/output_demo'" "Config.py"

dirs = [x for x in Path('preprocessed_data').iterdir() if x.is_dir()]
if len(dirs)  < 3:
    setup_preprocessed_data()

print("-"*30, "Default configuration", "-"*30, "\n\n")
!cat Config.py
!python "Test.py"
!python "ResultRefinement.py"
import re
from typing import Union, List
def natural_keys(path: Path) -> List[Union[int, str]]:
    """Sort path names by its cardinal numbers.

    Parameters
    ----------
    path : pathlib.Path
      The element to be sorted.
    """

    def atoi(c: str) -> Union[int, str]:
        """Try to convert a character to an int if possible.

        Parameters
        ----------
        c : str
          The character to check if it's int.
        """
        return int(c) if c.isdigit() else c

    return [atoi(c) for c in re.split('(\d+)', path.stem)]

#@markdown Now you can see the output within
#@markdown `AI_City_2020/track4-traffic-anomaly-detection/preprocessed_data/output_demo`.
#@markdown You can download the results to your computer or visualize the results on this cell.
from google.colab import files
Download_output_demo_zip = False #@param {type:"boolean"}
if Download_output_demo_zip:
  !zip output_demo.zip -r preprocessed_data/output_demo
  files.download('output_demo.zip')

else:
  import cv2
  import matplotlib.pyplot as plt
  import pandas as pd
  from IPython.display import display
  from google.colab.patches import cv2_imshow

  dirs = [x for x in Path('preprocessed_data/output_demo').iterdir() if x.is_dir()]
  dirs.sort(key=natural_keys)
  print("-"*33, "Videos with anomalies found", "-"*33, "\n\n")
  for dir in dirs:  
    dir = dir.resolve()

    print("-"*45, dir.name, "-"*45)
      # Dataframe with scenes found with anomalies
    anomaly_file = str(dir/"anomaly_events.txt")
    df_results = pd.read_csv(anomaly_file, sep=" ",
                             names=["video_id", "scene_id", "start_time",
                                    "end_time", "confident_score"])\
                                    .drop(columns=["video_id"])

    # Choose the first frame from the anomalies found
    frame = f"{df_results.iloc[0, 1]:03d}"
    events_jpg = next(dir.glob("".join(["**/", "events", frame, ".jpg"])))
    event_im = cv2.imread(str(events_jpg))
    cv2_imshow(event_im)
    cv2.waitKey(1) & 0xff
      
    display(df_results)

    # Display the figure within the PDF of anomalies
    anomaly_pdf = str(dir/(dir.name + "_anomaly.pdf"))
    anomaly_png = dir/(dir.name + "_anomaly.png")
    !pdftoppm -png {anomaly_pdf} {anomaly_png}
    anomaly_png = list(dir.glob("*.png"))[0]
    img = cv2.imread(str(anomaly_png))
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2_imshow(img)
    cv2.destroyAllWindows()
    !rm {anomaly_png}

  print("-"*97, "\n")
  print("Summary of videos with anomalies")
  df_result_all = pd.read_csv(dirs[0].parent/"result_all.txt", sep=" ",
                               names=["video_id", "time", "confidence"])
  
  display(df_result_all)


