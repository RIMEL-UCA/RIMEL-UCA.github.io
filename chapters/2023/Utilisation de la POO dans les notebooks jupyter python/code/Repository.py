
'''
THIS CODE TO PLOT SOME GRAPHS
'''
import json
import matplotlib.pyplot as plt
import re
from clone import g

def get_json_from_file(file_path):
    with open(file_path, 'r') as file:
        json_file = json.load(file)
    return json_file

def find_keyword_in_json(json_file, keyword):
    count = 0
    for repository in json_file:
        for notebook in json_file[repository]:
            for cell in notebook:
                for line in notebook[cell]:
                    for word in line:
                        if keyword in word:
                            count += 1
    return count

def find_keyword_in_json_file_per_repository(json_file, keyword):
    repos_keyword_dict = {}
    for repository in json_file:
        repos_keyword_dict[repository] = 0
        for notebook in json_file[repository]:
            for cell in notebook:
                for line in notebook[cell]:
                    for word in line:
                        if keyword in word:
                            repos_keyword_dict[repository] += 1
    return repos_keyword_dict

#regex_class = r'class\s\w+\s?\(?\w+\)?\s?:'
def find_class_definition_in_json_file_per_repository(json_file):
    repos_class_dict = {}
    for repository in json_file:
        repos_class_dict[repository] = 0
        for notebook in json_file[repository]:
            for cell in notebook:
                for line in notebook[cell]:
                    for word in line:
                        if re.search(r'class\s\w+\s?\(?\w+\)?\s?:', word):
                            repos_class_dict[repository] += 1
    return repos_class_dict

file_path = 'notebooks1.txt'
file = get_json_from_file(file_path)

result = find_keyword_in_json_file_per_repository(file, 'class')
print(find_keyword_in_json(file, 'class'))

#plot the result
plot = plt.bar(['class', 'def', 'init', 'self'], [find_keyword_in_json(file, ' class '), find_keyword_in_json(file, 'def'), find_keyword_in_json(file, 'init'), find_keyword_in_json(file, 'self')])
plt.show()

#plot the result per repository in a circle graph
plt.pie(result.values(), labels=result.keys(), autopct='%1.1f%%')
plt.show()

#plot the result per repository in a bar graph vertical name
plt.bar(result.keys(), result.values())
plt.xticks(rotation=90)
#increase the height of the graph
plt.subplots_adjust(bottom=0.4)
plt.show()

#plot the repos that use the keyword class and other that don't
repos_class = []
repos_no_class = []
for repository in file:
    if find_class_definition_in_json_file_per_repository(file)[repository] > 0:
        repos_class.append(repository)
    else:
        repos_no_class.append(repository)

print(repos_class)
print(repos_no_class)

#circle graph of repos that use the keyword class and other that don't
#plot the result per repository in a circle graph
plt.pie([len(repos_class), len(repos_no_class)], labels=['repos_class', 'repos_no_class'], autopct='%1.1f%%')
plt.show()

# plot the number of time a class is used in each repo in the repo_class list
plt.bar(repos_class, [find_class_definition_in_json_file_per_repository(file)[repository] for repository in repos_class])
plt.xticks(rotation=90)
#increase the height of the graph
plt.subplots_adjust(bottom=0.4)
plt.show()

list_of_repos_class = ['prateek1903/EDA-U.S-Accidents', 'solofruad/traffic-accidents', 'Dhananjana97/AccidentPrediction', 'AlexChicote/BarcelonaAccidents', 'greenwoodgiant/speeding-in-texas', 'olokshyn/motorbike-accidents', 'lopezbec/Traffic_Accident_Detection', 'amnag94/Accident-Prediction', 'VipinRao/Accident-Predictor', 'Rohansjamadagni/Accident-Detection', 'carlescarmonacalpe/eda_bcn_accidents', 'vulfalex/UK_accidents', 'GaryBarnes13/Accident-Severity']
list_of_repos_no_class = ['baixianghuang/travel', 'only-rohit/US-Accidents-Exploratory-Data-Analysis', 'shivambehl15/US-Accidents', 'bdan20/TrafficAccidents', 'physicsisawesome7/Accident-Classification', 'VishvajeetNimbalkar/Car-Accident', 'sagxam/accident-detection', 'acanales92/US-Accidents', 'ericstar20/AWS-EMR-US-Accidents', 'minashahabadi/US_accidents', 'maheshcheetirala/IBM-Car-AccidentseverityPrediction', 'rabindu123/Road-Accidents', 'wmymandy/TrafficAccidentPrediction', 'clarathays/US_Accidents', 'YaswanthReddy-UMBC/EDA_Accidents', 'AIdward/predict-traffic-accident', 'Saumil-Agarwal/Aircraft_Accident-Hackerearth', 'Sanikesh/Accident-Detection-YOLOv5', 'nasir11689/Accidents_data', 'LydiaO123/traffic-accident-project', 'Ovuowo-Rukevwe/US-Accident', 'Sachinnavgale/Aviation-Accident-Analysis', 'thonneau/gravite-accidents-circulation-france', 'shweteekta/Accident_detection', 'nark87/Coursera_Capstone', 'ziel5122/AccidentPrediction', 'dolongbien/HumanBehaviorBKU', 'LopezChris/Accident-Analysis', 'vgprasanna/Airplane-Accident', 'msammons82/DenverAccident', 'helmiwm/Traffic-Accident-Tweets-Classification', 'BhavnaM01/Accident-research', 'UTK-ML-Dream-Team/accident-severity-prediction', 'AbhinandGK/Prediction-for-Traffic-Accident-Severity', 'arkahome/Project_Accidents', 'vipinwako/Capstone-Project', 'ravij25/Peer-graded-Assignment-Capstone-Project---Car-accident-severity-', 'jillsergent/airline_accidents', 'dssg/rws_accident_prediction_public', 'escuderolucena/accidentes', 'iRaM-sAgOr/road_accident', 'pkallem/GoogleColab_BingeDrinking-AccidentsCorrelation', 'varun1524/accident_analysis', 'DeepthiAdarapu/UK-ACCIDENTS', 'mnovovil/AustinTrafficAccidentsToday', 'ayarelif/Car-Accident-in-USA', 'oaphyapran365/Road-accident-prediction-analysis-using-Machine-Learning', 'RonghuiZhou/us-accidents', 'xterm-hackslash/accident-detector', 'mprzybyla123/Traffic-Accident-Prediction', 'LIZABelkacem/Road_Accident_Prediction', 'MaxPowerfulness/Car-accident-prediction-model', 'MaximeTut/Analyse_accident', 'est987/BikeAccidentMapper', 'Aakash-Raman/Accident_Detection', 'Shoyan666/Car_Accident_analysis', 'BrittaInData/Road-Safety-UK', 'robertocmt/Accidentes', 'terurium/traffic_accident_survey', 'srisha24/Road_Accident_Prediction', 'deoldeep/Road-accidents', 'geethaguruju/road-accident', 'snehitvaddi/YOLOv3-Cloud-Based-Fire-Detection', 'lodi-m/US-Traffic-Accidents-Analysis', 'paavininanda/Car-accident-detection-and-notification', 'rajivkumar8532/ML-Project1', 'Rakhi098/Data-Analysis-US-accident-', 'vivekanandgoud/Kaggle_Accident-Severity', 'yatinkoul/Car-Accident-Severity-Prediction-Capstone-Project', 'sdgup/UK-Accidents', 'suraj-1-6-1/us-accidents', 'Axeldnahcram/France_accidents', 'divyar2630/Predict-Accident-Severity', 'trendct-data/aircraft-accidents', 'saifrais/w210-accident-detection', 'PriyaSangwan/RoadAccidents-Prevention-System', 'varungupta1405/sec_inc', 'tom-ambler/Coursera_Capstone', 'akshayshirsat94/Road-Accident-Severity', '92Amritpal/Road-Accidents', 'Josep-at-work/Traffic-Accident-Severity-Classification', 'vennela1115/Train-accident', 'NkululekoTech/Coursera_Capstone', 'sahiltindwani/DataScienceCourseraProject', 'varalakshmisept/Predicting-severity-of-the-accidents-using-US-Accidents-Datasets', 'KevinWu262/Accidents-Analysis', 'dijiaZhang1/Traffic-Accident-Fatality', 'pminchara/Accident-Detection', 'pokavv/pm_accident', 'G1993T/Saudi-Traffic-Accidents', 'AhmedNehro1994/Traffic-Accidents', 'jedidiah-oladele/Accidents-In-France', 'jpmrs1313/Time-Series', 'ertgrulyksk/UK-Traffic-Accidents', 'Patryk-Sl/car_accidents', 'chuntailin/Traffic_Accident_Analyse', 'icondor2019/oil_accidents', 'silentli/accident-prediction', 'kpatel3j/US-Accidents-Anlysis', 'NithyasriBabu/AccidentsDataViz', 'AmiraMT-cpu/ChocDetection-Project', 'oohyun15/Prediction-of-Traffic-Accident-Risk', 'itilakGH/car_accident_analysis_US', 'victoraccete/us-accidents', 'big-data-lab-team/accident-prediction-montreal', 'barazoulay/Accidents-report', 'sivasaba/Car-Accident-Severity-Prediction-Capstone', 'shorouq-alzaydi/AccidentSeverityPrediction', 'kate-stone/Exploration-of-NTSB-Accident-Data', 'nirajm09/Road-Accidents', 'dipin99/Accident-Prediction', 'mdcastille/ibm_capstone', 'iamajmalhassan/US_Accidents', 'johnsmithm/ner-accidente', 'Rutgers-Data-Science-Bootcamp/NYC_Bike_Accident_Analysis', 'hiren14/accident_detection', 'MedSun/RoadAccident', 'mihbort/car-accident-detection', 'skechung/TraffBot', 'DanielIzquierdo/accidentes-transito', 'vishalsoni7575/US-Accident-Analysis-', 'linaquiceno/ny_accidents', 'felluksch/traffic-accidents', 'WellersonPrenholato/filter-accidents', '5pratyusha/Analysis-on-road-accidents', 'RichardAbraham/Accident_Severity_UK_Classification', 'DennisPSmith5676/DataVisualization_US_Accidents', 'LuPa69/AccidentSeverity', 'Nishanth-K-S/US_Accidents_EDA', 'Aniruddha-Kulkarni/US-accidents', 'deven299/Crash', 'hugo-mi/SD701_Projet_Analyse_Accident_De_La_Route', 'ThusharKaranth/Data-Science-Project', 'sgs2892/DashcamAccidentPrediction', 'cepdnaclk/e17-co328-road-accident-analyzer', 'KostantinosKan/EDA-traffic_accidents', 'kiwi-pedia/US_Accidents', 'saivaruntejamudumba/Waze-Traffic-Analysis-for-Accident-Prediction', 'HIGHLYGAINED/Accident-Severity', 'Abhishek786singh/HackerEarth-Classifying-Airplane-Accidents', 'sundy1994/Project-US-accidents', 'bramstone/Predicting-US-Mining-Accidents', 'vgaurav3011/Accident-Detection-System-using-Image-Segmentation-and-Machine-Learning', 'Bot-Benedict/TeamMemoryLeak-Accidentalyzer_SCDFxIBM', 'lstark85/Project-2', 'TG20/traffic-accidents', 'kelvinchumbe/Airplane-Accident-Severity', 'ykarki1/US-Accidents', 'ghenimac/PredictAccidentSeverity', 'xSachinBharadwajx/Hacker_Earth_Airplane_Accident', 'syedusmanah/python-project', 'PriyanshiBhola/Accident-Detection', 'rtora/Sitting-in-Traffic-is-Terrible.-Is-there-anyway-to-avoid-it-', 'shubhamhegde/Data-Analysis-of-Road-Accidents', 'suyn1314/Traffic_accident', 'thmegy/AccidentsDeLaRoute', 'Qberto/analysis-arcgisAndPython-trafficCrashes', 'gustavo-ruizh/high-fatality-accidents-competition', 'AchyuthThimirishetty/Capstone-Project-Car_accident_severity', 'ShriyaRastogi/Coursera_Capstone', 'ImranRiazChohan/Aeroplane_accident_classification', 'jdvalenzuelah/accidentes-transito', 'Ecjoe/airplane-accident', 'yaz2/Coursera_Capstone', 'AhmadK98/UK-accidents-analysis', 'Brijeshbrijesh/Us-Accident-Analysis', 'jcorreac/UK-accidents', 'yashverma27/airplane_severity_of_accident', 'YounesAEO/yolov3-accident-detection', 'Cdev02/accidentes_bogota', 'foobarn/vic-accidents', 'LuisARG/EDA_AccidentesTrafico', 'AdeboyeML/Timeseries_Analysis_Forecast_UK_Road_Traffic_Accident', 'alex0440/Coursera_Capstone', 'GabrielEug2/data-science-accidents', 'Tanouzdogiparthi/Analysis-on-road-accidents-in-india', 'ekthaliz/accident', 'PitonEkrem/UK-Traffic-Accidents', 'Yoannli/Accident_Severity', 'PatrickBrayPersonal/us-accidents', 'AlejandroUPC/car_accidents', 'taticherla/Aviation-accident', 'VinodKumarJodu/RoadTrafficAccidents-App', 'ranjanj1/Traffic-Accident-Analysis-in-Washington-DC', 'RajatRasal/Road-Accidents-Analysis-AI-Hack', 'park-moonkyu/Seattle-Car-Severity', 'figuetbe/ChRoadAccidents', 'Ahmed471996/Road_Accidents', 'IngeonHwang/traffic_accident', 'MaaarcosG/AccidentesTransito', 'climberwb/airplane_accidents', 'magsoch/reducing-accident-analysis', 'jkgerig/nola-traffic-accidents', 'miquelpetit/uk-accidents-project', 'Mwaniki25/Twitter_Accident_Analysis', 'rhklite/traffic_accident_prediction', 'GayatriThati/-Coursera_Capstone', 'Aymericrag/Projet---Accidents', 'FabriceMesidor/TimeSeries_accident_UK', 'prateekb1912/car_accident_severity', 'Sagikap/Maritime-accidents-', 'shi02va/Aviation-accident', 'rdemedrano/xstnn', 'honeytung77/us-accidents2022', 'Siddhant026/UK-Accidents', 'Sarthak1904/traffic-management-based-pre-accident-and-post-accident-detection', 'AyeshaIshrathMN/Aviation-Accident', 'hyunchul1357/traffic-accident-analysis', 'fanshi118/CitiBike-and-Motor-Vehicle-Collisions', 'RohithKumar1999/Accident', 'jaaselam2000/Road-Traffic-Accident-Prediction-Model', 'AccidentalGuru/AccidentalGuru.github.io', 'soleilvertZZ/Coursera_Capstone', 'MulakaNaveen/Accident-severity', 'sudhirraj31/Road-Accident', 'Nrupesh29/train-accidents-analysis-spark', 'RuthwikBg/DMDD_Accidents', 'luozhongbin2017/Project1', 'LeanderHuyghe/Accident-Visualization', 'ykrasnikov/US-Traffic-Accidents', 'hareeshveettil/ML_RoadTrafficAcident_Severity_Classification', 'Robaie98/Accident_Severity', 'Surafeldemssie/NYC-Car-Accident', 'alexschalex/Bcn_Accidents', 'siddharthsethu/Aviation-Accident', 'manikantachowdhary/AccidentAnalysis', 'yjschein/Predicting_Car_Accident_Severity', 'Labheshm11/Unitedstates_Accident-analysis', 'sergio2526/Modelo-Accidentes', 'alba-lamas/AccidentsBCN', 'AttitudeAdjuster/Accident-Severity-Prediction', 'BinUmar13/ACCIDENTS', 'Jiekai77/US_Accidents_Analysis_Project', 'Joe-Bit-lab/Classifying-Chicago-Car-Accidents', 'sebastiandifrancesco/US_Accidents_2019-2020', 'michaelyipchen/cosevi-accidentes', 'akshay3236/Aviation-Accident', 'mukul20-21/Airplane-Accident-prediction-Hackerearth', 'prsh23/Aviation-Accident', 'MrRezoo/insurance-company', 'ashm8206/AirplaneAccidents', 'afaguilarr/accidentes_medellin', 'wckoeppen/whitewater-accidents', 'KDDS/DataScience-UK-AccidentDataAnalysis', 'vinemp/Accidenttwilio', 'radeeb/AirplaneAccidents', 'nadeemoffl/Airplane_Accident', 'Rudranu/Accident_Severity_prediction', 'jan-xu/accident-prediction', 'pateltejas968/Capstone-Project', 'bennyfungc/accident-severity', 'thaislins/traffic-accidents-brazil', 'leandromjunior/Accidents-EDA', 'Riteshchawla10/IBM-Data-Science-Capstone', 'Poorneswara-Prudhvi/Accidents', 'Abhidhy/accident-detection', 'nikhilmotiani/Applied-Data-Science-Capstone-', 'venkitaselvi/Us-accident', 'ishytasahore/How_Severe_can_an_Airplane_Accident_be', 'aremirata/uk_traffic_accidents', 'DanielAntonioMoreno/Coursera_Capstone', 'Dfriveraa/Accident-predictor', 'SimphiweK/Road-Accidents_Analytics', 'htefera/Accident-Analysis', 'emyhr/UK_accidents', 'noushinquazi/Accidental-Death-ML', 'Bielos/barranquilla-accidents', 'Krish98361/US-Accidents', 'rvlambda/RTA-MyProject', 'AMEERAZAM08/Road-Accident-Prediction-Using-ML', 'Ayh-l8/Capstone-Project---Car-accident-severity-Week-1-', 'bahadures/OIL_Spill_Accident-data', 'amit6895/US-Accidents-analysis', 'ansin218/accident-hotspot-prediction', 'shreyeah/accident_prediction', 'NagaJanakiDwadasi/US-Accident-Risk-Prediction', 'jeonprize/Accident', 'kbbaldha/AccidentAnalysis', 'BrunoBVR/projectAccidents', 'KonstantinMack/accidents_uk', 'shinz4u/barcelona_accidents', 'FGholiejad/munich_accident', 'AliZaiN-157/us-accident-analysis', 'Asutoshpadhy1/us_accidents', 'programeralebrije/US-Accidents', 'patrick4488/Crash_and_Summons_ML', 'aabritidutta/US-Accidents-Analysis']

###########################"
# get repositories stars and forks
# for class
repositories_stars_class = {}
repositories_forks_class = {}
for repository in list_of_repos_class:
    repositories_stars_class[repository] = g.get_repo(repository).stargazers_count
    repositories_forks_class[repository] = g.get_repo(repository).forks_count

# for no class
repositories_stars_no_class = {}
repositories_forks_no_class = {}
for repository in list_of_repos_no_class:
    repositories_stars_no_class[repository] = g.get_repo(repository).stargazers_count
    repositories_forks_no_class[repository] = g.get_repo(repository).forks_count

# plot the number of stars and forks for each repository
# for class
plt.bar(repositories_stars_class.keys(), repositories_stars_class.values())
plt.xticks(rotation=90)
#increase the height of the graph
plt.subplots_adjust(bottom=0.4)
plt.show()

plt.bar(repositories_forks_class.keys(), repositories_forks_class.values())
plt.xticks(rotation=90)
#increase the height of the graph
plt.subplots_adjust(bottom=0.4)
plt.show()



















