'''
THIS CODE IS FORE SAVING THE NOTEBOOKS IN A FOLDER FOR SONARCUBE METRICS
NEED TO CONVERT ALL .ipynb FILES TO .py FILES
'''
import json
import os
from github import Github


g = Github("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")


#list_repositories_class = ['prateek1903/EDA-U.S-Accidents', 'solofruad/traffic-accidents', 'Dhananjana97/AccidentPrediction', 'AlexChicote/BarcelonaAccidents', 'greenwoodgiant/speeding-in-texas', 'olokshyn/motorbike-accidents', 'lopezbec/Traffic_Accident_Detection', 'amnag94/Accident-Prediction', 'VipinRao/Accident-Predictor', 'Rohansjamadagni/Accident-Detection', 'carlescarmonacalpe/eda_bcn_accidents', 'vulfalex/UK_accidents', 'GaryBarnes13/Accident-Severity']
#list_repositories_class = ['Rohansjamadagni/Accident-Detection']
list_repositories_no_poo = ['baixianghuang/travel', 'only-rohit/US-Accidents-Exploratory-Data-Analysis', 'shivambehl15/US-Accidents', 'bdan20/TrafficAccidents', 'physicsisawesome7/Accident-Classification', 'VishvajeetNimbalkar/Car-Accident', 'sagxam/accident-detection', 'acanales92/US-Accidents', 'ericstar20/AWS-EMR-US-Accidents', 'minashahabadi/US_accidents', 'maheshcheetirala/IBM-Car-AccidentseverityPrediction', 'rabindu123/Road-Accidents', 'wmymandy/TrafficAccidentPrediction']

def fetch_notebooks(repository):
    try:
        repo = g.get_repo(repository)
    except:
        print("Error: " + repository)
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if file_content.path.endswith('.ipynb'):
                path = os.path.join(os.getcwd(), 'notebooks-python-no-poo')
                os.makedirs(path + '/' + repository.split('/')[1], exist_ok=True)
                #save cells code as py file
                with open('C:\\Users\\admin\\Desktop\\github-repository-scrapper\\github-repo-scraping\\github-repo-scraping\\notebooks-python-no-poo\\' + repository.split('/')[1] + '\\' + file_content.name[:-6] + '.py', 'wb') as f:
                    try:
                        #write all line of cells in the file
                        for cell in json.loads(file_content.decoded_content)['cells']:
                            if cell['cell_type'] == 'code':
                                for line in cell['source']:
                                    f.write(line.encode())
                                f.write('\n'.encode())    
                    except :
                        pass

for repository in list_repositories_no_poo:
    fetch_notebooks(repository)