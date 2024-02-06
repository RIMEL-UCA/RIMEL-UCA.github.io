import json
import sys
import re
import os
import uuid

def delete_special_car(str):
    clean_str = re.sub('[^a-zA-Z0-9._-]', '', str)
    return clean_str

def open_json_file(chemin):
    try:
        with open(chemin, 'r') as fichier:
            data = json.load(fichier)
            return data
    except FileNotFoundError:
        print(f"Le fichier {chemin} n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON dans le fichier {chemin}. Vérifiez le format JSON.")
        return None

def analyze_outputs(datas):
    results = []
    pattern = re.compile(r'\([^)]*\)')

    for data in datas:
        for sub_data in data:
            if sub_data['topic'] != 'No topic name found':
                match = pattern.search(sub_data['topic'])
                if not match:
                    cleaned_topic = delete_special_car(sub_data['topic'])
                    results.append(cleaned_topic)
                
    return results

def find_other_topics(pathToProject):
    results = []

    for dir, sub_dir, files in os.walk(pathToProject):
        for file in files:
            if file.endswith(".properties"):
                file_path = os.path.join(dir, file)
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            matches = re.findall(r'kafka\.topic\.[\w.-]+=["\']?([^"\']+)', line)
                            for match in matches:
                                if match:
                                    if match == 'kafka.topic.name':
                                        kafka_topic_name_value = re.search(r'kafka\.topic\.name=["\']?([^"\']+)', line)
                                        if kafka_topic_name_value:
                                            results.append(kafka_topic_name_value.group(1).strip())
                                    else:
                                        results.append(match.strip().replace("kafka.topic.",""))
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier {file_path}: {e}")

    return results

def save_topics(topics_list, topics_output):
    if len(topics_list) == 0:
        print(f"No topics found")
        exit(0)
    with open(topics_output, 'w') as ofile:
        json.dump(topics_list, ofile, indent=4)
        print(f"Topics written in {topics_output}")


if __name__ == '__main__':
    
    run_id = str(uuid.uuid4())

    if len(sys.argv) != 3:
        print("Usage: python script.py chemin/vers/votre/outputs chemin/vers/votre/projet" )
        sys.exit(1)

    path_outputs = sys.argv[1]
    path_project = sys.argv[2]
    project_output_path = f'./outputs/{run_id}--search-topics.json'
    
    results_outputs = set()
    
    for dir, sub_dir, files in os.walk(path_outputs):
        for file in files:
            if "search-microservices" not in file:
                path = os.path.join(path_outputs, file)
                results_outputs.update(analyze_outputs(open_json_file(path)))
        
        
    results_other_topics = find_other_topics(path_project)
    results_topics = list(results_outputs.union(set(results_other_topics)))

    if not os.path.exists('./outputs'):
        print(f"Folder ./outputs does not exist")
        os.makedirs('./outputs')
        print(f"Folder ./outputs created")
        
    save_topics(results_topics, project_output_path)