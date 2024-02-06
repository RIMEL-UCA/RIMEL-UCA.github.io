import json
import glob
import sys
import os
import difflib

def counter_prod_cons(path):
      with open(path, 'r') as f:
        data = json.load(f)
        count = 0
        for entry in data:
            for item in entry:
                if item.get("type") == "consumer" or item.get("type") == "producer":
                    count += 1
        return count

def counter_ser_top(path):
    with open(path, 'r') as f:
        data = json.load(f)
        count = len(data)
        return count
    
def calculate_topics_diversity(topics_number,services_number):
    return topics_number/services_number

def table_of_services_creator(path):
    with open(path, 'r') as fichier:
        table_services = json.load(fichier)
        return table_services

def count_services_comm_w_bus(paths, path_services):
    table_of_services = table_of_services_creator(path_services)
    map_services_uses_buses = {}
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            for entry in data:
                for item in entry:
                    if item.get("type") == "consumer" or item.get("type") == "producer":
                        service = item.get("service")
                        if service is not None:
                            if service in table_of_services:
                                if service not in map_services_uses_buses:
                                    map_services_uses_buses[service] = {
                                        "producers": 0,
                                        "consumers": 0
                                    }
                                if item.get("type") == "consumer":
                                    map_services_uses_buses[service]["consumers"] += 1
                                elif item.get("type") == "producer":
                                    map_services_uses_buses[service]["producers"] += 1
                            else:
                                for table_service in table_of_services:
                                    for word in service.split('-'):
                                        if difflib.SequenceMatcher(None, word, table_service).ratio() > 0.9:
                                            if service not in map_services_uses_buses:
                                                map_services_uses_buses[service] = {
                                                    "producers": 0,
                                                    "consumers": 0
                                                }
                                            if item.get("type") == "consumer":
                                                map_services_uses_buses[service]["consumers"] += 1
                                            elif item.get("type") == "producer":
                                                map_services_uses_buses[service]["producers"] += 1
                                            break
    return map_services_uses_buses

def create_metrics(path_producers, path_consumers, path_services, path_topics):   
    services_number = counter_ser_top(path_services)
    topics_number = counter_ser_top(path_topics)
    producers_number = counter_prod_cons(path_producers)
    consumers_number = counter_prod_cons(path_consumers)
    topics_diversity=calculate_topics_diversity(topics_number,services_number)
    number_services_comm_w_bus=count_services_comm_w_bus([path_producers,path_consumers],path_services)
    categorization = categorize_project(producers_number, consumers_number, topics_number, topics_diversity)
    
    data = {
    "producers_number": producers_number,
    "consumers_number": consumers_number,
    "services_number": services_number,
    "topics_number": topics_number,
    "topics_diversity": topics_diversity,
    "communication_rate_map": number_services_comm_w_bus,
    "services_names": table_of_services_creator(path_services),
    "topics_names": table_of_services_creator(path_topics),
    "categorization": categorization,
    }
    
    return data
    
def save_metrics(data, path):
    with open(path, 'w') as ofile:
        json.dump(data, ofile, indent=4)
        print(f"Metrics written in {path}")
        
def categorize_project(producers, consumers, topics, topics_diversity):
    categorization = {}

    if producers == 1 and consumers == 1 and topics == 1:
        categorization['difficulty'] = 'Beginner'
    elif producers <= 5 and consumers <= 5 and topics <= 5:
        categorization['difficulty'] = 'Intermediate'
    else:
        categorization['difficulty'] = 'Advanced'

    categorization['producers'] = producers
    categorization['consumers'] = consumers
    categorization['topics'] = topics
    categorization['topics_diversity'] = topics_diversity

    return categorization
    
def main():
    path_producers = glob.glob("outputs/*--search-producers.json")
    path_consumers = glob.glob("outputs/*--search-consumers.json")
    path_services = glob.glob("outputs/*--search-microservices.json")
    path_topics = glob.glob("outputs/*--search-topics.json")
    
    nom_du_projet = sys.argv[1] if len(sys.argv) > 1 else "Nom_du_projet_inconnu"
    
    path_metrics = f'./metrics/{nom_du_projet.split("/")[1]}--metrics.json'
    

    metrics = create_metrics(path_producers[0], path_consumers[0], path_services[0], path_topics[0])
    
    if not os.path.exists('./metrics'):
        os.makedirs('./metrics')
    
    save_metrics(metrics, path_metrics)

if __name__ == "__main__":
    main()