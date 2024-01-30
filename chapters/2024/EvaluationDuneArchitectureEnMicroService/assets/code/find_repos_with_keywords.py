from github import Github
import csv
import os

# Replace 'YOUR_ACCESS_TOKEN' with your actual GitHub access token
ACCESS_TOKEN = ''
g = Github(ACCESS_TOKEN)

# Keywords related to microservices architecture
architecturePatterns = ["microservices", "service-oriented", "SOA", "event", "consul"]
toolsAndFramework = ["docker", "kubernetes", "istio", "helm", "ci/cd", "jenkins", "travis", "circleci", "argo"]
developmentAndLanguageSpecific = ["spring", "nestjs", "django", "fastapi", "java", "go", "ruby", "python", "quarkus", "vertx", "react", "angular", "vuejs"]
messagingAndCommunication = ["kafka", "rabbitmq", "pub/sub", "grpc", "rest", "restful"]
topics = [architecturePatterns, toolsAndFramework, developmentAndLanguageSpecific, messagingAndCommunication]


for topic in topics: 
    print("------------------------------------------")
    print("Now fetching repositories for the following keywords: " + str(topic))
    print("------------------------------------------")

    for keyword in topic:
        print("Looking for : " + keyword)
        csv_file = keyword + ".csv"
        repo_names = []
        query = f'"{keyword}" in:readme,description,repo'
        repos = g.search_repositories(query)

        for repo in repos:
            repo_names.append(repo.full_name)

        csv_file = csv_file.replace("/", "-")
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Repository Name'])  # Header
            for name in repo_names:
                writer.writerow([name])