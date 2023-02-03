from typing import List

from git import Repo
import re
from tqdm import tqdm
import json
from glob import glob
import os

def find_all_properties_files(url_repository: str) -> List:
    return glob(url_repository + "/**/*.properties", recursive = True)

def find_env_variable_in_properties_files(url_repository: str) -> List:
    property_names = []

    for file_path in find_all_properties_files(url_repository):
        file_properties = {"file": file_path }
        file_properties["env_variables"] = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    file_properties["env_variables"].append({
                        "injected_name": line.split('=')[0],
                        "value": line.split('=')[1]
                    })
        property_names.append(file_properties)
    return property_names

def find_files_with_string(root_dir, string_to_search):
    java_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".java") or file.endswith(".properties"):
                java_file = os.path.join(subdir, file)
                with open(java_file, "r") as f:
                    file_contents = f.readlines()
                    for line in file_contents:
                        for word_to_search in string_to_search:
                            if word_to_search in line:
                                java_files.append({"file": java_file, "word": word_to_search, "line": line})
    return java_files

print("Environment variables in the .properties files")
URL_FILE = './spring-boot-admin'
properties_env = find_env_variable_in_properties_files(URL_FILE)

with open("properties_env.json", "w") as outfile:
    json.dump(properties_env, outfile)

print("\nFound environment variables in code")
root_dir = "./spring-boot-admin"
string_to_search = ["System.getenv", "@Value(", "@Autowired"]
java_files = find_files_with_string(root_dir, string_to_search)

with open("env_variables_in_code.json", "w") as outfile:
    json.dump(java_files, outfile)
