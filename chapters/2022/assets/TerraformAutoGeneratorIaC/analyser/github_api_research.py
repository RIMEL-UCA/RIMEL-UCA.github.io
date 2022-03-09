import time
import requests
from tqdm import tqdm
import json
import urllib.request
import os
import sys

token = "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"


def save_main_file(repo_name, url, dir):

    response = urllib.request.urlopen(url.replace(
        "https://github.com/", "https://raw.githubusercontent.com/").replace("/blob", ""))
    data = response.read()      # a `bytes` object
    # text = data.decode('utf-8') # a `str`; this step can't be used if data is binary

    repos_infos_file_dir = f"{dir}/{repo_name}"

    if not os.path.exists(repos_infos_file_dir):
        os.makedirs(repos_infos_file_dir)

    with open(f"{repos_infos_file_dir}/main.tf", "wb") as main_tf:
        main_tf.write(data)


def get_projects_with_topic_and_language(topic, language):

    r = requests.get(f"https://api.github.com/search/repositories?q=topic:{topic}+language:{language}&per_page=100",
                     headers={"Authorization": f"token {token}"})
    if r.status_code == 200:
        return r
    elif r.status_code == 403:
        time.sleep(5)
        return get_projects_with_topic_and_language(topic, language)


def get_project_full_names(topic, language):
    r = get_projects_with_topic_and_language(topic, language)
    items = []
    if r.status_code == 200:
        items = r.json()['items']
        repos_name = [result_name["full_name"] for result_name in items]
        return repos_name, items
    return [], items


def get_main_ts_link(repo_name):
    try:
        r = requests.get(f'https://api.github.com/search/code?q=repo:{repo_name}+filename:main.tf',
                         headers={"Authorization": f"token {token}"})

        if r.status_code == 200:
            response = r.json()

            try:
                if 'html_url' in response['items'][0]:

                    return response['items'][0]['html_url']
            except:
                return None
        elif r.status_code == 403:
            time.sleep(5)
            return get_main_ts_link(repo_name)
    except:
        time.sleep(10)
        return get_main_ts_link(repo_name)


def fetch_main_tf_for(topic, language, dir="", save=False, repos_name=None):
    main_tf_links = {}
    print(f"Getting main.tf from topic {topic} and language {language}")

    items = []

    if not repos_name:
        repos_name, items = get_project_full_names(topic, language)

    for repos_name in tqdm(repos_name):
        link = get_main_ts_link(repos_name)
        if link:
            main_tf_links[repos_name] = link

    if save:

        if not len(dir) > 0:
            dir = f"{topic}_{language}"

        if not os.path.exists(dir):
            print(f"Error {dir} doesn't exist!\nExiting...")
            sys.exit()

        with open(f"{dir}/all.json", "w") as content:
            json.dump(items, content)

    return main_tf_links


def save_main_tf_file(topic, language, save_all_infos=False):

    repos_infos_file_dir = f"Search_Keys_{topic}_html/{topic}_{language}"

    if not os.path.exists(repos_infos_file_dir):
        os.makedirs(repos_infos_file_dir)

    repos_infos_file = f"{repos_infos_file_dir}/main_content.json"

    main_tf_links = None
    repos_name = None

    if os.path.exists(repos_infos_file):
        main_tf_links = {}
        with open(repos_infos_file, "r") as content:
            main_tf_links = json.load(content)
            # main_tf_links.keys()
            try:
                repos_name = get_project_full_names(topic, language)
            except:
                print("Failed to get repos Name")

            if len(main_tf_links) != len(repos_name):
                main_tf_links = None
                #TODO Améliorer le processus en ne récupérant que les repos maquants


    download = False

    if not main_tf_links or len(main_tf_links) == 0:
        print(
            f"Warning the file {repos_infos_file} is empty." if main_tf_links else f"The file {repos_infos_file} is empty.")
        print("Downloading the content ...")

        main_tf_links = fetch_main_tf_for(
            topic, language, repos_infos_file_dir, save_all_infos, repos_name)

        download = main_tf_links != None and len(main_tf_links) > 0

    if not main_tf_links or len(main_tf_links) == 0:
        print(
            f"Error the file {repos_infos_file} is still empty!\nDownload failed!\nExiting...")
        sys.exit()

    if download:
        #TODO Améliorer le processus en ne sauvegardant que les repos manquants
        with open(repos_infos_file,"w") as content:
            json.dump(main_tf_links, content)

    print(f"Downloading maint.tf files matching with topic {topic} and language {language}")
    for repo_name in tqdm(main_tf_links):
        save_main_file(repo_name, main_tf_links[repo_name], "Downloads_terraform_html")


"""r = requests.get(f"https://api.github.com/search/code?q=filename:main.tf",
    headers={"Authorization":f"token {token}"})

print(r.json())"""
