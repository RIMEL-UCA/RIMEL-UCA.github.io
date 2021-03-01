# docker run --rm -ti -v C:\dev\courses\polytech_s10\retroinge:/work dkarchmervue/moviepy /bin/bash
# pip install requests
# python githubgetrepositories.py

import requests
import json
import os
import zipfile
import re

USER = 'montoyadamien'
API_TOKEN = 'API_KEY' # put your github api key here
GIT_API_URL = 'https://api.github.com/'
NUMBER_REPO_TO_FETCH = 40
repositories_using_dsl = 0
every_files_that_match = []
total_number_repositories = 0

def parse_response(r):
	print("Fetching results from : ", r.url, "\n")
	current_directory = os.getcwd()
	path = os.path.join(current_directory, "repositories")
	try:
		os.makedirs(path)
		print("Directory created successfully")
	except OSError as error:
		print("Directory can not be created / already exists")
	try:
		json_result = json.loads(r.text)
		total_number_repositories = json_result['total_count']
		for repo in json_result['items']:
			print("\tfetching repository")
			print("\t\t", repo['full_name'], " - stars : ", repo['watchers_count'], " - forks : ", repo['forks'])
			url = GIT_API_URL + "repos/" + repo['full_name'] + "/zipball"
			print("\t\tDownloading repo from", url, "...")
			download_repo(url, current_directory + "/repositories/", repo['full_name'].replace('/', '_') + ".zip")
	except ValueError	:
		print('Failed parse json', ValueError)


def download_repo(url, directory, archive_name):
	try: # file already in system so no download
		open(directory + archive_name)
		print("\t\t\tRepository already downloaded", directory + archive_name)
	except IOError:
		r = requests.get(url, allow_redirects=True)
		with open(directory + archive_name, 'wb') as f:
			f.write(r.content)
		extract_zip(directory, archive_name)


def extract_zip(directory, archive_name):
	with zipfile.ZipFile(directory + archive_name, 'r') as zip_ref:
		zip_ref.extractall(directory)


def get_api(url):
	try:
		headers = {"Authorization" : (USER + '/token:' + API_TOKEN).replace('\n', '')}
		r = requests.get(GIT_API_URL + url, headers=headers)
		parse_response(r)
	except ValueError:
		print('Failed to get api request from', ValueError)

def browse_every_file_directory(search_path):
	if not (search_path.endswith("/") or search_path.endswith("\\") ): 
			search_path = search_path + "/"
	for fname in os.listdir(path=search_path):
		if os.path.isdir(search_path + fname):
			get_files_that_contains_dsl(search_path + fname)


def get_files_that_contains_dsl(directory):
	global repositories_using_dsl
	print("\t\t\tGetting files that contains dsl")
	search_str = "@dsl.pipeline"
	file_type = ".py"
	print("Searching files that contains dsl in ->", directory)
	files = get_files_that_contains_dsl_in_directory(directory, search_str, file_type)
	if len(files) > 0:
		repositories_using_dsl += 1
	every_files_that_match.extend(files)
			

def get_files_that_contains_dsl_in_directory(search_path, search_str, file_type):
	files_that_match = []
	if not (search_path.endswith("/") or search_path.endswith("\\") ): 
			search_path = search_path + "/"
	for fname in os.listdir(path=search_path):
		if fname.endswith(file_type):
			print("Searching in .py file", fname)
			fo = open(search_path + fname)
			line = fo.readline()
			line_no = 1
			while line != '' :
				index = line.find(search_str)
				if index != -1:
					files_that_match.append(search_path + fname)
				line = fo.readline()  
				line_no += 1
			fo.close()
		elif os.path.isdir(search_path + fname):
			files_that_match.extend(get_files_that_contains_dsl_in_directory(search_path + fname, search_str, file_type))
	return files_that_match

get_api("search/repositories?q=kubeflow+pipelines&sort=stars&order=desc&page=1&per_page=" + str(NUMBER_REPO_TO_FETCH))

current_directory = os.getcwd()
path = os.path.join(current_directory, "repositories")
browse_every_file_directory(path)
print(every_files_that_match)

print("Number of files using dsl :", len(every_files_that_match), "in", repositories_using_dsl, "/", NUMBER_REPO_TO_FETCH,"repositories")
numberOfDslCondition = 0
examplesOfDslCondition = []
numberOfExitHandler = 0
numberOfRaiseError = 0
numberOfRaiseException = 0
typesOfRaise = {}

for file in every_files_that_match:
	fo = open(file)
	line = fo.readline()
	line_no = 1
	while line != '' :
		index = line.find('with dsl.Condition')
		if index != -1:
			numberOfDslCondition += 1
			newLine = line.replace("\t", "").replace("\n", "")
			examplesOfDslCondition.append(newLine)
		else:
			index = line.find('dsl.ExitHandler')
			if index != -1:
				numberOfExitHandler += 1
			else:
				res = re.match('raise [A-Za-z0-9]*Error', line)
				if res:
					numberOfRaiseError += 1
					lineError = res.group(1)
					if lineError in typesOfRaise:
						typesOfRaise[lineError] += 1
					else:
						typesOfRaise[lineError] = 1
				else:
					res = re.match('raise [A-Za-z0-9]*Exception', line)
					if res:
						numberOfRaiseException += 1
						lineError = res.group(1)
						if lineError in typesOfRaise:
							typesOfRaise[lineError] += 1
						else:
							typesOfRaise[lineError] = 1

		line = fo.readline()
		line_no += 1
	fo.close()

#for file in every_files_that_match:
#	os.system("grep -f '" + file + "' -e 'raise [a-zA-Z0-9]*' [|wc -l]")

print("There are", total_number_repositories, "repositories fetched from the api")
print("Number of dsl.Condition used :", numberOfDslCondition)
print("Examples of dsl.Condition used :", examplesOfDslCondition)
print("Number of exit handler used :", numberOfExitHandler)
print("Number of raise Error used :", numberOfRaiseError)
print("Number of raise Exception used :", numberOfRaiseException)
print("Types of raises with number :", typesOfRaise)
