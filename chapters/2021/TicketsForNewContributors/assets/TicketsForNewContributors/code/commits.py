import requests 
import json 

labels = {}
response = requests.get('https://api.github.com/repos/{owner}/{repo}/commits?author={author}'.format(owner='microsoft', repo='vscode', author='mjbvz'))

links = response.headers['Link']

linksSplitted = links.split(',')
# # print(type (links))
# print(links[0])

indexTopage = linksSplitted[1].find('page')

# print(linksSplitted)

# print(indexTopage)

# print(linksSplitted[1][68])
# n = len(linksSplitted[1])
index = indexTopage + 5
pageNumber = ''
while linksSplitted[1][index].isnumeric():
    pageNumber += linksSplitted[1][index]
    index += 1

# print(pageNumber)

for i in range (0,3):

    convertedPageNumberToInt = int(pageNumber) - i
    convertedPageNumberIntToString = str(convertedPageNumberToInt)
    print(convertedPageNumberIntToString)
    response1 = requests.get('https://api.github.com/repos/{owner}/{repo}/commits?author={author}&page={pageNumber}'.format(owner='microsoft', repo='vscode', author='mjbvz',pageNumber=convertedPageNumberIntToString))

    commits = response1.json()

    for commit in commits :
        commitMessage = commit['commit']['message']
        indexToHastag = commitMessage.find('#')
        if indexToHastag != -1:
            issueNumber = ''
            index1 = indexToHastag 
            while index1 < len(commitMessage) - 1 and commitMessage[index1 + 1].isnumeric():
                issueNumber += commitMessage[index1 + 1]
                index1 += 1
            # print(pageNumber)
            response2 = requests.get('https://api.github.com/repos/{owner}/{repo}/issues/{issueNumber}'.format(owner='microsoft', repo='vscode',issueNumber=issueNumber))
            # if 'labels' in response.json():
            issueLabels = response2.json()['labels'] if 'labels' in response2.json() else []
            # print(issueLabels)
            # numberOfLabels = len(issueLabels)  
            # for i in range (0,numberOfLabels):
            #     if issueLabels[i] is not None:
            #         if 'name' in issueLabels[i]:
            #             print(issueLabels[i]['name'])
            for label in issueLabels:
                if 'name' in label:
                    # print(label['name'])
                    if label['name'] not in labels:
                        labels[label['name']] = 1
                    else:
                        labels[label['name']] += 1

print(labels)

json_object = json.dumps(labels, indent=4)

with open("microsoft-vscode.json","w") as outfile:
    outfile.write(json_object)