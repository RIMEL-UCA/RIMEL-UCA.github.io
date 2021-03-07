import json
import os.path

def getJsonFileContent(fileName):
    if os.path.isfile("results/{}".format(fileName)):
        with open("results/{}".format(fileName),"r") as outfile:
            result = json.loads(outfile.read())
            result['fileName'] = fileName
            return result

def main():
    files = ['ansible-ansible.json', 'facebook-react-native.json', 'flutter-flutter.json', 'microsoft-vscode.json', 'ohmyzsh-ohmyzsh.json']
    filesContents = []
    for file in files:
        filesContents.append(getJsonFileContent(file))
    # print(filesContents)
    with open("results/all-projects.json", "w") as outfile:
        json_object = json.dumps(filesContents, indent=4)
        outfile.write(json_object)


if __name__ == '__main__':
    main()