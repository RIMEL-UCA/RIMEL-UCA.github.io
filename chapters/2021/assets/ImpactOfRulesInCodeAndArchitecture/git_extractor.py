import requests
import sys
import os

def main():

    if (len(sys.argv)<3) :
        raise Exception

    # if os.path.isfile("commit_file_gitlab.txt"):
    #     os.remove("commit_file_gitlab.txt")
    # if os.path.isfile("issue_file_gitlab.txt"):
    #     os.remove("issue_file_gitlab.txt")
    # if os.path.isfile("issue_file_github.txt"):
    #     os.remove("issue_file_github.txt")
    # if os.path.isfile("commit_file_github.txt"):
    #     os.remove("commit_file_github.txt")
    # if os.path.isfile("comment_file_github.txt"):
    #     os.remove("comment_file_github.txt")

    with open(sys.argv[1]) as file:
        if (sys.argv[2]=="gitlab") :
            for line in file:
                # GET https://gitlab.inria.fr/api/v4/projects/:id/repository/commits
                commits = requests.get("https://gitlab.inria.fr/api/v4/projects/"+line+"/repository/commits").json()
                commit_messages = []

                with open("results/git/StopCovid/commit_file_gitlab.txt", 'a') as commit_file:
                    for commit in commits:
                        commit_messages.append(commit["message"])
                        commit_file.write(commit["message"]+"\n")

                issues = requests.get("https://gitlab.inria.fr/api/v4/projects/"+line+"/issues").json()

                issue_dict = {}

                with open("results/git/StopCovid/issue_file_gitlab.txt", 'a') as issue_file:
                    issue_file.write("issue_id,issue_title,issue_description,issue_labels")

                    for issue in issues:
                        # Unauthorized to access to this route
                        # notes = requests.get("https://gitlab.inria.fr/api/v4/projects/" + line + "/issues/"+str(issue["id"])+"/notes").json()
                        # print(notes)
                        issue_dict[issue["id"]] = {"title": issue["title"],
                                                  "description": issue["description"], "labels" : issue["labels"]}

                        issue_file.write(str(issue["id"])+","+("" if issue["title"] is None else issue["title"])
                                         +","+("" if issue["description"] is None else issue["description"])
                                         +","+("" if issue["labels"] is None else str(issue["labels"]))+"\n")



        elif (sys.argv[2]=="github"):
            with open(sys.argv[1]) as file:
                for repo_line in file :
                    repo_line_splitted = repo_line.split(',')
                    owner = repo_line_splitted[0]
                    repo = repo_line_splitted[1].rstrip()
                    url = "https://api.github.com/repos/"+owner+"/"+repo+"/issues"
                    issues = requests.get(url).json()
                    issue_dict = {}

                    with open("results/git/CovidAlert/issue_file_github.txt", 'a') as issue_file:
                        issue_file.write("issue_id,issue_title,issue_body,issue_labels")
                        for issue in issues:
                            print(issue)
                            issue_dict[issue["number"]] = {"title": issue["title"], "body": issue["body"],"labels": issue["labels"]}
                            issue_file.write(str(issue["number"]) + "," + ("" if issue["title"] is None else issue["title"])
                                             + "," + ("" if issue["body"] is None else issue["body"])
                                             + "," + ("" if issue["labels"] is None else str(issue["labels"]))+"\n")

                    commit_messages = []
                    url = "https://api.github.com/repos/"+owner+"/"+repo+"/commits"
                    commits = requests.get(url).json()
                    with open("results/git/CovidAlert/commit_file_github.txt", 'a') as commit_file:
                        for commit in commits:
                            print(commit)
                            commit_messages.append(commit["commit"]["message"])
                            commit_file.write(commit["commit"]["message"]+"\n")

                    comments = requests.get("https://api.github.com/repos/"+owner+"/"+repo+"/comments").json()
                    comment_list = []

                    with open("results/git/CovidAlert/comment_file_github.txt", 'a') as comment_file:
                        for comment in comments :
                            print(comment)
                            comment_list.append((comment["commit_id"], comment["body"]))
                            comment_file.write(str(comment["commit_id"])+","+comment["body"]+"\n")


                    url = "https://api.github.com/repos/"+owner+"/"+repo+"/issues/comments"
                    issue_comments = requests.get(url).json()

                    issue_comment_list = []

                    with open("results/git/CovidAlert/issue_comment_file_github.txt", 'a') as issue_comment_file:
                        for issue_comment in issue_comments:
                            print(issue_comment)
                            issue_comment_list.append((issue_comment["issue_url"], issue_comment["body"]))
                            issue_comment_file.write(issue_comment["issue_url"]+","+issue_comment["body"]+"\n")

if __name__ == '__main__':
    main()