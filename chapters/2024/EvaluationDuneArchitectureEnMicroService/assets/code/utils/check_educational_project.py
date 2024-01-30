import requests
import sys


def is_educational_project(readme_content):
    keywords = ['course', 'teaching', 'tutorial', 'education', 'lecture', 'pedagogy']
    for word in keywords:
        if word in readme_content.lower():
            return True
    return False


def get_readme_content(repo_url, branch):
    readme_url = f"{repo_url.rstrip('/')}/raw/{branch}/README.md"
    try:
        response = requests.get(readme_url)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_educational_project.py <github_repo_url>")
        sys.exit(1)

    repo_url = sys.argv[1]
    branches = ['main', 'master', 'dev', 'develop']

    for branch in branches:
        readme_content = get_readme_content(repo_url, branch)
        if readme_content:
            if is_educational_project(readme_content):
                print(
                    f"This repository is likely related to a course, teaching, or educational project. (Branch: {branch})")
            else:
                print(
                    f"This repository does not appear to be related to a course or educational project. (Branch: {branch})")
            break
    else:
        print("README file not found in any of the standard branches (main, master, dev, develop).")


if __name__ == "__main__":
    main()
