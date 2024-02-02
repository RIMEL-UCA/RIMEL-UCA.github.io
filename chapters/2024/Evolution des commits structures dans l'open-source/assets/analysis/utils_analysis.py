import re
import os

def choose_project():
    print("\nChoose a project to analyze:")
    projects = os.listdir('results')
    for i, project in enumerate(projects):
        print(f"{i + 1}. {project}")
    
    choice = input("Enter your choice: ")
    return 'results/' + projects[int(choice) - 1]

def is_bot_commit(message):
    pattern = r'Merge.*|Revert.*|Update README.md'

    return bool(re.match(pattern, message))

def is_conventional_comment(message):
    pattern = r'\s?(Conventional Commit!?|CC!?|commit message!?|semantic message!?)\s*'
    return bool(re.search(pattern, message, re.IGNORECASE))

def is_conventional_commit(message):
    """ Checks if a commit message follows the Conventional Commits pattern. """
    return is_fix(message) or is_feat(message) or is_release(message) or is_test(message) or is_clean(message) or is_doc(message) or is_other(message)

def is_feat(message):
    """ Checks if a commit message is a feature. """
    pattern = r'\s?(feat!?|Feat!?|add!?|chore!?|Chore!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_release(message):
    """ Checks if a commit message is a release. """
    pattern = r'\s?(bump!?|release!?|revert!?|Revert!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_fix(message):
    """ Checks if a commit message is a fix. """
    pattern = r'\s?(fix!?|Fix!?|docs/fix!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_test(message):
    """ Checks if a commit message is a test. """
    pattern = r'\s?(test!?|Test!?|tests!?|perf!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_clean(message):
    """ Checks if a commit message is a clean. """
    pattern = r'\s?(refactor!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_doc(message):
    """ Checks if a commit message is a doc. """
    pattern = r'\s?(docs!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))

def is_other(message):
    """ Checks if a commit message is other. """
    pattern = r'\s?(style!?|ci!?|build!?|wip!?|)\s*(\(.*\))?!?\s*:\s*.+|🚧.*|🧹.*|🛠.*'
    return bool(re.match(pattern, message))