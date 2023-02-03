import glob
import os
import re
from typing import List

WORKING_REPOSITORY: str = "./test_repositories/sagan-master"


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def extract_environment_variable(word: str) -> str:
    """
    >>> extract_environment_variable("Bonjour")
    >>> ''
    >>> extract_environment_variable("${MY_VARIABLE_ENV}aaa")
    >>> 'MY_VARIABLE_ENV'
    :param word: the word you want to check
    :return: '' if the word is not an envionment variable, else the filtered environment variable
    """

    temp_word: str = word

    # Remove the prefix in the word
    for letter in temp_word:
        if letter.isupper():
            break
        word = word[1:]

    temp_word = word
    # Remove the suffix in the word
    for letter in temp_word[::-1]:
        if letter.isupper():
            break
        word = word[:-1]

    filtered_regex = flatten(re.findall(r"(^[A-Z0-9_]+)", word))
    if len(filtered_regex) == 0 or len(filtered_regex[0]) == 1:
        return ''

    if len(filtered_regex[0]) == len(word):
        return word


    return ''


def recover_environment_variable_in_a_file(url_file: str) -> List[str]:
    result = []
    with open(url_file) as file:
        for line_number, line in enumerate(file, 1):
            words_in_line = line.split()

            for word in words_in_line:
                env_variable = extract_environment_variable(word)
                if env_variable != '':
                    result.append({"line_number": line_number, "env_variable": env_variable, "line_content": line})

    return result




types = ('yml', 'java', 'js') # the tuple of file types
files = []
for file_type in types:
    files.extend(glob.glob(f'{WORKING_REPOSITORY}/**/*.{file_type}', recursive=True))

files = [f for f in files if os.path.isfile(f)]

result = {}
for file_url in files:
    result[file_url] = recover_environment_variable_in_a_file(file_url)

print(result)
