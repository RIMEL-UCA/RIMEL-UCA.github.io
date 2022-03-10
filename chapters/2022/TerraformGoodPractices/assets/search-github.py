import sys
import math
import datetime
import re

import requests
import argparse
import urllib.parse
from time import sleep


class GitHubException(Exception):
    pass


def get_rate_limit(token):
    # Send the request.
    headers = {"Authorization": f"token {token}"}
    response = requests.get(f"https://api.github.com/rate_limit", headers=headers)

    # Success.
    if 200 <= response.status_code < 300:
        return response.json()

    # Failure.
    raise GitHubException(f"Unable to get the rate limit: {response.status_code}.")


def get_next_reset(token):
    rate = get_rate_limit(token)
    return datetime.datetime.fromtimestamp(rate["resources"]["search"]["reset"])


def get_seconds_until_next_reset(token):
    reset = get_next_reset(token)
    now = datetime.datetime.now()
    return math.ceil(max(0.0, (reset - now).total_seconds()))


def search_code(token, query, page=1, sort="indexed", order="asc"):
    # Parse the inputs.
    query = urllib.parse.quote_plus(query)
    sort = urllib.parse.quote_plus(sort)
    order = urllib.parse.quote_plus(order)

    # Send the request.
    headers = {"Authorization": f"token {token}"}
    response = requests.get(f"https://api.github.com/search/code?q={query}&sort={sort}&order={order}&page={page}", headers=headers)

    # Success.
    if 200 <= response.status_code < 300:
        return response.json()

    # Failure.
    raise GitHubException(f"Unable to get the requested code: {response.status_code}.")


def get_repository_url(file_url):
    result = re.search(r"(https://github\.com/[^/]+/[^/]+)", file_url)
    return result.group(1)


def search_terraform_files(token, min_size, max_size, page=1):
    return search_code(token, f"required_providers extension:tf language:hcl size:{min_size}..{max_size}", page)


def count_terraform_files(token, min_size, max_size, size_step):
    # Get the total steps.
    total_steps = math.ceil((max_size - min_size) / size_step)
    print(f"Total steps: {total_steps}.", file=sys.stderr)

    # For each step.
    current_step = 1
    min_range = min_size
    max_range = min_size + size_step - 1
    while min_range < max_size:
        print(f"= Step: {current_step}/{total_steps}, Range: {min_range} - {max_range} B.", file=sys.stderr)

        try:
            # Search Terraform files.
            result = search_terraform_files(token, min_range, max_range)

            # Get the number of files.
            total_files = result["total_count"]
            print(f"Total files: {total_files}", file=sys.stderr)
            print(f"{min_range}, {max_range}, {total_files}")

            # Next step.
            current_step += 1
            min_range += size_step
            max_range += size_step

            # Delay between steps.
            print("Waiting before the next step.", file=sys.stderr)
            sleep(2)
        except GitHubException:
            seconds = get_seconds_until_next_reset(token) + 5
            print(f"Failed, will retry in {seconds} seconds.", file=sys.stderr)
            sleep(seconds)


def get_terraform_repositories(token, min_size, max_size, size_step):
    # Initialize the repositories.
    repositories = {}

    # Get the total steps.
    total_steps = math.ceil((max_size - min_size) / size_step)
    print(f"Total steps: {total_steps}.", file=sys.stderr)

    # For each step.
    current_step = 1
    min_range = min_size
    max_range = min_size + size_step - 1
    while min_range < max_size:
        print(f"= Step: {current_step}/{total_steps}, Range: {min_range} - {max_range} B.", file=sys.stderr)

        try:
            # Initialize the file counters.
            count_files = 0
            total_files = 0

            # For each page.
            current_page = 1
            while True:
                print(f"== Page {current_page}.", file=sys.stderr)

                try:
                    # Search Terraform files.
                    result = search_terraform_files(token, min_range, max_range, current_page)

                    # Get the total number of files.
                    if current_page == 1:
                        total_files = min(result["total_count"], 1000)
                        print(f"Total files: {total_files}", file=sys.stderr)

                    # Get the page size.
                    page_size = len(result["items"])
                    print(f"Page size: {page_size}", file=sys.stderr)

                    # Missing files.
                    if page_size == 0:
                        print("No more files could be found!", file=sys.stderr)
                        print(f"Missing {total_files - count_files} files.", file=sys.stderr)
                        break

                    # For each file.
                    current_file = 0
                    while current_file < page_size and count_files < total_files:
                        print(f"=== File {count_files + 1}/{total_files}.", file=sys.stderr)

                        # Get the repository url.
                        file = result["items"][current_file]
                        file_url = file["html_url"]
                        repo_url = get_repository_url(file_url)

                        # Add the repository url.
                        if repo_url not in repositories:
                            print(repo_url)
                            repositories[repo_url] = True

                        # Next file.
                        count_files += 1
                        current_file += 1

                    # Next page.
                    current_page += 1

                    # Found all files.
                    if count_files >= total_files:
                        break

                    # Delay between pages.
                    else:
                        print("Waiting before the next page.", file=sys.stderr)
                        sleep(2)
                except GitHubException:
                    seconds = get_seconds_until_next_reset(token) + 5
                    print(f"Failed, will retry in {seconds} seconds.", file=sys.stderr)
                    sleep(seconds)

            # Next step.
            current_step += 1
            min_range += size_step
            max_range += size_step

            # Delay between steps.
            print("Waiting before the next step.", file=sys.stderr)
            sleep(2)
        except GitHubException:
            seconds = get_seconds_until_next_reset(token) + 5
            print(f"Failed, will retry in {seconds} seconds.", file=sys.stderr)
            sleep(seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="the execution mode of the script", required=True)
    parser.add_argument("-t", "--token", type=str, help="the access token to the GitHub API", required=True)
    parser.add_argument("-a", "--min", type=int, help="the minimum size of the Terraform files to search", required=True)
    parser.add_argument("-b", "--max", type=int, help="the maximum size of the Terraform files to search", required=True)
    parser.add_argument("-s", "--step", type=int, help="the step to be used to generate the size intervals", required=True)
    args = parser.parse_args()

    if args.mode == "count":
        count_terraform_files(args.token, args.min, args.max, args.step)
    elif args.mode == "get":
        get_terraform_repositories(args.token, args.min, args.max, args.step)
    else:
        print(f"An invalid mode has been provided.", file=sys.stderr)
        print(f"Please use one of the following modes: 'count', 'get'.", file=sys.stderr)


main()
