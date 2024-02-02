import requests
import re
from datetime import datetime
from collections import defaultdict
import csv
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
github_token = os.getenv('GITHUB_TOKEN')  # Get GitHub token

# Replace 'YOUR_GITHUB_TOKEN' with your personal GitHub token
headers = {
    'Authorization': 'token ' + github_token,
    'Accept': 'application/vnd.github.v3+json',
}


def is_conventional_commit(message):
    """ Checks if a commit message follows the Conventional Commits pattern. """
    pattern = r'^(feat|fix|docs|style|refactor|perf|test|chore|Fix|Feat|Test|revert|Revert|Chore|add|docs/fix)\s*(\(.*\))?\s*:\s*.+|Merge.*|Revert.*|Update.*'
    return bool(re.match(pattern, message))

def get_commits(repo, page=1, per_page=100):
    """ Fetches commits from a specific repository with pagination. """
    commits_url = f"https://api.github.com/repos/{repo}/commits?per_page={per_page}&page={page}"
    response = requests.get(commits_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch commits")
        return []
    
def analyze_commits(commits):
    author_stats = defaultdict(lambda: {'total': 0, 'conventional': 0})
    time_stats = defaultdict(lambda: {'total': 0, 'conventional': 0})

    for commit in commits:
        message = commit['commit']['message']
        date_str = commit['commit']['committer']['date']
        date_str = date_str.rstrip('Z')  # Remove the 'Z' at the end
        date = datetime.fromisoformat(date_str)
        month_key = f"{date.year}-{date.month}"
        year_key = date.year

        author = commit['commit']['committer']['name']
        is_conv = is_conventional_commit(message)

        # Update author stats
        author_stats[author]['total'] += 1
        if is_conv:
            author_stats[author]['conventional'] += 1

        # Update time stats
        time_stats[month_key]['total'] += 1
        time_stats[year_key]['total'] += 1
        if is_conv:
            time_stats[month_key]['conventional'] += 1
            time_stats[year_key]['conventional'] += 1

    return author_stats, time_stats


def analyze_commits_per_author(commits):
    author_period_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'conventional': 0}))

    for commit in commits:
        message = commit['commit']['message']
        date_str = commit['commit']['committer']['date'].rstrip('Z')  # Remove the 'Z' at the end
        date = datetime.fromisoformat(date_str)
        period = f"{date.year}-{str(date.month).zfill(2)}"
        author = commit['commit']['committer']['name']
        is_conv = is_conventional_commit(message)

        # Update stats for each author per period
        author_period_stats[author][period]['total'] += 1
        if is_conv:
            author_period_stats[author][period]['conventional'] += 1

    return author_period_stats

# Function to save data to CSV
def save_to_csv(data, filename, header):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for key, value in data.items():
            writer.writerow([key, value])

# Function to save author period data to CSV
def save_author_period_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Period', 'Conventional Commit Percentage'])
        for author, periods in data.items():
            for period, stats in periods.items():
                conv_percent = (stats['conventional'] / stats['total']) * 100 if stats['total'] > 0 else 0
                writer.writerow([author, period, conv_percent])



def write_commits_to_file(commits, filename):
    with open(filename, 'w') as f:
        for commit in commits:
            author_info = commit['commit']['author']
            message = commit['commit']['message']
            f.write(f"Author: {author_info['name']}, Date: {author_info['date']}\n")
            f.write(f"Message: {message}\n")
            f.write("\n")


def count_bot_and_human_commits(file_path):
    total_commits = 0
    bot_commits = 0

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Author:'):
                total_commits += 1
                if 'bot' in line.lower():
                    bot_commits += 1

    human_commits = total_commits - bot_commits
    bot_commit_percentage = (bot_commits / total_commits) * 100 if total_commits > 0 else 0
    human_commit_percentage = (human_commits / total_commits) * 100 if total_commits > 0 else 0

    return bot_commits, human_commits, bot_commit_percentage, human_commit_percentage





def main():
    repo = input("Enter GitHub repository (format: user/repo)(default: freeCodeCamp/freeCodeCamp): ")
    if repo == '':
        repo = 'freeCodeCamp/freeCodeCamp'
    page = 1
    all_commits = []
    number_of_conventionnal = 0
    number_of_commits = input("How many commits do you want to fetch? (Press Enter for all commits, 100 min): ")
    if number_of_commits == '':
        number_of_commits = 2**31 - 1
    print("Fetching commits...")
    while True and len(all_commits) < int(number_of_commits):
        commits = get_commits(repo, page=page)
        if not commits:
            break
        all_commits.extend(commits)
        if(len(all_commits) % 1000 == 0):
            print("Commit fetched: ", len(all_commits))
        page += 1

    print(f"Total commits fetched: {len(all_commits)}")

    while True:
        print("\nGitHub Repository Commit Analyzer")
        print("1. Save commits to txt")
        print("2. Save author statistics to CSV")
        print("3. Save time-based statistics to CSV")
        print("4. Save author-period statistics to CSV")
        print("5. What percentage of commits are from bots?")
        print("6. Get percentage of conventionnal commits")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            print("Saving commits to results/all_commits.txt")
            if 'all_commits' in locals():
                write_commits_to_file(all_commits, 'results/all_commits.txt')

        elif choice == '2':
            if 'all_commits' in locals():
                author_stats, _ = analyze_commits(all_commits)
                save_to_csv({author: (stats['conventional'] / stats['total']) * 100 for author, stats in author_stats.items()},
                            'results/author_stats.csv', ['Author', 'Conventional Commit Percentage'])
                print("Author statistics saved to author_stats.csv.")
            else:
                print("No commits fetched. Please fetch commits first.")

        elif choice == '3':
            if 'all_commits' in locals():
                _, time_stats = analyze_commits(all_commits)
                save_to_csv({period: (stats['conventional'] / stats['total']) * 100 for period, stats in time_stats.items()},
                            'results/time_stats.csv', ['Period', 'Conventional Commit Percentage'])
                print("Time-based statistics saved to time_stats.csv.")
            else:
                print("No commits fetched. Please fetch commits first.")

        elif choice == '4':
            if 'all_commits' in locals():
                author_period_stats = analyze_commits_per_author(all_commits)
                save_author_period_to_csv(author_period_stats, 'author_period_stats.csv')
                print("Author-period statistics saved to author_period_stats.csv.")
            else:
                print("No commits fetched. Please fetch commits first.")
        elif choice == '5':
            if 'all_commits' in locals():
                bot_commits, human_commits, bot_commit_percentage, human_commit_percentage = count_bot_and_human_commits('results/all_commits.txt')
                print(f"Total commits: {bot_commits + human_commits}")
                print(f"Bot commits: {bot_commits} ({bot_commit_percentage}%)")
                print(f"Human commits: {human_commits} ({human_commit_percentage}%)")
            else:
                print("No commits fetched. Please fetch commits first.")
        elif choice == '6':
             for commit in all_commits:
                if(is_conventional_commit(commit['commit']['message'])):
                    number_of_conventionnal += 1 

                print(f"Percentage of conventionnal commits: ", (number_of_conventionnal/len(all_commits))*100)

        elif choice == '6':
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
