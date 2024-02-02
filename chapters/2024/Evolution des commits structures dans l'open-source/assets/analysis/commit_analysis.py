import csv
import datetime
from collections import defaultdict
from .utils_analysis import *
# Fonction pour compter les commits effectués par des bots et des humains
def count_bot_and_human_commits(file_path):
    total_commits = 0
    bot_commits = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('Author:'):
                total_commits += 1
                if 'bot' in line.lower():
                    bot_commits += 1

    human_commits = total_commits - bot_commits
    bot_commit_percentage = (bot_commits / total_commits) * 100 if total_commits > 0 else 0
    human_commit_percentage = (human_commits / total_commits) * 100 if total_commits > 0 else 0

    return bot_commits, human_commits, bot_commit_percentage, human_commit_percentage


def preprocess_commit_message(message):
    # Extract the substring before the first occurrence of parentheses or colon
    message_parts = message.split('(')
    message_parts = message_parts[0].split(':')
    message_parts = message_parts[0].split('/')
    return message_parts[0]


ENGLISH_WORDS = {'converted', 'various', 'grammar', 'grammatical', 'resolve', 'run', 'update', 'spelling', 'failing',
                 'understanding', 'tooltip', 'enhance', 'changeset'}


def analyze_commits(file_path):
    author_stats = defaultdict(lambda: {'total': 0, 'feat': 0, 'release': 0, 'fix': 0, 'test': 0, 'clean': 0, 'doc': 0, 'other': 0, 'conventional': 0})
    time_stats = defaultdict(lambda: {'total': 0, 'conventional': 0})
    author_period_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'feat': 0, 'release': 0, 'fix': 0, 'test': 0, 'clean': 0, 'doc': 0, 'other': 0, 'conventional': 0}))
    contributors_by_period = defaultdict(lambda: set())
    non_english_commits_by_period = defaultdict(lambda: {'total': 0, 'non_english': 0})

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines) - 3, 3):  # Chaque commit est séparé par un saut de ligne
        author_line = lines[i].strip()
        message_line = lines[i + 1].strip()

        # Vérifier si la ligne contient les informations attendues
        if not author_line.startswith("Author:"):
            continue

        # Extraction de l'auteur et du message
        author = author_line.split("Author: ")[1].split(", Date")[0].strip()
        if (len(message_line.split("Message: ")) >= 2):
            message = message_line.split("Message: ")[1].strip() if message_line.startswith("Message:") else ""
        else:
            message = ""

        # Extraction de la date
        date_str = author_line.split("Date: ")[1].strip() if "Date: " in author_line else ""

        if date_str:
            try:
                date = datetime.datetime.fromisoformat(date_str.rstrip('Z'))
                period = f"{date.year}-{str(date.month).zfill(2)}"
                month_key = f"{date.year}-{date.month:02d}"
                year_key = date.year
            except ValueError as e:
                print(f"Error parsing date: {e}")
        else:
            print("No date information found in author_line.")

        is_conv = is_conventional_commit(message)
        is_bot = is_bot_commit(message)

        # print(is_bot,is_conv,  message,  )
        if(is_bot):
            continue
        # Mise à jour des statistiques
        add_author_stats(author_stats, author, is_conv, message)
        add_author_period_stats(author_period_stats, author, period, is_conv, message)
        if is_conv:
            time_stats[month_key]['conventional'] += 1
            time_stats[year_key]['conventional'] += 1

        time_stats[month_key]['total'] += 1
        time_stats[year_key]['total'] += 1
        contributors_by_period[period].add(author)

        # DETECTION COMMITS AUTRE LANGUE:
        # message = message_line.split("Message: ")[1].split()[0].strip()
        # message = preprocess_commit_message(message)
        # try:
        #     lang, _ = langid.classify(message)
        #     if message.lower() not in ENGLISH_WORDS and lang != 'en':
        #         non_english_commits_by_period[period]['non_english'] += 1
        # except:
        #     pass
        # non_english_commits_by_period[period]['total'] += 1

    author_stats = sorted(author_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    return author_stats, time_stats, author_period_stats, contributors_by_period, non_english_commits_by_period

def add_author_stats(author_stats, author, is_conv, message):
    author_stats[author]['total'] += 1
    if is_conv:
        author_stats[author]['conventional'] += 1
        author_stats[author]['feat'] += 1 if is_feat(message) else 0
        author_stats[author]['release'] += 1 if is_release(message) else 0
        author_stats[author]['fix'] += 1 if is_fix(message) else 0
        author_stats[author]['test'] += 1 if is_test(message) else 0
        author_stats[author]['clean'] += 1 if is_clean(message) else 0
        author_stats[author]['doc'] += 1 if is_doc(message) else 0
        author_stats[author]['other'] += 1 if is_other(message) else 0

def add_author_period_stats(author_period_stats, author, period, is_conv, message):
    author_period_stats[author][period]['total'] += 1
    if is_conv:
        author_period_stats[author][period]['conventional'] += 1
        author_period_stats[author][period]['feat'] += 1 if is_feat(message) else 0
        author_period_stats[author][period]['release'] += 1 if is_release(message) else 0
        author_period_stats[author][period]['fix'] += 1 if is_fix(message) else 0
        author_period_stats[author][period]['test'] += 1 if is_test(message) else 0
        author_period_stats[author][period]['clean'] += 1 if is_clean(message) else 0
        author_period_stats[author][period]['doc'] += 1 if is_doc(message) else 0
        author_period_stats[author][period]['other'] += 1 if is_other(message) else 0

def save_contributors_by_period_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Period', 'Number of Contributors'])
        for period, contributors in data.items():
            writer.writerow([period, len(contributors)])

def save_non_english_commits_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Period', 'Non-English Percentage'])
        for period, stats in data.items():
            total_commits = stats['total']
            non_english_commits_count = stats['non_english']
            non_english_percentage = (non_english_commits_count / total_commits) * 100 if total_commits > 0 else 0
            writer.writerow([period, non_english_percentage])

def save_author_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Total Commits', 'Feat Commits', 'Release Commits', 'Fix Commits', 'Test Commits', 'Clean Commits', 'Doc Commits', 'Other Commits'])
        for author, stats in data:
            writer.writerow([author, stats['total'], stats['feat'], stats['release'], stats['fix'], stats['test'], stats['clean'], stats['doc'], stats['other']])

def save_time_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Period', 'Conventional Commit', 'All Commit'])
        for period, stats in data.items():
            if re.search(r'\d{4}-\d{2}', str(period)):
                writer.writerow([period, stats['conventional'], stats['total']])

# Function to save author period data to CSV
def save_author_period_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Author', 'Period', 'Total Commits', 'Feat Commits', 'Release Commits', 'Fix Commits', 'Test Commits', 'Clean Commits', 'Doc Commits', 'Other Commits'])
        for author, periods in data.items():
            for period, stats in periods.items():
                writer.writerow([author, period, stats['total'], stats['feat'], stats['release'], stats['fix'], stats['test'], stats['clean'], stats['doc'], stats['other']])

def write_commits_to_file(commits, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for commit in commits:
            author_info = commit['commit']['author']
            message = commit['commit']['message']
            f.write(f"Author: {author_info['name']}, Date: {author_info['date']}\n")
            f.write(f"Message: {message}\n")
            f.write("\n")

# Fonction pour effectuer l'analyse et sauvegarder les résultats
def perform_analysis():
    main_path = choose_project()
    commits_path = os.path.join(main_path, 'all_commits.txt')

    while True:
        print("\nGitHub Repository Commit Analyzer")
        print("1. Save author statistics to CSV")
        print("2. Save time-based statistics to CSV")
        print("3. Save author-period statistics to CSV")
        print("4. Save contributors by period to CSV")
        print("5. Save non-english commits by period to CSV")
        print("6. Save general statistics to txt")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            author_stats, _, _, _, _ = analyze_commits(commits_path)
            save_author_to_csv(author_stats, os.path.join(main_path, 'author_stats.csv'))
            print("Author statistics saved to author_stats.csv.")
        elif choice == '2':
            _, time_stats, _, _, _ = analyze_commits(commits_path)
            save_time_to_csv(time_stats, os.path.join(main_path, 'time_stats.csv'))
            print("Time-based statistics saved to time_stats.csv.")

        elif choice == '3':
            _, _, author_period_stats, _, _ = analyze_commits(commits_path)
            save_author_period_to_csv(author_period_stats, os.path.join(main_path, 'author_period_stats.csv'))
            print("Author-period statistics saved to author_period_stats.csv.")

        elif choice == '4':
            _, _, _, contributors_by_period, _ = analyze_commits(commits_path)
            save_contributors_by_period_to_csv(contributors_by_period, os.path.join(main_path, 'contributors_by_period.csv'))
            print("Contributors by period saved to contributors_by_period.csv.")

        elif choice == '5':
            _, _, _, _, non_english_by_period = analyze_commits(commits_path)
            save_non_english_commits_to_csv(non_english_by_period, os.path.join(main_path, 'non_english.csv'))
            print("Non-English commits by period saved to non_english.csv.")

        elif choice == '6':
            bot_commits, human_commits, bot_commit_percentage, human_commit_percentage = count_bot_and_human_commits(
                commits_path)
            print(f"Total commits: {bot_commits + human_commits}")
            print(f"Bot commits: {bot_commits} ({bot_commit_percentage}%)")
            print(f"Human commits: {human_commits} ({human_commit_percentage}%)")
            conventional_commit = 0
            with open(commits_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            for i in range(0, len(lines), 3):  # Chaque commit est séparé par un saut de ligne
                message_line = lines[i + 1]
                if is_conventional_commit(message_line.replace("Message: ", "")):
                    conventional_commit += 1
                else:
                    print(message_line)

            with open(os.path.join(main_path, 'general_stats.txt'), 'a', encoding='utf-8') as file:
                file.write(f"Total commits: {bot_commits + human_commits}\n")
                file.write(f"Bot commits: {bot_commits} ({bot_commit_percentage}%)\n")
                file.write(f"Human commits: {human_commits} ({human_commit_percentage}%)\n")
                file.write(f"Percentage of conventionnal commits: {(conventional_commit / (len(lines) / 3)) * 100}\n")
            print(f"Percentage of conventionnal commits: ", (conventional_commit / (len(lines) / 3)) * 100,
                  conventional_commit, len(lines))

        elif choice == '7':
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")
