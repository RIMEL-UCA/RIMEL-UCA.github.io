import csv
import datetime

from collections import defaultdict
from .utils_analysis import *

def analyze_comments(comments_path):
    author_stats = defaultdict(lambda: {'total': 0, 'conventional': 0})
    time_stats = defaultdict(lambda: {'total': 0, 'conventional': 0})
    author_period_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'conventional': 0}))

    with open(comments_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i in range(0, len(lines) - 3, 3):  # Chaque comment est séparé par un saut de ligne
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

        # Mise à jour des statistiques
        is_conv = is_conventional_comment(message)
        author_stats[author]['total'] += 1
        author_stats[author]['conventional'] += 1 if is_conv else 0

        time_stats[month_key]['total'] += 1
        time_stats[month_key]['conventional'] += 1 if is_conv else 0

        author_period_stats[author][month_key]['total'] += 1
        author_period_stats[author][month_key]['conventional'] += 1 if is_conv else 0

    author_stats = sorted(author_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    return author_stats, time_stats, author_period_stats

def save_author_statistics(data, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Author", "Total Comments", "Conventional Comments"])
        for author, stats in data:
            writer.writerow([author, stats['total'], stats['conventional']])

def save_time_statistics(data, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Year-Month", "Total Comments", "Conventional Comments"])
        for period, stats in data.items():
            writer.writerow([period, stats['total'], stats['conventional']])

def save_author_period_statistics(data, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Author", "Year-Month", "Total Comments", "Conventional Comments"])
        for author, periods in data.items():
            for period, stats in periods.items():
                writer.writerow([author, period, stats['total'], stats['conventional']])

def perform_analysis():
    main_path = choose_project()
    comments_path = os.path.join(main_path, 'comments', 'all_comments.txt')

    while True:
        print("\nGitHub Repository Comment Analyzer")
        print("1. Save author statistics to CSV")
        print("2. Save time-based statistics to CSV")
        print("3. Save author-period statistics to CSV")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            author_stats, _, _ = analyze_comments(comments_path)
            save_author_statistics(author_stats, os.path.join(main_path, 'author_comments_stats.csv'))
            print("Author statistics saved to author_comments_stats.csv")
        elif choice == '2':
            _, time_stats, _ = analyze_comments(comments_path)
            save_time_statistics(time_stats, os.path.join(main_path, 'time_comments_stats.csv'))
            print("Time statistics saved to time_comments_stats.csv")
        elif choice == '3':
            _, _, author_period_stats = analyze_comments(comments_path)
            save_author_period_statistics(author_period_stats, os.path.join(main_path, 'author_period_comments_stats.csv'))
            print("Author-period statistics saved to author_period_comments_stats.csv")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")