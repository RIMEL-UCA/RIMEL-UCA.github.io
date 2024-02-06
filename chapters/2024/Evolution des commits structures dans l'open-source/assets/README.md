# GitHub Commit Analyzer

## Introduction

The GitHub Commit Analyzer is a Python application for scraping, analyzing, and visualizing data from GitHub commits. It is designed to provide insights into commit patterns, such as the frequency of conventional commits, the proportion of commits from bots versus humans, and commit trends over time.

## Requirements

- Python 3.x
- Required Python packages: requests, dotenv, pandas, matplotlib, seaborn, csv, datetime, re

## Installation

- Clone the repository to your local machine.
- Install the required Python packages:

`pip install -r requirements.txt`

## Features and Implementation

### Data Scraping (/scraping)

- Fetches commit data from a GitHub repository.
- Allows specification of the repository and the number of commits to fetch.
- Processes and saves the commit data for further analysis.

### Data Analysis (/analysis)

- Analyzes the scraped commit data.
- Provides statistics on the conventional commit patterns.
- Differentiates between commits made by bots and humans.
- Analyzes commit data based on author and time.

### Data Visualization (/plotter)

- Visualizes the analyzed data.
- Supports various types of plots, including bar and line plots, to represent commit statistics.

### Main Program (main.py)

- The entry point of the application.
- Presents an interactive menu to choose between scraping, analyzing, and plotting data.

## Running the Program

Start the Program: Run main.py in your Python environment:

`python3 main.py`

Choose an Option:

- Scrape data from a specified GitHub repository.
- Analyze existing commit data.
- Plot the analyzed data.
- Exit the program.

Follow On-screen Instructions: Each option will guide you through further steps, such as entering repository details for scraping, choosing specific analyses, or selecting plot types for visualization.
All the results from these actions are saved in the /results directory.

## Link to studied repositories
https://github.com/tinacms/tina.io
https://github.com/puppeteer/puppeteer
https://github.com/freeCodeCamp/freeCodeCamp
https://github.com/apache/echarts
https://github.com/yargs/yargs
https://github.com/facebook/docusaurus
https://github.com/nocodb/nocodb
https://github.com/makeplane/plane
https://github.com/apache/superset
https://github.com/ApolloAuto/apollo
https://github.com/ecomfe/zrender/
https://github.com/nrwl/nx
https://github.com/electron/electron
https://github.com/denoland/deno
