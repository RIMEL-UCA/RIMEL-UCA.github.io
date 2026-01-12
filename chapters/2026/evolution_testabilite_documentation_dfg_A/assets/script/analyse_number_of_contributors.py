import sys
import requests
import statistics
from typing import List, Dict
import time
import os

def get_all_repos(org_name: str, token: str = None) -> List[Dict]:
    """
    Fetch all repositories from a GitHub organization.
    
    Args:
        org_name: Name of the GitHub organization
        token: GitHub personal access token (optional but recommended for rate limits)
    
    Returns:
        List of repository dictionaries
    """
    repos = []
    page = 1
    headers = {}
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    while True:
        url = f'https://api.github.com/orgs/{org_name}/repos'
        params = {'page': page, 'per_page': 100}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code}")
            print(f"Message: {response.json().get('message', 'Unknown error')}")
            break
        
        page_repos = response.json()
        
        if not page_repos:
            break
        
        repos.extend(page_repos)
        page += 1
        
        # Respect rate limits
        time.sleep(0.5)
    
    return repos


def get_contributor_count(owner: str, repo_name: str, token: str = None) -> int:
    """
    Get the number of contributors for a specific repository.
    
    Args:
        owner: Repository owner (organization name)
        repo_name: Repository name
        token: GitHub personal access token
    
    Returns:
        Number of contributors
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f'https://api.github.com/repos/{owner}/{repo_name}/contributors'
    
    per_page = 100
    params = {'per_page': per_page, 'anon': 'true', 'page': 1}
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 204:  # No contributors
        return 0
    
    if response.status_code != 200:
        print(f"Error fetching contributors for {repo_name}: {response.status_code}")
        return 0
    
    # Check Link header for pagination
    link_header = response.headers.get('Link', '')
    
    if 'rel="last"' in link_header:
        # Extract the last page number
        import re
        last_page_match = re.search(r'page=(\d+)>; rel="last"', link_header)
        if last_page_match:
            last_page = int(last_page_match.group(1))
            
            # Get the last page to count remaining contributors
            params = {'page': last_page, 'per_page': per_page, 'anon': 'true'}
            last_response = requests.get(url, headers=headers, params=params)
            
            if last_response.status_code == 200:
                last_page_count = len(last_response.json())
                total = (last_page - 1) * per_page + last_page_count
                return total
    
    # If no pagination, just count the contributors on the first page
    contributors = response.json()
    return len(contributors)


def analyze_contributors(org_name: str, token: str = None):
    """
    Analyze contributor counts across all repositories in an organization.
    
    Args:
        org_name: Name of the GitHub organization
        token: GitHub personal access token (recommended)
    """
    print(f"Fetching repositories from organization: {org_name}")
    repos = get_all_repos(org_name, token)
    
    if not repos:
        print("No repositories found or error occurred.")
        return
    
    print(f"Found {len(repos)} repositories. Analyzing contributors...\n")
    
    repo_stats = []
    
    for i, repo in enumerate(repos, 1):
        repo_name = repo['name']
        print(f"[{i}/{len(repos)}] Processing: {repo_name}...", end=' ')
        
        contributor_count = get_contributor_count(org_name, repo_name, token)
        repo_stats.append({
            'name': repo_name,
            'contributors': contributor_count,
            'url': repo['html_url']
        })
        
        print(f"{contributor_count} contributors")
        
        time.sleep(0.5)
    
    # Sort by contributor count
    repo_stats.sort(key=lambda x: x['contributors'], reverse=True)
    
    contributor_counts = [r['contributors'] for r in repo_stats]
    
    if not contributor_counts:
        print("No contributor data available.")
        return
    
    avg_contributors = statistics.mean(contributor_counts)
    median_contributors = statistics.median(contributor_counts)
    min_contributors = min(contributor_counts)
    max_contributors = max(contributor_counts)
    
    if len(contributor_counts) > 1:
        stdev_contributors = statistics.stdev(contributor_counts)
    else:
        stdev_contributors = 0
    
    # Print results
    print("\n" + "="*80)
    print(f"CONTRIBUTOR ANALYSIS FOR: {org_name}")
    print("="*80)
    print(f"\nTotal repositories analyzed: {len(repo_stats)}")
    print(f"\nStatistics:")
    print(f"  Average contributors: {avg_contributors:.2f}")
    print(f"  Median contributors: {median_contributors:.2f}")
    print(f"  Minimum contributors: {min_contributors}")
    print(f"  Maximum contributors: {max_contributors}")
    print(f"  Standard deviation: {stdev_contributors:.2f}")
    
    small_threshold = avg_contributors
    large_threshold = avg_contributors + stdev_contributors
    
    print(f"\n" + "-"*80)
    print("SUGGESTED CATEGORIES:")
    print("-"*80)
    print(f"  Small projects: < {small_threshold:.0f} contributors")
    print(f"  Medium projects: {small_threshold:.0f} - {large_threshold:.0f} contributors")
    print(f"  Large projects: > {large_threshold:.0f} contributors")
    
    print(f"\n" + "-"*80)
    print("TOP 10 REPOSITORIES BY CONTRIBUTORS:")
    print("-"*80)
    for repo in repo_stats[:10]:
        print(f"  {repo['contributors']:4d} contributors - {repo['name']}")
    
    print(f"\n" + "-"*80)
    print("BOTTOM 10 REPOSITORIES BY CONTRIBUTORS:")
    print("-"*80)
    for repo in repo_stats[-10:]:
        print(f"  {repo['contributors']:4d} contributors - {repo['name']}")
    
    import csv
    output_file = f"{org_name}_contributors_analysis.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'contributors', 'url'])
        writer.writeheader()
        writer.writerows(repo_stats)
    
    print(f"\n" + "="*80)
    print(f"Detailed results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    if sys.argv.__len__() < 2:
        print("Usage: python analyse_number_of_contributors.py <organization_name> <github_token>")
        sys.exit(1)

    ORGANIZATION_NAME = sys.argv[1]
    GITHUB_TOKEN = sys.argv[2]
    analyze_contributors(ORGANIZATION_NAME, GITHUB_TOKEN)
