import json
from googleapiclient.discovery import build

# Replace with your own API key and Search Engine ID (cx)
API_KEY = "NOT MY KEY"
SEARCH_ENGINE_ID = "YOU ENGINE ID"

def get_result_count(query):
    # Build the Custom Search API service
    service = build("customsearch", "v1", developerKey=API_KEY)

    # Perform the search
    result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()

    # Extract the total results count
    total_results = result.get("searchInformation", {}).get("totalResults", 0)
    return int(total_results)

# Load companies and tools from JSON files
with open("companies.json", "r") as companies_file:
    companies = json.load(companies_file)

with open("tools.json", "r") as tools_file:
    tools = json.load(tools_file)

# Loop through all combinations of companies and tools
for company in companies:
    for tool in tools:
        query = f"allintitle: {company} {tool}"
        result_count = get_result_count(query)
        print(f"Query: {query} | Number of results: {result_count}")