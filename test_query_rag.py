import requests

# Define the API endpoint and API key
url = "http://127.0.0.1:8001/query-rag/"
api_key = "new_secret_key"

# Define the query
query = "What is the purpose of a business degree?"

# Send the POST request
headers = {"x-api-key": api_key, "Content-Type": "application/json"}
data = {"query": query}
response = requests.post(url, headers=headers, json=data)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
