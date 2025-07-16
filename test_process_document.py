import requests

# Define the API endpoint and API key
url = "http://127.0.0.1:8001/process-document/"
api_key = "new_secret_key"

# Path to the document to upload
file_path = "/home/intern1/files/image2.pdf"  # Replace with the path to your document

# Send the POST request
headers = {"x-api-key": api_key}
files = {"file": open(file_path, "rb")}
response = requests.post(url, headers=headers, files=files)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
