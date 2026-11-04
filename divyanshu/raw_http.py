import json
import os
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
url = "https://generativelanguage.googleapis.com/v1beta/interactions"

headers = {
    "x-goog-api-key": api_key,
    "content-type": "application/json"
}

payload = {
    "model": "gemini-3.8-flash",
    "input": "Explain how AI works in a few words"
}

# encode string data into bytes
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        res_body = json.loads(response.read().decode('utf-8'))
        output_text = res_body.get("interaction", {}).get("outputText")
        print(output_text)
except urllib.error.HTTPError as e:
    print(f'HTTP Error: {e.code} - {e.read().decode('utf-8')}')
