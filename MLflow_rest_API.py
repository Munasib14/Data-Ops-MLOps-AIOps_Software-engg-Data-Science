import json
import requests

url = 'http://localhost:1234/invocations'
headers = {'Content-type':'application/json'}


request_data = json.dumps({"data":[[7.4,0.8,0, 1.9, 0.07, 10, 34, 0.9978, 3.51, 0.56, 9.4]]})
response = requests.post(url, request_data, headers=headers)
print("Predicted Label")
print(response.text)