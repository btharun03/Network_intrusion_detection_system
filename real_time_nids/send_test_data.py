import requests
import time

url = 'http://localhost:5000/predict'

dummy_data = {
    "sample": [0.1] * 41  
}

while True:
    try:
        
        response = requests.post(url, json=dummy_data)
        
        print(response.json())
    except Exception as e:
        print(f"An error occurred: {e}")
    
    time.sleep(2)
