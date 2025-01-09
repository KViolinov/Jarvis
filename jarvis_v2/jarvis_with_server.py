import requests
from datetime import datetime
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama3")

# Initialize Firebase Admin SDK
firebase_url_input = "https://jarvis-3d931-default-rtdb.firebaseio.com/input.json"
firebase_url_output = "https://jarvis-3d931-default-rtdb.firebaseio.com/output.json"

# Convert the current time to a string
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Get data from Firebase
response = requests.get(firebase_url_input)
data = response.json()
print("Data from Firebase:", data)

# Sort the data based on timestamp (descending order)
sorted_data = sorted(data.items(), key=lambda x: datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S"), reverse=True)

# Get the last record (most recent)
last_record = sorted_data[0]
print("Most recent record:", last_record)

# Invoke the model with the last record's data
result = model.invoke(input=last_record[0])  # Remove max_tokens parameter

# If you want to limit the output length, truncate it manually
max_tokens = 100  # Desired token length (approximated by characters)
result_text = result[:max_tokens]  # Truncate the result text

# Prepare the result data for Firebase
output_data = {last_record[0]: result_text}  # Use the key from the last record and the model's output as a value

# Send data to Firebase
requests.patch(firebase_url_output, json=output_data)
print(f"The model sent {output_data}")
