import pyttsx3
import speech_recognition as sr
from langchain_ollama import OllamaLLM

# Initialize the Llama 3 model using the OllamaLLM
model = OllamaLLM(model="llama3")
engine = pyttsx3.init()
r = sr.Recognizer()

# Function to record speech and convert it to text
def record_text():
    while True:
        try:
            with sr.Microphone() as source2:
                print("Listening...")
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)

                # Recognize speech using Google API
                MyText = r.recognize_google(audio2, language="en-US")  # Change language if needed
                print(f"You said: {MyText}")
                return MyText

        except sr.RequestError as e:
            print(f"API Request Error: {e}")
            return "Error: API unavailable"
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")
            engine.say("Sorry, I didn't catch that. Please try again.")
            engine.runAndWait()

# Function to run the chatbot
def chatbot():
    print("Welcome to the Llama 3 chatbot! Say 'exit' to end the conversation.")

    while True:
        # Wait for the user to speak
        user_input = record_text()
        print(user_input)

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            engine.say("Goodbye!")
            engine.runAndWait()
            break

        # Invoke the model with user input
        result = model.invoke(input=user_input)

        # Display the response from Llama 3
        print(f"Llama 3: {result}")
        engine.say(result)
        engine.runAndWait()

# Run the chatbot
chatbot()
