import pyautogui
import time
import os
import speech_recognition as sr
from elevenlabs import play
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key="sk_a1f900fbd7f869b73954edc03d983b4fbebcfb597118b137")
r = sr.Recognizer()

def record_text():
    """Listen for speech and return the recognized text."""
    try:
        with sr.Microphone() as source:
            #print("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)

            # Recognize speech using Google API
            MyText = r.recognize_google(audio, language="bg-BG")
            print(f"You said: {MyText}")
            return MyText.lower()

    except sr.RequestError as e:
        print(f"API Request Error: {e}")
        return None
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that. Please try again.")
        return None

def call_phone():
    # Press the 'Win' key to open the Start menu
    pyautogui.press('winleft')
    time.sleep(1)

    # Type 'Phone Link' in the Start menu search bar
    pyautogui.write('Phone Link')
    time.sleep(1)

    # Press 'Enter' to launch the app
    pyautogui.press('enter')
    time.sleep(5)  # Wait for the app to open

    # Open the webcam
    audio = client.generate(text="На кого искате да звънна?", voice="Brian")
    play(audio)

    print("Listening for contact info...")
    contact_info = record_text()

    if "тати" in contact_info or "баща ми" in contact_info:
        audio = client.generate(text="Добре, звъня на баща ви", voice="Brian")
        play(audio)

        # Example: Let's assume you are typing the number into the first text field
        pyautogui.write('+359888503801')  # Type the number into the active input field
        pyautogui.press('enter')  # Press Enter to submit if needed

    elif "мама" in contact_info or "майка ми" in contact_info:
        audio = client.generate(text="Добре, звъня на мама",
                                voice="Brian")
        play(audio)

        # Example: Let's assume you are typing the number into the first text field
        pyautogui.write('+359888433144')  # Type the number into the active input field
        pyautogui.press('enter')  # Press Enter to submit if needed


    time.sleep(2)  # Allow time for the app to process

