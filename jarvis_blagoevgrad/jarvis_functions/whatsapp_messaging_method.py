import pywhatkit as kit
import pyautogui
import time
import pyautogui
import time
import os
import speech_recognition as sr
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import subprocess
from api_keys import ELEVEN_LABS_API, GEMINI_KEY

client = ElevenLabs(api_key=ELEVEN_LABS_API)
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

def whatsapp_send_message():
    audio = client.generate(text="На кого искате да пратя съобшение?", voice="Brian")
    play(audio)

    print("Listening for camera info...")
    contact_info = record_text()

    if "тати" in contact_info or "баща ми" in contact_info:
        audio = client.generate(text="Добре, съобщението ще бъде към баща ви. А какво ще искате да бъде съобщението?",
                                voice="Brian")
        play(audio)

        print("Listening for message info...")
        message_info = record_text()

        #subprocess.run(["powershell", "Start-Process firefox.exe"])
        # Send the message (it types but does not send)
        kit.sendwhatmsg_instantly("+359888503801", message_info)

        # Wait for WhatsApp Web to load and type the message
        time.sleep(2)  # Adjust this if needed

        # Press "Enter" to send the message
        pyautogui.press("enter")

    elif "мама" in contact_info or "майка ми" in contact_info:
        audio = client.generate(text="Добре, съобщението ще бъде към мама. А какво ще искате да бъде съобщението?",
                                voice="Brian")
        play(audio)

        print("Listening for message info...")
        message_info = record_text()

        #subprocess.run(["powershell", "Start-Process firefox.exe"])

        # Send the message (it types but does not send)
        kit.sendwhatmsg_instantly("+359888503801", message_info)

        # Wait for WhatsApp Web to load and type the message
        time.sleep(2)  # Adjust this if needed

        # Press "Enter" to send the message
        pyautogui.press("enter")
