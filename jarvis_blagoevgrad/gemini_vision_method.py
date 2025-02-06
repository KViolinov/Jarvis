import cv2
from PIL import Image
import google.generativeai as genai
from google.cloud import vision
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

os.environ["GEMINI_API_KEY"] = "AIzaSyBzMQutGJnduWwKcTrmvAvP_QiTj8zaJ3I"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

system_instruction = (
    "Вие сте Джарвис, полезен и информативен AI асистент."
    "Винаги отговаряйте професионално и кратко, но също се дръж приятелски."
    "Поддържайте отговорите кратки, но информативни."
    "Осигурете, че всички отговори са фактологически точни и лесни за разбиране."
)

chat = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [system_instruction],
        }
    ]
)


def gemini_vision():
    # Open the webcam
    audio = client.generate(text="Камерата по подразбиране ли да използвам?", voice="Brian")
    play(audio)

    print("Listening for camera info...")
    camera_info = record_text()

    if "да" in camera_info:
        cap = cv2.VideoCapture(0)
        audio = client.generate(text="Добре, използвам web камерата на компютъра ви", voice="Brian")
        play(audio)
    elif "не" in camera_info or "другата" in camera_info:
        cap = cv2.VideoCapture(2)
        audio = client.generate(text="Добре, използвам камерата от ви ар хедсета",
                                voice="Brian")
        play(audio)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Create a named window
    cv2.namedWindow("Capture Window", cv2.WINDOW_NORMAL)

    # Create a named window and resize it
    cv2.namedWindow("Capture Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Window", 800, 600)  # Set window size to 800x600

    # Countdown from 3
    for i in range(3, 0, -1):
        # Display the countdown on the OpenCV window
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Add countdown text (centered)
        cv2.putText(frame, str(i), (350, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

        # Show frame
        cv2.imshow("Capture Window", frame)
        cv2.waitKey(1000)  # Wait for 1 second

    # Capture the final image when countdown hits 1
    # pygame.mixer.music.load("camera_shutter.wav")
    # pygame.mixer.music.play()
    ret, frame_bgr = cap.read()
    if not ret:
        print("Error: Failed to capture final image.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # Convert BGR to RGB for Gemini
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    captured_image = Image.fromarray(frame_rgb)

    # Close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Provide a prompt
    prompt = "Опиши какво виждаш на снимката."

    # Send the image to the Gemini Vision model
    response = model.generate_content([prompt, captured_image])

    # Print the AI's response
    print("\nAI Response:")
    print(response.text)
    audio = client.generate(text=response.text, voice="Brian")
    play(audio)
