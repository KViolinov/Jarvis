import os
import math
import pygame
import random
import spotipy
import webbrowser
import speech_recognition as sr
import google.generativeai as genai
from Demos.OpenEncryptedFileRaw import dst_fname
from langchain_ollama import OllamaLLM
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from spotipy.oauth2 import SpotifyClientCredentials
import win32com.client as win32
from datetime import datetime, timedelta
import dateparser


# Initialize Pygame
pygame.init()
pygame.mixer.init()
client = ElevenLabs(api_key="sk_b08176d1b7fcb93fcbac72a2b3e57a97be11f8754bff8379")
r = sr.Recognizer()

# Seting up spotify
client_id = 'dacc19ea9cc44decbdcb2959cd6eb74a'
client_secret = '11e970f059dc4265a8fe64aaa80a82bf'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Setting up Gemini
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

# Screen Dimensions
# info = pygame.display.Info()
# WIDTH, HEIGHT = info.current_w, info.current_h
# screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jarvis Interface")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 128, 255)
CYAN = (0, 255, 255)
ORANGE1 = (255, 165, 0)
ORANGE2 = (255, 115, 0)
GREEN1 = (0, 219, 0)
GREEN2 = (4, 201, 4)
PINK1 = (255, 182, 193)  # Light Pink
PINK2 = (255, 105, 180)  # Hot Pink
PURPLE1 = (157, 0, 255)
PURPLE2 = (164, 34, 245)

# Fonts
font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
font_small = pygame.font.Font(pygame.font.get_default_font(), 20)

# Clock
clock = pygame.time.Clock()

# Rotating Circle Parameters
center = (WIDTH // 2, HEIGHT // 2)
max_radius = min(WIDTH, HEIGHT) // 3
angle = 0
speed = 1

# Particle Parameters
particles = []
num_particles = 100

# Pulse effect variables
pulse_factor = 1
pulse_speed = 0.05
min_size = 3
max_size = 3

# Color Transition
current_color_1 = list(BLUE)
current_color_2 = list(CYAN)
target_color_1 = list(BLUE)
target_color_2 = list(CYAN)
color_transition_speed = 10

jarvis_responses = [
    "Тук съм, как мога да помогна?",
    "Слушам, как мога да Ви асистирам?",
    "Как мога да Ви помогна днес?",
    "Как мога да Ви помогна?",
    "Да?",
    "Слушам?"
]

selected_songs = [
    "Another one bites the dust - Queen",
    "Back in black",
    "Shoot to Thrill",
    "Thunderstruck",
    "You Give Love a Bad Name",
    "Highway to Hell - AC/DC",
    "September - Earth, Wind & Fire",
    "Should I Stay or Should I Go - Remastered",
    "If You Want Blood(You've Got It) - AC/DC",
    "Welcome Tо The Jungle - Guns N' Roses"
]

status_list = []

jarvis_voice = "Brian" #deffault voice
#current_model = "Jarvis"

# Ball initial random positions
random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]

# State Variables
model_answering = False
is_collided = False
is_generating = False
wake_word_detected = False

running = True
current_song = ""
current_artist = ""
album_cover = None
current_progress = 0
song_duration = 0

def send_email(subject, body, to_email):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = subject
    mail.Body = body
    mail.To = to_email
    mail.Send()

def parse_natural_time(natural_time):
    """
    Parses a natural language time expression (e.g., '3 часа следобяд днес', 'tomorrow', 'next Wednesday')
    into a datetime object.
    """

    # Manually handle 'днес' and 'утре' since dateparser fails sometimes
    now = datetime.now()

    # Replace Bulgarian words with English for better parsing
    normalized_time = (
        natural_time.replace("днес", "today")
        .replace("утре", "tomorrow")
        .replace("следобяд", "PM")
        .replace("сутринта", "AM")
    )

    # Try parsing with dateparser
    event_time = dateparser.parse(
        normalized_time,
        languages=['bg', 'en'],  # Use both Bulgarian and English
        settings={'PREFER_DATES_FROM': 'future'}
    )

    # If dateparser fails, manually handle simple cases
    if event_time is None:
        if "днес" in natural_time:
            event_time = now.replace(hour=15, minute=0, second=0, microsecond=0)
        elif "утре" in natural_time:
            event_time = (now + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Could not parse the given time expression: {natural_time}")

    return event_time

def create_outlook_appointment(subject, start_time, duration):
    outlook = win32.Dispatch("Outlook.Application")
    appointment = outlook.CreateItem(1)  # 1 = olAppointmentItem

    appointment.Subject = subject
    appointment.Start = start_time
    appointment.Duration = duration
    appointment.ReminderMinutesBeforeStart = 15
    appointment.Save()

    print(f"✅ Appointment '{subject}' scheduled for {start_time}")

def blend_color(current, target, speed):
    """Gradually transitions the current color toward the target color."""
    for i in range(3):
        diff = target[i] - current[i]
        if abs(diff) > speed:
            current[i] += speed if diff > 0 else -speed
        else:
            current[i] = target[i]

def draw_particles(surface, particles, target_mode=False):
    """Draws particles on the surface. If target_mode is True, arrange them in a circle and pulse."""
    global angle, pulse_factor

    for i, particle in enumerate(particles):
        if target_mode:
            # Calculate target circular positions
            target_x = center[0] + math.cos(math.radians(angle + i * 360 / len(particles))) * max_radius
            target_y = center[1] + math.sin(math.radians(angle + i * 360 / len(particles))) * max_radius
            # Smoothly move particles towards their circular positions
            particle["x"] += (target_x - particle["x"]) * 0.05
            particle["y"] += (target_y - particle["y"]) * 0.05

            # Pulse effect
            pulse_factor = min(max_size, pulse_factor + pulse_speed) if pulse_factor < max_size else max(min_size, pulse_factor - pulse_speed)
        else:
            # Move particles randomly when in default mode
            particle["x"] += particle["dx"]
            particle["y"] += particle["dy"]

            # Keep particles within the screen bounds
            if particle["x"] <= 0 or particle["x"] >= WIDTH:
                particle["dx"] *= -1
            if particle["y"] <= 0 or particle["y"] >= HEIGHT:
                particle["dy"] *= -1

        # Draw the particle
        pygame.draw.circle(surface, tuple(current_color_2), (int(particle["x"]), int(particle["y"])), int(pulse_factor))

def draw_response(model):
    """Update settings when the model is answering."""
    global target_color_1, target_color_2, is_collided, angle, speed

    if model == "Jarvis":
        target_color_1 = list(GREEN1)
        target_color_2 = list(GREEN2)
    elif model == "Friday":
        target_color_1 = list(PINK1)
        target_color_2 = list(PINK2)
    elif model == "Veronica":
        target_color_1 = list(PURPLE1)
        target_color_2 = list(PURPLE2)
    speed = 1
    is_collided = True
    angle += speed

def draw_thinking():
    """Update settings when the model is listening."""
    global target_color_1, target_color_2, is_collided, angle, speed
    target_color_1 = list(ORANGE1)
    target_color_2 = list(ORANGE2)
    speed = 1.5
    is_collided = True
    angle += speed

def draw_default():
    """Update settings when the model is not answering."""
    global target_color_1, target_color_2, is_collided, speed
    target_color_1 = list(BLUE)
    target_color_2 = list(CYAN)
    speed = 1
    is_collided = False

def draw_text(surface, text, position, font, color):
    """Draws text onto the surface."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

def update_status(new_status):
    # Add new status to the list
    status_list.append(new_status)

    # Ensure the list only has 5 elements
    if len(status_list) > 5:
        status_list.pop(0)  # Remove the oldest status (first element)

def record_text():
    """Listen for speech and return the recognized text."""
    try:
        with sr.Microphone() as source:
            print("Listening...")
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

def chatbot():
    """Main chatbot loop."""
    global wake_word_detected, model_answering, is_generating, current_model

    current_model= "Jarvis"
    # wake_word_detected = True
    # model_answering = True
    # is_generating = False
    # audio = client.generate(text="Здравейте, аз съм Джарвис, езиков модел на Джеминай обучен от Гугъл. "
    #                              "Аз съм тук, за да отговоря на въпросите ви, да помогна със задачи или да водя разговори на всякакви теми. "
    #                              "Ако искате да ме попитате нещо, просто ме повикайте по име.", voice="Brian")
    # play(audio)
    # model_answering = False
    #
    # wake_word_detected = True
    # current_model = "Friday"
    # model_answering = True
    # is_generating = False
    # audio = client.generate(text="Здравейте, аз съм Friday, езиков модел на LLama3. "
    #                              "Тук съм, за да помогна с въпроси и задачи, като използвам последните технологии за машинно обучение. "
    #                               "Ако искате да ме попитате нещо, просто ме повикайте по име.", voice="Matilda")
    # play(audio)
    # model_answering = False
    #
    # wake_word_detected = True
    # current_model = "Veronica"
    # model_answering = True
    # is_generating = False
    # audio = client.generate(text="Здравейте, аз съм Вероника, езиков модел на DeepSeek. "
    #                              "Моето предназначение е да ви предоставям точна информация и да бъда на разположение за всякакви въпроси или задачи, "
    #                              "с които се нуждаете от помощ. Ако искате да ме попитате нещо, просто ме повикайте по име.", voice="Sarah")
    # play(audio)
    # model_answering = False

    print("Welcome to Jarvis! Say 'Jarvis' to activate. Say 'exit' to quit.")

    while True:
        if not wake_word_detected:
            # Listen for the wake word
            print("Waiting for wake word...")
            user_input = record_text()

            if user_input and ("джарвис" in user_input or "джарви" in user_input):
                wake_word_detected = True
                current_model = "Jarvis"
                pygame.mixer.music.load("beep.flac")
                pygame.mixer.music.play()

                print("Wake word detected!")
                model_answering = True
                is_generating = False

                jarvis_voice = "Brian"
                response = random.choice(jarvis_responses)
                audio = client.generate(text=response, voice=jarvis_voice)
                play(audio)

                model_answering = False
                is_generating = True

            elif user_input and "friday" in user_input:
                wake_word_detected = True
                current_model = "Friday"
                pygame.mixer.music.load("beep.flac")
                pygame.mixer.music.play()

                print("Wake word detected!")
                model_answering = True
                is_generating = False

                jarvis_voice = "Matilda"
                response = random.choice(jarvis_responses)
                audio = client.generate(text=response, voice=jarvis_voice)
                play(audio)

                model_answering = False
                is_generating = True

            elif user_input == "exit":
                print("Goodbye!")
                jarvis_voice = "Brian"
                audio = client.generate(text="Goodbye!", voice=jarvis_voice)
                play(audio)
                break

        else:
            # Actively listen for commands
            print("Listening for commands...")
            user_input = record_text()

            if user_input is None:
                print("Error: No input detected.")
                continue

            if "пусни" in user_input and ("песен" in user_input or "музика" in user_input):
                track_name = random.choice(selected_songs)
                result = sp.search(q=track_name, limit=1)

                # Get the song's URI
                track_uri = result['tracks']['items'][0]['uri']
                print(f"Playing track: {track_name}")

                # Get the current device
                devices = sp.devices()
                # Find the LAPTOP_KOSI device by its ID
                pc_device_id = '7993e31456b6d73672f9c7bcee055fb10ae52f23'

                audio = client.generate(text="Пускам пет едно системата", voice=jarvis_voice)
                play(audio)

                update_status(f"Played {track_name}")

                # Start playback on the LAPTOP_KOSI device
                sp.start_playback(device_id=pc_device_id, uris=[track_uri])
                print("Playback started on LAPTOP_KOSI.")
                model_answering = False
                is_generating = False
                wake_word_detected = False
                continue

            if "пратиш" in user_input and ("имейл" in user_input or "писмо" in user_input):
                audio = client.generate(text="Разбира се, към кого бихте желали да пратите имейла?", voice=jarvis_voice)
                play(audio)

                print("Listening for email info...")
                user_input = record_text()

                if "тати" in user_input or "баща ми" in user_input:
                    to_email = "bojidarbojinov@outlook.com"
                elif "мама" in user_input or "майка ми" in user_input:
                    to_email = "kameliqbojinova@outlook.com"

                audio = client.generate(text="Каква ще е темата на вашето писмо?", voice=jarvis_voice)
                play(audio)

                print("Listening for email info...")
                subject = record_text()

                audio = client.generate(text="Какво искате да изпратите?", voice=jarvis_voice)
                play(audio)

                print("Listening for email info...")
                body = record_text()

                audio = client.generate(text="Супер, преди да изпратя имейла, ще ви кажа какво съм си записал", voice=jarvis_voice)
                play(audio)

                if to_email == "bojidarbojinov@outlook.com":
                    audio = client.generate(text="Имейла е към Божидар Божинов (баща ви)", voice=jarvis_voice)
                    play(audio)
                elif to_email == "kameliqbojinova@outlook.com":
                    audio = client.generate(text="Имейла е към Камелия Божинова (майка ви)", voice=jarvis_voice)
                    play(audio)
                audio = client.generate(text="Темата на писмото е " + subject + "И съдържанието на писмото е " + body, voice=jarvis_voice)
                play(audio)

                audio = client.generate(text="Всичко наред ли е с информацията в писмото?", voice=jarvis_voice)
                play(audio)

                print("Listening for approval...")
                user_input = record_text()

                if user_input == "да":
                    audio = client.generate(text="Супер, пращам имейла", voice=jarvis_voice)
                    play(audio)
                    send_email(subject, body, to_email)
                    update_status(f"Sent an email to {to_email}")

                elif user_input == "не":
                    audio = client.generate(text="Сорка", voice=jarvis_voice)
                    play(audio)

                model_answering = False
                is_generating = False
                wake_word_detected = False
                continue

            if "прочетеш" in user_input and ("писма" in user_input or "имейли" in user_input or "пис" in user_input):
                # Initialize Outlook
                outlook = win32.Dispatch("Outlook.Application").GetNamespace("MAPI")
                inbox = outlook.GetDefaultFolder(6)  # 6 = Inbox

                # Get all messages sorted by received time (newest first)
                messages = inbox.Items
                messages.Sort("[ReceivedTime]", True)  # Sort descending (newest first)

                # Retrieve the last 5 emails
                num_emails = 3  # Change this number if you need more
                latest_messages = [messages.GetNext() for _ in range(num_emails)]

                audio = client.generate(text="Ето последните 3 имейла в пощата ви: ", voice=jarvis_voice)
                play(audio)
                # Print email details
                for i, email in enumerate(latest_messages, start=1):
                    print(f"\n📧 Email {i}:")
                    print(f"Subject: {email.Subject}")
                    print(f"From: {email.SenderName}")
                    print(f"Received: {email.ReceivedTime}")
                    print("\n--- Email Body ---\n")
                    print(email.Body)  # Full email body
                    print("\n--- End of Email ---\n")
                    audio = client.generate(text=f"Имейл номер {i}, изпратено е от {email.SenderName}, "
                                                 f"темата е {email.Subject}, а съдържанието на писмото е {email.Body}", voice=jarvis_voice)
                    play(audio)
                
                update_status(f"Read last 3 emails")
                model_answering = False
                is_generating = False
                wake_word_detected = False
                continue

            if (("събитие" in user_input or "събити" in user_input or "събития" in user_input)
                    and ("създадеш" in user_input or "Създадеш" in user_input or "създай" in user_input or "Създай" in user_input)):
                # subject of event
                audio = client.generate(text="Разбира се, как искате да се казва събитието?", voice=jarvis_voice)
                play(audio)

                print("Listening for apointment info...")
                subject = record_text()

                # time of event
                audio = client.generate(text="За кога да бъде това събитие?", voice=jarvis_voice)
                play(audio)

                print("Listening for apointment info...")
                user_input = record_text()

                # duration of event
                audio = client.generate(text="Колко време ще продължи това събитие?", voice=jarvis_voice)
                play(audio)

                print("Listening for apointment info...")
                duration = record_text()

                try:
                    event_time = parse_natural_time(user_input)
                    print(f"Parsed event time: {event_time}")  # Debug output
                    audio = client.generate(
                        text=f"Супер, запазвам събитие {subject}, в {event_time.strftime('%H:%M %d-%m-%Y')}, и ще трае 1 час",
                        voice=jarvis_voice)
                    play(audio)
                    create_outlook_appointment(subject, event_time, duration = 60)
                    update_status(f"Made an event")
                    model_answering = False
                    is_generating = False
                    wake_word_detected = False
                    continue
                except ValueError as e:
                    print(f"❌ Error: {e}")

                # Направи ми събитие за 3 следобяд днес, което да продължи 1 час, и да се казва "нахрани котката"

            if user_input:
                # Start thinking state
                is_generating = True

                if (current_model == "Jarvis"): #Jarvis model (Gemini)
                    result = chat.send_message({"parts": [user_input]})
                elif (current_model == "Friday"): #Friday model (Llama3)
                    model = OllamaLLM(model="llama3")
                    result = model.invoke(input=user_input)

                # Done generating the answer
                is_generating = False
                model_answering = True

                # Answering based on model
                if (current_model == "Jarvis"): #Jarvis answering
                    print(f"Jarvis: {result.text}")
                    audio = client.generate(text=result.text, voice=jarvis_voice)
                    play(audio)
                elif (current_model == "Friday"): #Friday answering
                    print(f"FRIDAY: {result}")
                    audio = client.generate(text=result, voice=jarvis_voice)
                    play(audio)
                model_answering = False

            # Reset wake word detection after command
            wake_word_detected = False

# Main Loop
running = True
chatbot_thread = None

# Run chatbot in a separate thread
import threading
chatbot_thread = threading.Thread(target=chatbot)
chatbot_thread.start()

while running:
    screen.fill(BLACK)

    # Event Handling
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Toggle behavior based on whether the model is generating or answering
    if is_generating:
        draw_thinking()  # Show thinking state
    elif model_answering:
        draw_response(current_model) # Show answering state
    else:
        draw_default()  # Default state when nothing is happening

    # Smooth Color Transition
    blend_color(current_color_1, target_color_1, color_transition_speed)
    blend_color(current_color_2, target_color_2, color_transition_speed)

    # Draw Particles
    draw_particles(screen, random_particles, target_mode=is_collided)

    # Draw Text
    draw_text(screen, "Jarvis Interface", (10, 10), font_large, WHITE)
    draw_text(screen, "System Status: All Systems Online", (10, 60), font_small, tuple(current_color_2))

    # Update Display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
