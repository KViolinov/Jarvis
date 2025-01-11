import os
import io
import math
import pygame
import random
import spotipy
import requests
import webbrowser
import speech_recognition as sr
import google.generativeai as genai
from langchain_ollama import OllamaLLM
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth


# Initialize Pygame
pygame.init()
pygame.mixer.init()
client = ElevenLabs(api_key="sk_4895d5832580be20287fa0914ec3a9a7da4756056d21b418")
r = sr.Recognizer()


# Seting up spotify
client_id = 'dacc19ea9cc44decbdcb2959cd6eb74a'
client_secret = '11e970f059dc4265a8fe64aaa80a82bf'
sp = spotipy.Spotify(auth_manager=spotipy.SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri='http://localhost:8888/callback',
    scope='user-read-currently-playing'))  # Scope for currently playing song

# Setting up Gemini
os.environ["GEMINI_API_KEY"] = "AIzaSyBzMQutGJnduWwKcTrmvAvP_QiTj8zaJ3I"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

system_instruction = (
    "You are Jarvis, a helpful and informative AI assistant. "
    "Always respond in a professional and concise manner."
    "Keep answers short but informative"
    "Ensure all responses are factually accurate and easy to understand."
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
font_large = pygame.font.Font(None, 48)
font_small = pygame.font.Font(None, 32)

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
    "Yes sir, what can I assist you with?",
    "I'm here, how can I help?",
    "At your service, sir.",
    "What do you need, sir?",
    "Listening, how can I assist you?",
    "How may I be of help today?",
    "I'm ready, what's your command?",
    "What can I do for you, sir?",
    "Always ready to help, sir.",
    "How can I assist you?"
]

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

    speed = 1
    is_collided = True
    angle += speed

def draw_thinking():
    """Update settings when the model is listening."""
    global target_color_1, target_color_2, is_collided, angle, speed
    target_color_1 = list(ORANGE1)
    target_color_2 = list(ORANGE2)
    speed = 0.75
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

def fetch_current_track():
    """Fetch the current playing track and its album cover."""
    try:
        current_track = sp.currently_playing()
        if current_track and current_track['is_playing']:
            song = current_track['item']['name']
            artist = ", ".join([a['name'] for a in current_track['item']['artists']])
            album_cover_url = current_track['item']['album']['images'][0]['url']
            progress_ms = current_track['progress_ms']  # Progress in milliseconds
            duration_ms = current_track['item']['duration_ms']  # Duration in milliseconds
            return song, artist, album_cover_url, progress_ms, duration_ms
        return None, None, None, 0, 0
    except Exception as e:
        print(f"Error fetching track: {e}")
        return None, None, None, 0, 0

def load_album_cover(url):
    """Download and convert the album cover image to a Pygame surface."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image_data = io.BytesIO(response.content)
            image = pygame.image.load(image_data, "jpg")
            return pygame.transform.scale(image, (300, 300))  # Scale to 300x300
    except Exception as e:
        print(f"Error loading album cover: {e}")
    return None

def draw_progress_bar(surface, x, y, width, height, progress, max_progress):
    """Draw a progress bar to represent the song timeline."""
    # Check if max_progress is non-zero to avoid division by zero
    if max_progress > 0:
        # Calculate the progress ratio
        progress_ratio = progress / max_progress
        progress_width = int(width * progress_ratio)
    else:
        progress_width = 0  # If duration is zero, show no progress

    # Draw the empty progress bar (background)
    pygame.draw.rect(surface, (50, 50, 50), (x, y, width, height))

    # Draw the filled progress bar (foreground)
    pygame.draw.rect(surface, GREEN1, (x, y, progress_width, height))

def record_text():
    """Listen for speech and return the recognized text."""
    try:
        with sr.Microphone() as source:
            print("Listening...")
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)

            # Recognize speech using Google API
            MyText = r.recognize_google(audio, language="en-US")
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

    print("Welcome to Jarvis! Say 'Jarvis' to activate. Say 'exit' to quit.")

    while True:
        if not wake_word_detected:
            # Listen for the wake word
            print("Waiting for wake word...")
            user_input = record_text()

            if user_input and "jarvis" in user_input:
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

    # Fetch current track periodically (e.g., every 3 seconds)
    if pygame.time.get_ticks() % 3000 < 50:  # Update every 3 seconds
        song, artist, album_cover_url, progress_ms, duration_ms = fetch_current_track()
        if song and artist:  # Only update if song and artist are available
            current_song = song
            current_artist = artist
            current_progress = progress_ms
            song_duration = duration_ms

        # Draw the progress bar for the song timeline
        # Adjust the progress bar position if needed
    progress_bar_x = (WIDTH - 700) // 2
    progress_bar_y = HEIGHT - 30  # Adjust y-position for progress bar
    draw_progress_bar(screen, progress_bar_x, progress_bar_y, 700, 10, current_progress, song_duration)

    # Draw song information above the progress bar
    if current_song:
        song_surface = font_small.render(current_song, True, WHITE)
        song_text_x = (WIDTH - song_surface.get_width()) // 2
        song_text_y = progress_bar_y - 20  # Adjust y-position for song name
        screen.blit(song_surface, (song_text_x, song_text_y))

    # Update Display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
