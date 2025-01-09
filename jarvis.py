#working program that uses local model on pc

import os
import math
import pygame
import random
import speech_recognition as sr
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import webbrowser

# Initialize Pygame
pygame.init()
pygame.mixer.init()
client = ElevenLabs(api_key="******")
r = sr.Recognizer()

#Setting up Gemini
os.environ["GEMINI_API_KEY"] = "***********"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

system_instruction = (
    "You are Jarvis, a helpful and informative AI assistant. "
    "Always respond in a professional and concise manner."
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
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
# WIDTH, HEIGHT = 1920, 1080
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
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

# Ball initial random positions
random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]

# State Variables
model_answering = False
is_collided = False
is_generating = False
wake_word_detected = False

def blend_color(current, target, speed):
    """Gradually transitions the current color toward the target color."""
    for i in range(3):
        if current[i] < target[i]:
            current[i] = min(current[i] + speed, target[i])
        elif current[i] > target[i]:
            current[i] = max(current[i] - speed, target[i])

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

def draw_response():
    """Update settings when the model is answering."""
    global target_color_1, target_color_2, is_collided, angle, speed
    target_color_1 = list(GREEN1)
    target_color_2 = list(GREEN2)
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
    global wake_word_detected, model_answering, is_generating

    print("Welcome to Jarvis! Say 'Jarvis' to activate. Say 'exit' to quit.")

    while True:
        if not wake_word_detected:
            # Listen for the wake word
            print("Waiting for wake word...")
            user_input = record_text()

            if user_input and "jarvis" in user_input:
                wake_word_detected = True
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

            elif user_input and "friday" in user_input:
                wake_word_detected = True
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

            # if user_input and "open" in user_input and "spotify" in user_input:
            #     print("Command recognized: Open Spotify!")
            #     webbrowser.open("spotify:track:2qBmtZnPSQouvADmqaHKxk")
            #     # Skip further processing of this input
            #     continue

            if user_input:
                # Start thinking state
                is_generating = True

                # Send input to the model
                result = chat.send_message({"parts": [user_input]})

                # Done generating the answer
                is_generating = False
                model_answering = True

                print(f"Jarvis: {result.text}")
                audio = client.generate(text=result.text, voice=jarvis_voice)
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
        draw_response()  # Show answering state
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
