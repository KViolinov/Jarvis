#front UI that connects to the sever for the model

import pygame
import math
import random
import requests
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from datetime import datetime

# Initialize Pygame
pygame.init()
client = ElevenLabs(api_key="sk_2baa3247d3920a331b6841bcb9412194e5757366c71a0b36")
firebase_url_input = "https://jarvis-3d931-default-rtdb.firebaseio.com/input.json"
firebase_url_output = "https://jarvis-3d931-default-rtdb.firebaseio.com/output.json"
r = sr.Recognizer()

# Screen Dimensions
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
# WIDTH, HEIGHT = 1920, 1080
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jarvis Interface")

# Convert the current time to a string
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    "Listening, how can I assist you?"
]

# Ball initial random positions
random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]

# State Variables
model_answering = False
is_collided = False
is_generating = False


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
    speed = 1
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
    while True:
        try:
            with sr.Microphone() as source2:
                print("Listening...")
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)

                # Recognize speech using Google API
                MyText = r.recognize_google(audio2, language="en-US")  # Change language if needed
                print(f"You said: {MyText}")

                # Check for "Jarvis" keyword
                if "jarvis" in MyText.lower():
                    model_answering = True
                    print("Jarvis detected!")

                    random_response = random.choice(jarvis_responses)
                    audio = client.generate(text=random_response, voice="Brian")  # Respond with "Yes sir"
                    play(audio)

                    model_answering = False
                    continue
                else:
                    return MyText

        except sr.RequestError as e:
            print(f"API Request Error: {e}")
            return "Error: API unavailable"
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")

def chatbot():
    global model_answering, is_generating
    print("Welcome to Jarvis! Say 'exit' to end the conversation.")

    while True:
        #user_input = record_text()
        user_input = input()
        if user_input.lower() == "exit":
            print("Goodbye!")
            audio = client.generate(text="Goodbye!", voice="Brian")
            play(audio)
            break

        # Start thinking state
        is_generating = True
        model_answering = False  # Set to False initially to prevent immediate answering

        #result = model.invoke(input=user_input)  # Get the model's answer

        response = requests.get(firebase_url_output)
        data = response.json()
        print("Data from Firebase:", data)

        # Sort the data based on timestamp (descending order)
        sorted_data = sorted(data.items(), key=lambda x: datetime.strptime(x[1], "%Y-%m-%d %H:%M:%S"), reverse=True)

        # Get the last record (most recent)
        last_record = sorted_data[0]
        print("Most recent record:", last_record)

        is_generating = False  # Done generating the answer
        model_answering = True  # Now the model is answering


        print(f"Jarvis: {last_record[1]}")
        audio = client.generate(text=last_record[1], voice="Brian")
        play(audio)
        model_answering = False

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
