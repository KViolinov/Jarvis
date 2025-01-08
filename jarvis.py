#currently working verison. Needs improvement in future

import pygame
import math
import random
import speech_recognition as sr
from langchain_ollama import OllamaLLM
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# Initialize Pygame
pygame.init()
model = OllamaLLM(model="llama3")
client = ElevenLabs(api_key="sk_714be7a72ac45024cf880f10c2ff71594c95369b315bc415")
r = sr.Recognizer()

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

# Ball initial random positions
random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]

# State Variables
model_answering = False
is_collided = False


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
                    print("Jarvis detected!")
                    audio = client.generate(text="Yes sir", voice="Brian")  # Respond with "Yes sir"
                    play(audio)
                    continue
                else:
                    return MyText

        except sr.RequestError as e:
            print(f"API Request Error: {e}")
            return "Error: API unavailable"
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Please try again.")

def chatbot():
    global model_answering
    print("Welcome to the Llama 3 chatbot! Say 'exit' to end the conversation.")

    while True:
        user_input = record_text()
        if user_input.lower() == "exit":
            print("Goodbye!")
            audio = client.generate(text="Goodbye!", voice="Brian")
            play(audio)
            break

        # Start answering state
        model_answering = True
        result = model.invoke(input=user_input)

        print(f"Llama 3: {result}")
        #audio = client.generate(text=result, voice="Brian")
        #play(audio)
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

    # Toggle behavior based on model answering
    if model_answering:
        draw_response()
    else:
        draw_default()

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
