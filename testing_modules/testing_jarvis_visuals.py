import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Screen Dimensions (Get the actual screen resolution)
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
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
max_radius = min(WIDTH, HEIGHT) // 3  # Scale the maximum radius based on screen size
angle = 0
speed = 1  # Initial rotation speed

# Particle Parameters
particles = []
num_particles = 100

# Pulse effect variables
pulse_factor = 1  # To control the size of the particles for pulsing effect
pulse_speed = 0.05  # The speed at which the pulse effect grows/shrinks
min_size = 3  # Minimum size of particles
max_size = 3  # Maximum size of particles

# Color Transition
current_color_1 = list(BLUE)
current_color_2 = list(CYAN)
target_color_1 = list(BLUE)
target_color_2 = list(CYAN)
color_transition_speed = 10  # The speed of color transition (lower is faster)

# Ball initial random positions
random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]
initial_positions = random_particles.copy()  # Save the initial random positions


def blend_color(current, target, speed):
    """Gradually transitions the current color toward the target color."""
    for i in range(3):  # Iterate through RGB components
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

            # Pulse effect: change particle size over time
            pulse_factor = min(max_size, pulse_factor + pulse_speed) if pulse_factor < max_size else max(min_size, pulse_factor - pulse_speed)
        else:
            # Move particles randomly when the A key is not pressed
            particle["x"] += particle["dx"]
            particle["y"] += particle["dy"]

            # Keep particles within the screen bounds
            if particle["x"] <= 0 or particle["x"] >= WIDTH:
                particle["dx"] *= -1
            if particle["y"] <= 0 or particle["y"] >= HEIGHT:
                particle["dy"] *= -1

        # Draw the particle with the pulsing size
        pygame.draw.circle(surface, tuple(current_color_2), (int(particle["x"]), int(particle["y"])), int(pulse_factor))


def generate_particles():
    """Generate new random particles."""
    global random_particles
    random_particles = [{"x": random.randint(0, WIDTH), "y": random.randint(0, HEIGHT), "dx": random.uniform(-2, 2), "dy": random.uniform(-2, 2)} for _ in range(num_particles)]


def draw_text(surface, text, position, font, color):
    """Draws text onto the surface."""
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)


# Main Loop
running = True
is_collided = False
while running:
    screen.fill(BLACK)

    # Event Handling
    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check for "A" key press
    if keys[pygame.K_a]:
        target_color_1 = list(ORANGE1)
        target_color_2 = list(ORANGE2)
        speed = 1  # Keep rotation speed constant
        is_collided = True  # Particles should form a circle and pulse
        angle += speed  # Increment the rotation angle while "A" is pressed
    else:
        target_color_1 = list(BLUE)
        target_color_2 = list(CYAN)
        speed = 1  # Reset speed
        is_collided = False  # Particles are scattered randomly

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
