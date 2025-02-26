"""Manual and Automatic Control for CARLA with Confidence Visualization."""

# Add CARLA PythonAPI to sys.path
import sys
import os

carla_path = r"C:\Users\jpgar\CARLA_0.9.14\WindowsNoEditor\PythonAPI"
if carla_path not in sys.path:
    sys.path.append(carla_path)
    sys.path.append(os.path.join(carla_path, "carla"))  # Ensure the carla module is found

# Now import CARLA modules
import carla
import pygame
import math
import random
import time
import glob
import numpy as np
from agents.navigation.behavior_agent import BehaviorAgent

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.confidence_bar_width = 200
        self.confidence_bar_height = 10
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)

    def tick(self, world, clock):
        """Update the HUD elements."""
        self.server_fps = self._server_clock.get_fps()
        self.frame = world.world.get_snapshot().frame
        self.simulation_time = world.world.get_snapshot().timestamp.elapsed_seconds

    def render_confidence_bar(self, display, confidence):
        x = self.dim[0] // 2 - self.confidence_bar_width // 2
        y = 20  # Position at the top of the screen

        # Determine bar color based on confidence level
        if confidence > 0.7:
            color = (0, 255, 0)  # Green
        elif confidence > 0.3:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red

        # Draw bar background
        pygame.draw.rect(display, (50, 50, 50), (x, y, self.confidence_bar_width, self.confidence_bar_height))
        # Draw confidence level
        pygame.draw.rect(display, color,
                         (x, y, int(self.confidence_bar_width * confidence), self.confidence_bar_height))

        # Display percentage text
        text_surface = self._font_mono.render(f"Confidence: {int(confidence * 100)}%", True, (255, 255, 255))
        display.blit(text_surface, (x + self.confidence_bar_width // 2 - 30, y + 15))

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def render(self, display, world):
        self.render_confidence_bar(display, world.confidence_level)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.hud = hud
        self.confidence_level = 1.0  # Default: 100% confidence
        self.is_autonomous = False  # Mode toggle
        self.restart(args)
        self.agent = BehaviorAgent(self.player, behavior="normal")
        self.destination = self.get_random_destination()
        self.agent.set_destination(self.destination)
        self.world.on_tick(self.update_confidence)

    def restart(self, args):
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(10):
            self.player = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
            if self.player:
                print("Vehicle spawned successfully!")
                break

        if not self.player:
            raise RuntimeError("Failed to spawn vehicle. Check if the simulator is running properly.")

        # Attach a camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(self.hud.dim[0]))  # Match Pygame window width
        camera_bp.set_attribute("image_size_y", str(self.hud.dim[1]))  # Match Pygame window height
        camera_bp.set_attribute("fov", "110")  # Wide field of view

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
        self.camera.listen(lambda image: self.process_image(image))

        print("Camera successfully attached and configured!")

    def process_image(self, image):
        # Convert raw data to a NumPy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)

        # Ensure correct shape (CARLA images are BGRA, not RGBA)
        array = array.reshape((image.height, image.width, 4))

        # Convert BGRA to RGB format
        array = array[:, :, :3][:, :, ::-1]  # Drop Alpha and swap BGR → RGB

        # Transpose axes to match Pygame format (CARLA images are stored differently)
        self.frame = np.transpose(array, (1, 0, 2))  # Swap axes for correct orientation

        print("Processed camera frame correctly!")  # Debugging

    def update_confidence(self, event_type="normal"):
        deviation = self.get_lane_deviation()
        self.confidence_level = max(0.0, 1.0 - (deviation * 0.5))

        if event_type == "collision":
            self.confidence_level = max(0.0, self.confidence_level - 0.3)
        elif event_type == "lane_invasion":
            self.confidence_level = max(0.0, self.confidence_level - 0.2)
        elif event_type == "complex_scenario":
            self.confidence_level = max(0.0, self.confidence_level - 0.1)
        else:
            self.confidence_level = min(1.0, self.confidence_level + 0.02)

        self.hud.notification(f"Confidence Level: {int(self.confidence_level * 100)}%")
        if deviation > 1.0:
            self.hud.notification("Warning: High deviation detected!")

    def get_lane_deviation(self):
        vehicle_pos = self.world.get_map().get_waypoint(self.player.get_location()).transform.location
        return math.sqrt((self.player.get_location().x - vehicle_pos.x)**2 + (self.player.get_location().y - vehicle_pos.y)**2)

    def get_random_destination(self):
        spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(spawn_points).location

    def tick(self, clock):
        """Update the world state."""
        if self.is_autonomous:
            if self.agent.done():
                self.destination = self.get_random_destination()
                self.agent.set_destination(self.destination)
            control = self.agent.run_step()
            control.manual_gear_shift = False
            self.player.apply_control(control)

        self.hud.tick(self, clock)

    def render(self, display):
        if hasattr(self, 'frame') and self.frame is not None:
            # Convert NumPy array to Pygame surface
            surface = pygame.surfarray.make_surface(self.frame)
            display.blit(surface, (0, 0))  # Draw camera feed correctly

        self.hud.render(display, self)  # Render the confidence bar on top

    def toggle_autonomy(self):
        """Switch between manual and automatic modes."""
        self.is_autonomous = not self.is_autonomous
        mode = "Autonomous" if self.is_autonomous else "Manual"
        self.hud.notification(f"Switched to {mode} mode")


class KeyboardControl(object):
    def __init__(self, world):
        self.world = world
        self._autopilot_enabled = False  # Start in manual mode
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world.hud.notification("Press 'T' to switch between manual and automatic mode", seconds=4.0)

    def parse_events(self):
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_t:
                    self.world.toggle_autonomy()
                if event.key == pygame.K_ESCAPE:
                    return True

        # **Handle manual control only if autopilot is disabled**
        if not self.world.is_autonomous:
            self._parse_vehicle_keys(keys)
            self.world.player.apply_control(self._control)

        return False

    def _parse_vehicle_keys(self, keys):
        """Process keyboard input for manual driving."""
        if keys[pygame.K_w]:
            self._control.throttle = min(self._control.throttle + 0.02, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[pygame.K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0.0

        steer_increment = 0.05
        if keys[pygame.K_a]:
            self._steer_cache -= steer_increment
        elif keys[pygame.K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache *= 0.9  # Gradually return to center

        self._steer_cache = max(-1.0, min(1.0, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)

        self._control.hand_brake = keys[pygame.K_SPACE]


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)
        sim_world = client.get_world()

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world)

        clock = pygame.time.Clock()
        sim_world.tick()

        while True:
            clock.tick_busy_loop(60)

            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world is not None and hasattr(world, 'player') and world.player is not None:
            world.player.destroy()
        pygame.quit()

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='CARLA Manual & Automatic Control Client')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='Window resolution')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
