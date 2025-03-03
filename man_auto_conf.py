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
import csv
from agents.navigation.behavior_agent import BehaviorAgent
from configparser import ConfigParser
from visual_feedback_types import integrate_with_hud

os.environ["SDL_XINPUT_ENABLED"] = "0"  # Disable XInput (prevents merged triggers)
os.environ["SDL_JOYSTICK_HIDAPI"] = "1"  # Use HIDAPI for raw DirectInput
os.environ["SDL_GAMECONTROLLERCONFIG"] = ""  # Reset any previous SDL mappings

class HUD:
    def __init__(self, width, height, show_confidence=True):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        self.notifications = []
        self.speed = 0.0
        self.radar_distance = "N/A"
        self.radar_speed = "N/A"
        self.confidence_level = 1.0  # Default max confidence
        self.show_confidence = show_confidence  # Toggle for confidence display

    def notification(self, text):
        """Adds a notification to be displayed on HUD."""
        self.notifications.append((text, pygame.time.get_ticks()))

    def tick(self, world, clock):
        """Updates HUD info from vehicle and sensors."""
        if world.player:
            vel = world.player.get_velocity()
            self.speed = math.sqrt(vel.x**2 + vel.y**2) * 3.6  # Convert m/s to km/h
        else:
            self.speed = 0.0

    def toggle_confidence_display(self):
        """Toggle whether to show the confidence bar."""
        self.show_confidence = not self.show_confidence
        state = "ON" if self.show_confidence else "OFF"
        self.notification(f"Confidence Display: {state}")

    def render(self, display, world):
        # Create a transparent surface
        hud_surface = pygame.Surface((self.dim[0], 150), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 80))  # Black with 80 transparency

        # Speed Display
        speed_text = self.font.render(f"{int(self.speed)} km/h", True, (255, 255, 255))
        hud_surface.blit(speed_text, (self.dim[0] // 2 - 40, 30))

        # Radar Display
        radar_text = self.small_font.render(f"Radar: {self.radar_distance}m | {self.radar_speed} m/s", True,
                                            (255, 200, 0))
        hud_surface.blit(radar_text, (20, 60))

        # Confidence Bar - only show if enabled
        if self.show_confidence:
            bar_x, bar_y = self.dim[0] // 2 - 100, 100
            pygame.draw.rect(hud_surface, (50, 50, 50), (bar_x, bar_y, 200, 10), 2)

            # Determine color based on confidence
            if world.confidence_level > 0.7:
                color = (0, 255, 0)  # Green
                # No warning needed
            elif world.confidence_level > 0.3:
                color = (255, 255, 0)  # Yellow
                # Low confidence warning
                if world.is_autonomous:
                    warning_text = self.small_font.render("Low Confidence - Be Ready", True, (255, 255, 0))
                    hud_surface.blit(warning_text, (bar_x + 210, bar_y))
            else:
                color = (255, 0, 0)  # Red
                # Critical confidence warning - flashing
                if world.is_autonomous and int(time.time() * 2) % 2 == 0:  # Flash twice per second
                    warning_text = self.font.render("TAKEOVER RECOMMENDED", True, (255, 0, 0))
                    hud_surface.blit(warning_text, (self.dim[0] // 2 - warning_text.get_width() // 2, bar_y - 30))

            pygame.draw.rect(hud_surface, color,
                             (bar_x, bar_y, int(200 * world.confidence_level), 10))

            # Display percentage text
            text_surface = self.small_font.render(f"Confidence: {int(world.confidence_level * 100)}%", True,
                                                  (255, 255, 255))
            hud_surface.blit(text_surface, (bar_x + 100 - 30, bar_y + 15))

        # Autonomous Mode Indicator
        mode_text = "Autonomous" if world.is_autonomous else "Manual"
        mode_color = (0, 255, 0) if world.is_autonomous else (255, 0, 0)
        mode_display = self.font.render(mode_text, True, mode_color)
        hud_surface.blit(mode_display, (self.dim[0] - 150, 30))

        # Blit the transparent HUD onto the main display
        display.blit(hud_surface, (0, self.dim[1] - 150))

HUD = integrate_with_hud(HUD)

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
        self.radar = None
        self.restart(args)
        self.agent = BehaviorAgent(self.player, behavior="normal")
        self.destination = self.get_random_destination()
        self.agent.set_destination(self.destination)
        self.world.on_tick(self.update_confidence)
        self.last_human_control = carla.VehicleControl()  # Initialize with default
        self.takeover_threshold = 0.15  # Sensitivity for takeover detection
        self.last_takeover_time = 0  # To prevent rapid switching
        self.takeover_cooldown = 2.0  # Seconds between allowed takeovers

    def restart(self, args):
        blueprint_library = self.world.get_blueprint_library()
        blueprint = blueprint_library.find('vehicle.tesla.model3') #Imports Tesla M3
        spawn_points = self.world.get_map().get_spawn_points()

        for _ in range(10):
            self.player = self.world.try_spawn_actor(blueprint, random.choice(spawn_points))
            if self.player:
                print("Vehicle spawned successfully!")
                break

        if not self.player:
            raise RuntimeError("Failed to spawn vehicle. Check if the simulator is running properly.")

        #Vehicle Physics (assuming Tesla M3)
        physics_control = self.player.get_physics_control()
        physics_control.mass = 1611
        physics_control.drag_coefficient = 0.23
        physics_control.center_of_mass = carla.Vector3D(0, 0, -0.5)
        physics_control.torque_curve = [carla.Vector2D(0, 400), carla.Vector2D(1, 400)]  # Flat torque curve
        physics_control.max_rpm = 18000  # High RPM for electric motors
        physics_control.moi = 1.0  # Moment of inertia
        physics_control.damping_rate_full_throttle = 0.0
        physics_control.damping_rate_zero_throttle_clutch_engaged = 2.0
        physics_control.damping_rate_zero_throttle_clutch_disengaged = 0.5
        physics_control.use_gear_autobox = True
        physics_control.gear_switch_time = 0.1  # Quick gear shifts
        physics_control.clutch_strength = 10.0
        physics_control.final_ratio = 9.0  # Adjust final drive ratio
        physics_control.max_velocity = 72.5

        self.player.apply_physics_control(physics_control)

        # Attach a camera in the **driver's perspective**
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(self.hud.dim[0]))  # Match display width
        camera_bp.set_attribute("image_size_y", str(self.hud.dim[1]))  # Match display height
        camera_bp.set_attribute("fov", "90")  # Realistic field of view

        #**Driver's view position**
        camera_transform = carla.Transform(
            carla.Location(x=0.5, y=0.0, z=1.5),  # X: Move slightly forward, Y: Center, Z: Eye level
            carla.Rotation(pitch=0, yaw=0, roll=0)  # Keep level with the dashboard
        )

        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
        self.camera.listen(lambda image: self.process_image(image))

        print("Camera set to driver's perspective!")

        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute("horizontal_fov", "45")
        radar_bp.set_attribute("vertical_fov", "20")
        radar_bp.set_attribute("range", "50")
        radar_transform = carla.Transform(carla.Location(x=2.0, y=0.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.player)
        self.radar.listen(lambda radar_data: self.process_radar(radar_data))

    def process_image(self, image):
        # Convert raw data to a NumPy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)

        # Ensure correct shape (CARLA images are BGRA, not RGBA)
        array = array.reshape((image.height, image.width, 4))

        # Convert BGRA to RGB format
        array = array[:, :, :3][:, :, ::-1]  # Drop Alpha and swap BGR → RGB

        #flipping
        #array = array[:, ::-1. :]

        # Transpose axes to match Pygame format (CARLA images are stored differently)
        self.frame = np.transpose(array, (1, 0, 2))  # Swap axes for correct orientation

        #print("Processed camera frame correctly!")  # Debugging

    def update_confidence(self, event_type="normal"):
        """Update the system's confidence level based on specified events rather than lane deviation."""
        previous_confidence = self.confidence_level

        # Only modify confidence based on specific events or scenarios
        if event_type == "collision":
            self.confidence_level = max(0.0, self.confidence_level - 0.3)
        elif event_type == "lane_invasion":
            self.confidence_level = max(0.0, self.confidence_level - 0.2)
        elif event_type == "complex_scenario":
            self.confidence_level = max(0.0, self.confidence_level - 0.1)
        elif event_type == "curve_ahead":
            self.confidence_level = max(0.0, self.confidence_level - 0.05)
        elif event_type == "restore":
            # Gradually restore confidence
            self.confidence_level = min(1.0, self.confidence_level + 0.02)

        # If confidence changed significantly, show notification
        if abs(previous_confidence - self.confidence_level) > 0.05:
            self.hud.notification(f"Confidence Level: {int(self.confidence_level * 100)}%")

            if self.confidence_level < 0.3 and previous_confidence >= 0.3:
                self.hud.notification("WARNING: Low confidence, takeover recommended", seconds=3.0)

    def get_lane_deviation(self):
        if not self.player or not self.player.is_alive:
            print("Warning: Attempted to get lane deviation, but vehicle is missing.")
            return 0.0  # Return default deviation
        try:
            vehicle_pos = self.world.get_map().get_waypoint(self.player.get_location()).transform.location
            return math.sqrt((self.player.get_location().x - vehicle_pos.x) ** 2 + (
                        self.player.get_location().y - vehicle_pos.y) ** 2)
        except RuntimeError:
            print("Warning: Player vehicle was destroyed mid-operation.")
            return 0.0

    def get_random_destination(self):
        spawn_points = self.world.get_map().get_spawn_points()
        return random.choice(spawn_points).location

    def process_radar(self, radar_data):
        min_distance = float('inf')
        max_speed = 0
        for detection in radar_data:
            if detection.depth < min_distance:
                min_distance = detection.depth
                max_speed = detection.velocity
        self.hud.radar_distance = f"{min_distance:.2f}"
        self.hud.radar_speed = f"{max_speed:.2f}"

    def detect_human_input(self):
        """Detect significant human input for takeover."""
        # Initialize attributes if not present
        if not hasattr(self, 'takeover_threshold'):
            self.takeover_threshold = 1.00  # Sensitivity for takeover detection
        if not hasattr(self, 'last_takeover_time'):
            self.last_takeover_time = 0  # To prevent rapid switching
        if not hasattr(self, 'takeover_cooldown'):
            self.takeover_cooldown = 2.0  # Seconds between allowed takeovers

        # Ignore small inputs (e.g., accidental touches)
        if hasattr(self, 'last_human_control'):
            # Check for significant steering input
            if abs(self.last_human_control.steer) > self.takeover_threshold:
                return True
            # Check for significant throttle input
            if self.last_human_control.throttle > self.takeover_threshold:
                return True
            # Check for any brake input (safety critical)
            if self.last_human_control.brake > self.takeover_threshold:
                return True
        return False

    def tick(self, clock):
        """Update the world state with automatic takeover detection."""
        current_time = time.time()

        # Handle autonomous mode
        if self.is_autonomous:
            # Check if agent needs a new destination
            if self.agent.done():
                # Find a new path ahead
                setup_better_agent(self)

            # Get AI control
            try:
                ai_control = self.agent.run_step()
                ai_control.manual_gear_shift = False
            except Exception as e:
                print(f"Agent error: {e}")
                ai_control = carla.VehicleControl()

            # Check for human takeover
            if self.detect_human_input() and (current_time - self.last_takeover_time > self.takeover_cooldown):
                self.is_autonomous = False
                self.last_takeover_time = current_time
                self.hud.notification("Human takeover detected")

                # Play a sound if possible
                try:
                    pygame.mixer.init()
                    pygame.mixer.Sound("takeover_sound.wav").play()
                except:
                    pass  # Sound is optional
            else:
                # Apply AI control if still autonomous
                if self.is_autonomous:
                    self.player.apply_control(ai_control)

        # Occasionally test confidence level changes (for testing purposes)
        # Uncomment for testing only
        # if self.is_autonomous and random.random() < 0.005:  # 0.5% chance per tick
        #     self.update_confidence("curve_ahead")
        # elif self.is_autonomous and random.random() < 0.01:  # 1% chance per tick
        #     self.update_confidence("restore")

        # Always update other HUD elements
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world view and HUD."""
        if hasattr(self, 'frame') and self.frame is not None:
            surface = pygame.surfarray.make_surface(self.frame)
            display.blit(surface, (0, 0))  # Draw camera feed

        self.hud.render(display, self)  # Render full HUD elements

    def toggle_autonomy(self):
        self.is_autonomous = not self.is_autonomous
        mode = "Autonomous" if self.is_autonomous else "Manual"
        self.hud.notification(f"Switched to {mode} mode")


#class KeyboardControl(object):
    #def __init__(self, world):
        ##self.world = world
        #self._control = carla.VehicleControl()
        #self._steer_cache = 0.0
        #self._reverse = False
       # world.hud.notification("Press 'T' to switch between manual and automatic mode")

  #  def parse_events(self):
    #    keys = pygame.key.get_pressed()
     #   for event in pygame.event.get():
       #     if event.type == pygame.QUIT:
        #        return True
        #    if event.type == pygame.KEYUP:
         #       if event.key == pygame.K_t:
         #           self.world.toggle_autonomy()
         #       if event.key == pygame.K_q:  # Toggle reverse gear only once per press
          #          self._reverse = not self._reverse
          ##          self._control.gear = -1 if self._reverse else 1
          #          self.world.hud.notification(f"Reverse: {'ON' if self._reverse else 'OFF'}")
          #      if event.key == pygame.K_ESCAPE:
          #          return True

    #    if not self.world.is_autonomous:
    #        self._parse_vehicle_keys(keys)
    #        self.world.player.apply_control(self._control)
   #     return False

  #  def _parse_vehicle_keys(self, keys):
  #      if keys[pygame.K_w]:
  #          self._control.throttle = min(self._control.throttle + 0.02, 1.00)
  #      else:
  #          self._control.throttle = 0.0

  #      if keys[pygame.K_s]:
  #          self._control.brake = min(self._control.brake + 0.2, 1)
  #      else:
  #          self._control.brake = 0.0

        # Reverse toggle logic
 #       if keys[pygame.K_q]:  # Toggle reverse mode
   #         self._reverse = not self._reverse
    #        self._control.gear = -1 if self._reverse else 1
     #       self.world.hud.notification(f"Reverse: {'ON' if self._reverse else 'OFF'}")

        # Ensure gear is correctly set when reversing
    #    if self._reverse and keys[pygame.K_w]:  # Apply reverse throttle
    #        self._control.throttle = min(self._control.throttle + 0.02, 1.00)
    #        self._control.gear = -1
    #    elif not self._reverse:
    #        self._control.gear = 1  # Reset to forward if not reversing

   #     steer_increment = 0.05
    #    if keys[pygame.K_a]:
     #       self._steer_cache -= steer_increment
     #   elif keys[pygame.K_d]:
     #       self._steer_cache += steer_increment
     #   else:
    #        self._steer_cache *= 0.9

     #   self._steer_cache = max(-1.0, min(1.0, self._steer_cache))
    #    self._control.steer = round(self._steer_cache, 1)
     #   self._control.hand_brake = keys[pygame.K_SPACE]

class SteeringWheelControl(object):
    def __init__(self, world):
        self.world = world
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._reverse = False
        self.toggle_pressed = False
        world.hud.notification("Steering Wheel Control Active")

        pygame.joystick.init()
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        # Load configuration file
        self._parser = ConfigParser()
        config_file = "wheel_config.ini"

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Missing {config_file}. Create it with correct wheel mappings.")

        self._parser.read(config_file)

        # Try both G923 and G29 sections
        if self._parser.has_section("G923 Racing Wheel"):
            wheel_type = "G923 Racing Wheel"
        elif self._parser.has_section("G29 Racing Wheel"):
            wheel_type = "G29 Racing Wheel"
        else:
            raise configparser.NoSectionError("No valid wheel section found in wheel_config.ini")

        try:
            self._steer_idx = int(self._parser.get(wheel_type, "steering_wheel"))
            self._throttle_idx = int(self._parser.get(wheel_type, "throttle"))
            self._brake_idx = int(self._parser.get(wheel_type, "brake"))
            self._reverse_idx = int(self._parser.get(wheel_type, "reverse"))
            self._handbrake_idx = int(self._parser.get(wheel_type, "handbrake"))
        except ValueError as e:
            raise ValueError(f"Invalid config value in {config_file}: {e}")

        print(f"Using {wheel_type} for input mapping.")

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self.world.hud.change_display_type("none")
                elif event.key == pygame.K_F2:
                    self.world.hud.change_display_type("bar")
                elif event.key == pygame.K_F3:
                    self.world.hud.change_display_type("gauge")
                elif event.key == pygame.K_F4:
                    self.world.hud.change_display_type("workload")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:  # 'C' for confidence toggle
                        self.world.hud.toggle_confidence_display()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.world.toggle_autonomy()
                elif event.key == pygame.K_ESCAPE:
                        return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == self._reverse_idx:
                    self._reverse = not self._reverse
                    self._control.gear = -1 if self._reverse else 1
                    self.world.hud.notification(f"Reverse: {'ON' if self._reverse else 'OFF'}")

        self._parse_wheel_inputs()
        self.world.player.apply_control(self._control)
        return False

    def _parse_wheel_inputs(self):
        """Parse steering wheel inputs and pass to world object."""
        self._control.steer = self._joystick.get_axis(self._steer_idx)
        pedal_input = self._joystick.get_axis(self._throttle_idx)
        brake_input = self._joystick.get_axis(self._brake_idx)

        # Check toggle button pressed state
        toggle_button = 4  # This is your button index for autonomous toggle
        toggle_button_pressed = self._joystick.get_button(toggle_button)

        # Only toggle when button is pressed and wasn't pressed before
        if toggle_button_pressed and not self.toggle_pressed:
            self.world.toggle_autonomy()
            print("Toggling autonomy mode")

        #self._control.steer = -steerCmd

        # Update toggle pressed state
        self.toggle_pressed = toggle_button_pressed

        # Separate throttle & brake manually
        if pedal_input < 0:  # Negative values = Throttle
            self._control.throttle = abs(pedal_input)
            self._control.brake = 0.0
        else:  # Positive values = Brake
            self._control.brake = pedal_input
            self._control.throttle = 0.0

        # Store raw control in world object for takeover detection
        self.world.last_human_control = carla.VehicleControl(
            throttle=self._control.throttle,
            steer=self._control.steer,
            brake=self._control.brake,
            hand_brake=self._control.hand_brake,
            reverse=self._reverse
        )

        # Only apply control if in manual mode
        if not self.world.is_autonomous:
            self.world.player.apply_control(self._control)

class DataLogger:
    def __init__(self, output_dir="experiment_data"):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filename = f"{output_dir}/experiment_{timestamp}.csv"

        # Initialize the CSV file with headers
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'confidence', 'speed', 'steering',
                             'throttle', 'brake', 'lane_deviation',
                             'is_autonomous', 'show_confidence', 'participant_id'])

        self.participant_id = "test"  # Default ID

    def log_data(self, world):
        # Get current vehicle state
        if not world.player:
            return

        vehicle = world.player
        control = vehicle.get_control()
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # m/s to km/h

        # Calculate lane deviation
        lane_deviation = world.get_lane_deviation()

        # Write to CSV
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                time.time(),
                world.confidence_level,
                speed,
                control.steer,
                control.throttle,
                control.brake,
                lane_deviation,
                world.is_autonomous,
                world.hud.show_confidence,
                self.participant_id
            ])

    def set_participant_id(self, id):
        self.participant_id = id


def setup_highway_scenario(world):
    # Find a suitable highway section in Town06
    map = world.world.get_map()
    spawn_points = map.get_spawn_points()

    # Try to find a point on the highway
    highway_points = []
    for point in spawn_points:
        # Check if this point is on a highway (higher speed limit)
        waypoint = map.get_waypoint(point.location)
        if waypoint.lane_type == carla.LaneType.Driving and waypoint.lane_width > 3.5:
            highway_points.append(point)

    if highway_points:
        # Choose a random highway point
        spawn_point = random.choice(highway_points)

        # Create a proper transform object from the spawn point
        location = spawn_point.location
        rotation = spawn_point.rotation
        transform = carla.Transform(location, rotation)

        # Debug print
        print(f"Type of transform: {type(transform)}")

        # Set vehicle at this point
        world.player.set_transform(transform)

        # Find a destination along the highway
        current = map.get_waypoint(spawn_point.location)
        next_waypoints = []

        # Generate a path of waypoints
        for i in range(10):
            next_wp = current.next(30.0)[0]
            next_waypoints.append(next_wp)
            current = next_wp

        # Set the destination
        if next_waypoints:
            destination = next_waypoints[-1].transform.location
            world.agent.set_destination(destination)

            # Set autonomous mode
            world.is_autonomous = True
            world.hud.notification("Highway scenario started - Autonomous mode")

            return True

    world.hud.notification("Failed to find suitable highway point")
    return False


def setup_better_agent(world):
    """Create and configure a better driving agent."""
    if hasattr(world, 'agent') and world.agent:
        # Get current location
        vehicle = world.player
        if not vehicle:
            return False

        # Create a new behavior agent with better parameters
        world.agent = BehaviorAgent(world.player, behavior="cautious")  # Try "cautious" instead of "normal"

        # Set agent parameters - Handle differences in API versions
        try:
            # Try the newer API with opt_dict
            if hasattr(world.agent, 'opt_dict'):
                world.agent.opt_dict["target_speed"] = 30.0
                world.agent.opt_dict["max_steering"] = 0.5
                world.agent.opt_dict["ignore_vehicles"] = True
            # Try older API with direct attributes
            else:
                if hasattr(world.agent, 'target_speed'):
                    world.agent.target_speed = 30.0 / 3.6  # km/h to m/s
                if hasattr(world.agent, 'max_steering'):
                    world.agent.max_steering = 0.5
                # Ignore vehicles might not be available in all versions

            print("Agent parameters set successfully")
        except Exception as e:
            print(f"Warning: Could not set all agent parameters: {e}")

        # Find a reasonable destination
        spawn_points = world.world.get_map().get_spawn_points()
        if spawn_points:
            # Try to find one that's ahead on the same road
            current_waypoint = world.world.get_map().get_waypoint(vehicle.get_location())
            ahead_waypoints = []

            # Generate waypoints ahead
            current = current_waypoint
            for i in range(20):  # Look 20 waypoints ahead
                next_wps = current.next(10.0)
                if not next_wps:
                    break
                ahead_waypoints.append(next_wps[0])
                current = next_wps[0]

            if ahead_waypoints:
                # Set destination to the farthest waypoint
                destination = ahead_waypoints[-1].transform.location
                world.agent.set_destination(destination)
                print(f"Agent destination set {len(ahead_waypoints)} waypoints ahead")
                return True

        # Fallback: use a random spawn point as destination
        if spawn_points:
            destination = random.choice(spawn_points).location
            world.agent.set_destination(destination)
            print("Agent destination set to random spawn point")
            return True

    return False


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)
        sim_world = client.get_world()

        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height, display_type="bar")  # Default to showing confidence
        world = World(sim_world, hud, args)
        controller = SteeringWheelControl(world)

        # Initialize attributes for takeover detection
        world.last_human_control = carla.VehicleControl()
        world.takeover_threshold = 0.15
        world.last_takeover_time = 0
        world.takeover_cooldown = 2.0

        # Setup better autonomous driving
        setup_better_agent(world)

        # Set initial autonomous mode
        world.is_autonomous = True
        world.hud.notification("Starting in autonomous mode")

        # Add data logger
        data_logger = DataLogger()
        data_logger.set_participant_id("test")

        clock = pygame.time.Clock()
        sim_world.tick()

        while True:
            clock.tick_busy_loop(60)

            if controller.parse_events():
                return

            world.tick(clock)

            # Log data
            data_logger.log_data(world)

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
