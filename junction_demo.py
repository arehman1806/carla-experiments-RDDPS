# Import necessary modules from CARLA
import carla
import random
import time
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent

# Connect to the server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # seconds

# Get the world
world = client.get_world()

# Get a random vehicle blueprint
bp_library = world.get_blueprint_library()
vehicle_bp = random.choice(bp_library.filter('vehicle.*'))

# Choose a random spawn point
spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

# Spawn the vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
spectator = world.get_spectator()
transform_vehicle = vehicle.get_transform()
spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
time.sleep(5)




# Add a controller to the vehicle
controller = carla.VehicleControl()
controller.steer = 0
controller.throttle = 0.5
controller.brake = 0.0
controller.hand_brake = False
controller.reverse = False

# Drive forward
vehicle.apply_control(controller)

# Define the location to turn right
turn_right_location = carla.Location(x=10, y=10, z=1)

# Control loop
while True:
    # Get the current location of the vehicle
    current_location = vehicle.get_location()

    # Check if the vehicle is near the turn right location
    if current_location.distance(turn_right_location) < 5.0:
        # Start turning right
        controller.steer = 0.3
        vehicle.apply_control(controller)
    elif current_location.distance(turn_right_location) < 1.0:
        # Straighten the vehicle after the turn
        controller.steer = 0
        vehicle.apply_control(controller)
        break

controller.stop()
vehicle.destroy()

