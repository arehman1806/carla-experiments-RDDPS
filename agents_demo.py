# Import necessary modules from CARLA
import carla
import random
import time
from agents.navigation.rddps_agent import RDDPSAgent
# from agents.navigation.behavior_agent import BehaviorAgent
import numpy as np
import torch
import cv2

from yolov7_carla_object_detection.carla_detect import CarlaObjectDetector

annotated_image_queue = []


# initialize object detector
obj_detector = CarlaObjectDetector()

def process_img(image):
    # Convert the image from CARLA format to an OpenCV image (RGB)
    img0 = np.array(image.raw_data).reshape((image.height, image.width, 4))
    img0 = img0[:, :, :3]
    annotated_image = obj_detector.detect(img0)
    annotated_image_queue.append(annotated_image)
    

# Connect to the server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # seconds

# Get the world
world = client.get_world()

# Set the world in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 0.05 seconds (20 FPS)
world.apply_settings(settings)



ego_vehicle = None

# Get the ego vehicle
while ego_vehicle is None:
    print("Waiting for the ego vehicle...")
    time.sleep(1)
    possible_vehicles = world.get_actors().filter('vehicle.*')
    for vehicle in possible_vehicles:
        if vehicle.attributes['role_name'] == 'hero':
            print("Ego vehicle found")
            ego_vehicle = vehicle
            break


# Spawn the vehicle

spectator = world.get_spectator()
transform_vehicle = ego_vehicle.get_transform()
spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
time.sleep(1)

blueprint_library = world.get_blueprint_library()
cam_bp = blueprint_library.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '640')
cam_bp.set_attribute('image_size_y', '480')
sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))

# spawn the sensor and attach to vehicle.
sensor = world.spawn_actor(cam_bp, sensor_transform, attach_to=ego_vehicle)
sensor.listen(lambda image: process_img(image))
# sensor.listen(lambda image: image.save_to_disk(f"./recording/{image.frame}.png"))

agent = RDDPSAgent(ego_vehicle)
destination = carla.Location(x=-77.6, y=60, z=0)
agent.set_destination(destination)

world = client.get_world()
count = 0
while True:
    if agent.done():
        print("The target has been reached, stopping the simulation")
        break

    ego_vehicle.apply_control(agent.run_step())
    if len(annotated_image_queue) != 0:
        # cv2.imshow("annotated images", annotated_image_queue.pop())
        cv2.imwrite(f"./recording/{count}.png", annotated_image_queue.pop())
        count += 1
    world.tick()

