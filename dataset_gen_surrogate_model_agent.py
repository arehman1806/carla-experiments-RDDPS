# Import necessary modules from CARLA
import carla
import random
import time
from agents.navigation.stand_still_agent import StandStillAgent
from agents.navigation.rddps_agent import RDDPSAgent
import py_trees
# from agents.navigation.behavior_agent import BehaviorAgent
import numpy as np
import torch
import cv2

from queue import Queue

from junction_annotator import JunctionAnnotator
from state_extractor import StateExtractor

from yolov7_carla_object_detection.carla_detect import CarlaObjectDetector

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

camera_image_queue = Queue()
def process_img(image):
    # Convert the image from CARLA format to an OpenCV image (RGB)
    img0 = np.array(image.raw_data).reshape((image.height, image.width, 4))
    img0 = img0[:, :, :3]
    camera_image_queue.put(np.array(img0))


annotated_image_queue = []




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
cam_bp.set_attribute('image_size_x', '1216')
cam_bp.set_attribute('image_size_y', '1216')
sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))

# spawn the sensor and attach to vehicle.
sensor = world.spawn_actor(cam_bp, sensor_transform, attach_to=ego_vehicle)
sensor.listen(lambda image: process_img(image))
# sensor.listen(lambda image: image.save_to_disk(f"./recording/{image.frame}.png"))

agent = StandStillAgent(ego_vehicle)
destination = carla.Location(x=-77.6, y=60, z=0)
agent.set_destination(destination)

world = client.get_world()
count = 0

# initialize object detector
obj_detector = CarlaObjectDetector()
junction_annotator = JunctionAnnotator(world, ego_vehicle=ego_vehicle, camera_bp=cam_bp, camera=sensor)
state_extractor = StateExtractor()
    
def annotatate_with_junction(camera_image, bbox):
    for i in range(len(bbox) - 1):
        p1 = bbox[i]
        p2 = bbox[i+1]
        cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
    p1 = bbox[0]
    p2 = bbox[len(bbox) - 1]
    cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
    return camera_image


traffic_light = ego_vehicle.get_traffic_light()
traffic_light.set_state(carla.TrafficLightState.Green)
print("Triggered the light")

while True:
    if agent.done():
        print("The target has been reached, stopping the simulation")
        break
    at_junction = False
    if not camera_image_queue.empty():
        camera_image = camera_image_queue.get()
        detections, annotated_camera_image = obj_detector.detect(camera_image)
        junction_bbox = junction_annotator.annotate()
        # print(junction_bbox)
        if junction_bbox is not None:
            at_junction, distance_to_junction = state_extractor.check_overlap(detections, junction_bbox)
            # print(f"{at_junction}, {distance_to_junction}")
        else:
            distance_to_junction = "car not near junction yet"
        if junction_bbox is not None:
            pass
            # annotated_camera_image = annotatate_with_junction(annotated_camera_image, junction_bbox)
        cv2.imshow('Annotated Images',annotated_camera_image)
        if cv2.waitKey(1) == ord('q'):
            break
        # cv2.imshow("annotated images", annotated_image_queue.pop())
        # cv2.imwrite(f"./recording/{count}.png", annotated_image_queue.pop())
        count += 1
    ego_vehicle.apply_control(agent.run_step(non_ego_at_junction=at_junction))
    world.tick()
sensor.destroy()

