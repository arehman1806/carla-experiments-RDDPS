import carla
import random
import time
from agents.navigation.stand_still_agent import StandStillAgent
from scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import numpy as np
import cv2
from queue import Queue

from junction_annotator import JunctionAnnotator
from state_extractor import StateExtractor

from yolov7_carla_object_detection.carla_detect import CarlaObjectDetector
from typing import List, Tuple
from collections import deque
import copy
import csv
import os
import subprocess
from scenario_runner.srunner.tools.route_manipulation import interpolate_trajectory, interpolate_wp_trajectory


class DatasetGenPerceptionModelAgent:

    def __init__(self, start_timestamp, scenario_name, ego_junction_distance, actor_junction_distance) -> None:
        # Connect to the server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(1000.0)  # seconds
        # Get the world
        self.world = self.client.get_world()

        # remove stationary env vehicles:
        env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Vehicles)
        objects_to_toggle = {car.id for car in env_objs}
        self.world.enable_environment_objects(objects_to_toggle, False)
        
        self.map = self.world.get_map()
        self.dummy_tick = False

        self.current_image_index = 0
        self.current_rgb_image_index = 0
        self.data_collection_started = False
        self.frames_skipped = []
        self.start_timestamp = start_timestamp
        self.scenario_name = scenario_name
        self.ego_junction_distance = ego_junction_distance
        self.actor_junction_distance = actor_junction_distance
        self.no_skipped_frames = 0
        dir_path = f"./recording/{self.start_timestamp}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist


        self.ego_vehicle, self.other_vehicles = self.get_ego_and_other_vehicles()
        self.active_scenario_vehicle = self.other_vehicles[0]
        self.agent = StandStillAgent(self.ego_vehicle)

        # spawn the sensor and attach to vehicle.
        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        cam_bp.set_attribute('image_size_x', '608')
        cam_bp.set_attribute('image_size_y', '608')
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))
        self.sensor = self.world.spawn_actor(cam_bp, sensor_transform, attach_to=self.ego_vehicle)
        self.sensor.listen(lambda image: self.process_img(image))


        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '608')
        cam_bp.set_attribute('image_size_y', '608')
        # cam_bp.set_attribute('motion_blur_intensity', '1')
        # cam_bp.set_attribute('motion_blur_max_distortion', '1')
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))
        self.rgb_sensor = self.world.spawn_actor(cam_bp, sensor_transform, attach_to=self.ego_vehicle)
        self.rgb_sensor.listen(lambda image: self.process_img(image, rgb=True))


        # Get the attributes from the camera
        image_w = cam_bp.get_attribute("image_size_x").as_int()
        image_h = cam_bp.get_attribute("image_size_y").as_int()
        fov = cam_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = self.build_projection_matrix(image_w, image_h, fov)
        self.current_camera_image = None

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

        self.ego_traffic_light, self.op_traffic_light = self.get_traffic_lights()
        affected_waypoints = self.op_traffic_light.get_affected_lane_waypoints()
        num_wp = 0
        ego_tl_point = self.ego_traffic_light.get_affected_lane_waypoints()[0]
        self.ego_stop_pt = self.get_junction_stop_point(ego_tl_point)
        self.actor_stop_pt = self.get_junction_stop_point(affected_waypoints[0])
        
        
        # set the spectator to ego vehicle
        spectator = self.world.get_spectator()
        transform_vehicle = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        time.sleep(1)


        
        self.junction_annotator = JunctionAnnotator(self.world, ego_vehicle=self.ego_vehicle, camera_bp=cam_bp, camera=self.sensor)
        self.state_extractor = StateExtractor()

        self.count = 0

        # Set the world in synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 0.05 seconds (20 FPS)
        self.world.apply_settings(settings)
    
    def get_junction_stop_point(self, current_tl_stop_point):
        intersection = self.get_next_intersection(current_tl_stop_point)
        ent_exit_pts = intersection.get_waypoints(carla.LaneType.Driving)
        junction_pts = []
        for sublish in ent_exit_pts:
            junction_pts.extend(sublish)
        loc_target = current_tl_stop_point.transform.location
        stop_point = None
        min_distance = 1e9
        for point in junction_pts:
            loc = point.transform.location
            dist = loc.distance(loc_target)
            if dist < min_distance:
                stop_point = point
                min_distance = dist
        return stop_point

    def get_next_intersection(self, waypoint):
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return waypoint.get_junction()
        else:
            return waypoint.get_junction()
    
    def process_img(self, image: carla.Image, rgb=False):
        # Convert the image from CARLA format to an OpenCV image (RGB)
        frame_id = image.frame
        image_file_name = f"{self.scenario_name}_{frame_id}"
        if not self.data_collection_started or not self.keep_frame(frame_id):
            # print(f"skipping img frame {frame_id}")
            return
        
        if rgb:
            # print(f"RGB image frame: {frame_id}")
            image.save_to_disk(f"./recording/{self.start_timestamp}/rgb/{image_file_name}.png")
            self.current_rgb_image_index = frame_id
        else:
            # print(f"image frame: {frame_id}")
            image.save_to_disk(f"./recording/{self.start_timestamp}/instance/{image_file_name}.png")
            self.current_image_index = frame_id

    def get_traffic_lights(self) -> Tuple[carla.TrafficLight, carla.TrafficLight]:
        ego_light = self.ego_vehicle.get_traffic_light()
        op_traffic_light = self.op_traffic_light = CarlaDataProvider.annotate_trafficlight_in_group(ego_light)["opposite"][0]
        return ego_light, op_traffic_light

    
    def get_ego_and_other_vehicles(self) -> Tuple[carla.Vehicle, List[carla.Actor]]:
        ego_vehicle = None
        while ego_vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            non_ego = []
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    ego_vehicle = vehicle
                else:
                    non_ego.append(vehicle)
        return ego_vehicle, non_ego
    
    def set_active_vehicle(self):
        for i in range(10000):
            for vehicle in self.other_vehicles:
                # print(vehicle.get_transform().location)
                if vehicle.get_transform().location.z > -3:
                    if vehicle.get_transform().location.x == 0 and vehicle.get_transform().location.y == 0:
                        return False
                    self.active_scenario_vehicle = vehicle
                    print(f"found after {i} ticks")
                    return True
            self.dummy_tick = True
            self.world.tick()
        return False

    
    def get_state(self):
        other_vehicle_loc = self.active_scenario_vehicle.get_location()
        other_vehicle_wp = self.map.get_waypoint(other_vehicle_loc)
        actor_distance = other_vehicle_loc.distance(self.actor_stop_pt.transform.location)
        if other_vehicle_wp.is_junction:
            actor_distance = -actor_distance
        
        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc)
        # ego_distance = ego_loc.distance(self.ego_stop_pt.transform.location)
        ego_distance = len(interpolate_wp_trajectory(self.world, [ego_wp, self.ego_stop_pt], 1))
        if ego_wp.is_junction:
            ego_distance = -len(interpolate_wp_trajectory(self.world, [self.ego_stop_pt, ego_wp], 1))
        
        
        return ego_distance, actor_distance

    
    def save_to_csv(self, data_list, filename="states.csv"):
        filepath = f"./recording/{self.start_timestamp}/{filename}"
        file_exists = os.path.isfile(filepath)
        headers = ['image_file_name', 'ego_distance', 'ego_velocity', 'actor_distance', 'ego_junction_distance', 'actor_junction_distance']
        with open(filepath, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            for data in data_list:
                writer.writerow(data)

    def keep_frame(self, frame_id):
        if self.no_skipped_frames == 0:
            return True
        if (frame_id - self.starting_frame) % self.no_skipped_frames == 0:
            return True
        return False

    def start_data_collection(self):
        data = []
        # Setting the light to green is the trigger for scenario. See DatasetGenSurrogateModel scenario
        self.ego_traffic_light.freeze(False)
        self.ego_traffic_light.set_state(carla.TrafficLightState.Green)
        print("Triggered the light")
        count = 0
        previous_ego_distance, previous_actor_distance = 0, 0

        self.starting_frame = self.world.get_snapshot().frame
        self.data_collection_started = True
        while True:
            if self.active_scenario_vehicle.get_transform().location.z < -4:
                if not self.set_active_vehicle():
                    print("no active non scenario vehicle for 100 ticks")
                    break
            
            frame_id = self.world.get_snapshot().frame
            if not self.keep_frame(frame_id):
                # print(f"skipping frame {frame_id}")
                self.world.tick()
                continue
            
            if self.agent.done():
                print("The target has been reached, stopping the simulation")
                break
            
            while self.current_image_index < frame_id or self.current_rgb_image_index < frame_id:
                time.sleep(0.1)
                # print("waiting for image sync")
            self.world.tick()
        self.sensor.destroy()
    
    def read_csv_frames(self, filename="states.csv"):
        filepath = f"./recording/{self.start_timestamp}/{filename}"
        if not os.path.exists(filepath):
            return []
        
        frames = []
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                frames.append(row['image_file_name'][:-2])
        return frames

    def get_saved_frames(self, directory_name):
        dir_path = f"./recording/{self.start_timestamp}/{directory_name}"
        if not os.path.exists(dir_path):
            return []
        
        frames = [filename.split('.')[0] for filename in os.listdir(dir_path) if filename.endswith('.png')]
        return frames
    
    def delete_csv_entry(self, frame_id, filename="states.csv"):
        filepath = f"./recording/{self.start_timestamp}/{filename}"
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as csvfile:
            lines = csvfile.readlines()
        with open(filepath, 'w') as csvfile:
            for line in lines:
                if line.split(',')[0] != str(frame_id):
                    csvfile.write(line)
    
    
    def verify_and_delete_mismatched_frames(self):
        csv_frames = self.read_csv_frames()
        saved_instance_frames = self.get_saved_frames("instance")
        saved_rgb_frames = self.get_saved_frames("rgb")

        # Frames present in CSV but not in the saved images
        for frame in csv_frames:
            if frame not in saved_instance_frames:
                # Deleting CSV entry
                self.delete_csv_entry(frame)
            if frame not in saved_rgb_frames:
                # Deleting CSV entry
                self.delete_csv_entry(frame)

        # Frames present in saved images but not in the CSV
        for frame in saved_instance_frames:
            if frame not in csv_frames:
                os.remove(f"./recording/{self.start_timestamp}/instance/{frame}.png")
        
        for frame in saved_rgb_frames:
            if frame not in csv_frames:
                os.remove(f"./recording/{self.start_timestamp}/rgb/{frame}.png")



    
    def cleanup(self):
        CarlaDataProvider.cleanup()
        self.sensor.destroy()

if __name__ == "__main__":
    import subprocess
    import os
    import signal
    junction_params = {
        1: [30, 31],
        2: [30, 32],
        3: [31, 37],
        4: [29, 36]
    }
    start_record_name = time.time()
    start_record_name = "new_dataset"
    try:
        for i in range(1, 5):
            # carla_simulator = subprocess.Popen("/opt/carla-simulator/CarlaUE4.sh -prefernvidia", shell=True, preexec_fn=os.setsid)
            # print("waiting for simulator startup")
            # time.sleep(20)
            scenario_name = f"DatasetGenPerceptionModelRS_{i}"
            junction_param = junction_params.get(i)
            print(f"starting scenario: {scenario_name}")
            # scenario = subprocess.Popen(["python3", "./scenario_runner/scenario_runner.py", "--scenario", scenario_name, "--reloadWorld"], preexec_fn=os.setsid)
            print("waiting for scenario to start")
            # time.sleep(30)
            print("scenario started")
            print("starting data collection")
            datasetgenerator = DatasetGenPerceptionModelAgent(start_record_name, scenario_name, junction_param[0], junction_param[1])
            datasetgenerator.start_data_collection()
            # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
            # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
            # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
            # os.killpg(os.getpgid(carla_simulator.pid), signal.SIGTERM)
            # os.killpg(os.getpgid(carla_simulator.pid), signal.SIGTERM)
            # time.sleep(2)
            # os.killpg(os.getpgid(carla_simulator.pid), signal.SIGTERM)
            # print("killing the simulator")
            # time.sleep(10)
        datasetgenerator.verify_and_delete_mismatched_frames()
    except KeyboardInterrupt:
        # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
        # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
        # os.killpg(os.getpgid(scenario.pid), signal.SIGTERM)
        datasetgenerator.cleanup()
    
