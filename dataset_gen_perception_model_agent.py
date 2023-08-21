# Import necessary modules from CARLA
import carla
import random
import time
from agents.navigation.stand_still_agent import StandStillAgent
from agents.navigation.rddps_agent import RDDPSAgent
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


class DatasetGenPerceptionModelAgent:

    def __init__(self) -> None:
        # Connect to the server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # seconds
        # Get the world
        self.world = self.client.get_world()

        self.map = self.world.get_map()
        self.dummy_tick = False

        self.current_image_index = 0
        self.current_rgb_image_index = 0
        self.data_collection_started = False
        self.frames_skipped = []
        self.start_timestamp = time.time()
        dir_path = f"./recording/{self.start_timestamp}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist



        self.camera_image_queue = deque()
        self.ego_vehicle, self.other_vehicles = self.get_ego_and_other_vehicles()
        self.active_scenario_vehicle = self.other_vehicles[0]
        self.agent = StandStillAgent(self.ego_vehicle)
        # destination = carla.Location(x=-77.6, y=60, z=0)
        # self.agent.set_destination(destination)

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
        cam_bp.set_attribute('motion_blur_intensity', '1')
        cam_bp.set_attribute('motion_blur_max_distortion', '1')
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
        # Wait for the world to be ready
        # if CarlaDataProvider.is_sync_mode():
        #     self.dummy_tick = True
        #     self.world.tick()
        # else:
        #     self.world.wait_for_tick()

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
        settings.fixed_delta_seconds = 0.05  # 0.05 seconds (20 FPS)
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
        # if self.dummy_tick:
        #     self.dummy_tick = False
        #     return
        frame_id = image.frame
        if not self.data_collection_started or frame_id in self.frames_skipped:
            print(f"skipping img frame {frame_id}")
            return
        
        if rgb:
            print(f"RGB image frame: {frame_id}")
            image.save_to_disk(f"./recording/{self.start_timestamp}/rgb/{frame_id}.png")
            self.current_rgb_image_index = frame_id
        else:
            print(f"image frame: {frame_id}")
            image.save_to_disk(f"./recording/{self.start_timestamp}/instance/{frame_id}.png")
            self.current_image_index = frame_id

    def get_traffic_lights(self) -> Tuple[carla.TrafficLight, carla.TrafficLight]:
        ego_light = self.ego_vehicle.get_traffic_light()
        op_traffic_light = self.op_traffic_light = CarlaDataProvider.annotate_trafficlight_in_group(ego_light)["opposite"][0]
        return ego_light, op_traffic_light

    
    def get_ego_and_other_vehicles(self) -> Tuple[carla.Vehicle, List[carla.Actor]]:
        ego_vehicle = None

        # Get the ego vehicle
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
        for i in range(10):
            for vehicle in self.other_vehicles:
                if vehicle.get_transform().location.z > 0:
                    self.active_scenario_vehicle = vehicle
                    print(f"found after {i} ticks")
                    return True
            self.dummy_tick = True
            self.world.tick()
        return False

    def annotatate_with_junction(self, camera_image, bbox):
        for i in range(len(bbox) - 1):
            p1 = bbox[i]
            p2 = bbox[i+1]
            cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
        p1 = bbox[0]
        p2 = bbox[len(bbox) - 1]
        cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
        return camera_image
    
    def get_state(self):
        other_vehicle_loc = self.active_scenario_vehicle.get_location()
        other_vehicle_wp = self.map.get_waypoint(other_vehicle_loc)
        actor_distance = other_vehicle_loc.distance(self.actor_stop_pt.transform.location)
        if other_vehicle_wp.is_junction:
            actor_distance = -actor_distance
        
        ego_loc = self.ego_vehicle.get_location()
        ego_wp = self.map.get_waypoint(ego_loc)
        ego_distance = ego_loc.distance(self.ego_stop_pt.transform.location)
        if ego_wp.is_junction:
            ego_distance = -ego_distance
        
        return ego_distance, actor_distance
    
    def get_ground_truth(self):
        # Get the camera matrix 
        world_2_camera = np.array(self.sensor.get_transform().get_inverse_matrix())
        bb = self.active_scenario_vehicle.bounding_box
        # p1 = self.get_image_point(bb.location, self.K, world_2_camera)
        verts = [v for v in bb.get_world_vertices(self.active_scenario_vehicle.get_transform())]
        x_max = -10000
        x_min = 10000
        y_max = -10000
        y_min = 10000

        for vert in verts:
            p = self.get_image_point(vert, self.K, world_2_camera)
            # Find the rightmost vertex
            if p[0] > x_max:
                x_max = p[0]
            # Find the leftmost vertex
            if p[0] < x_min:
                x_min = p[0]
            # Find the highest vertex
            if p[1] > y_max:
                y_max = p[1]
            # Find the lowest  vertex
            if p[1] < y_min:
                y_min = p[1]
        return (x_min, y_min, x_max, y_max)


    def draw_bb_on_image(self, xyxy, img):
        x_min, y_min, x_max, y_max = xyxy
        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    
    def save_to_csv(self, filename, data_list):
        file_exists = os.path.isfile(filename)
        headers = ['frame_id', 'ego_distance', 'ego_velocity', 'actor_distance']
        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            for data in data_list:
                writer.writerow(data)

    def start_data_collection(self):
        skip_frames = 2
        data = []
        # Setting the light to green is the trigger for scenario. See DatasetGenSurrogateModel scenario
        self.ego_traffic_light.set_state(carla.TrafficLightState.Green)
        print("Triggered the light")
        count = 0
        previous_ego_distance, previous_actor_distance = 0, 0

        self.data_collection_started = True
        while True:
            frame_id = self.world.get_snapshot().frame
            for _ in range(0, skip_frames):
                print(f"skipping frames {frame_id}")
                self.frames_skipped.append(frame_id)
                if len(self.frames_skipped) > 1000:
                    self.frames_skipped = self.frames_skipped[-100:]
                self.world.tick()
            if self.active_scenario_vehicle.get_transform().location.z < -4:
                if not self.set_active_vehicle():
                    print("no active non scenario vehicle for 100 ticks")
                    break
            
            if self.agent.done():
                print("The target has been reached, stopping the simulation")
                break
            
            while self.current_image_index < frame_id or self.current_rgb_image_index < frame_id:
                time.sleep(0.1)
                print("waiting for image sync")
            print(f"current frame: ", frame_id)
            previous_ego_distance, previous_actor_distance = self.get_state()
            for i, vel in enumerate([10,14,16,18,20,25]):
                frame_id_iv = f"{frame_id}_{i}"
                data.append({"frame_id": frame_id_iv, "ego_distance": previous_ego_distance, "ego_velocity": vel, "actor_distance": previous_actor_distance})
            # self.ego_vehicle.apply_control(self.agent.run_step())
            self.world.tick()
            if len(data) > 100:
                self.save_to_csv("scenario_1.csv", data)
                data = []
            # time.sleep(0.1)
        self.save_to_csv("scenario_1.csv", data)
        self.sensor.destroy()
    
    def cleanup(self):
        CarlaDataProvider.cleanup()
        self.sensor.destroy()

if __name__ == "__main__":
    datasetgenerator = DatasetGenPerceptionModelAgent()
    try:
        datasetgenerator.start_data_collection()
    except KeyboardInterrupt:
        
        datasetgenerator.cleanup()
    
