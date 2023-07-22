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


class DatasetGenSurrogateModel:

    def __init__(self) -> None:
        # Connect to the server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # seconds
        # Get the world
        self.world = self.client.get_world()
        # Set the world in synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 0.05 seconds (20 FPS)
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.dummy_tick = True


        self.camera_image_queue = deque()
        self.ego_vehicle, self.other_vehicles = self.get_ego_and_other_vehicles()
        self.active_scenario_vehicle = self.other_vehicles[0]
        self.agent = StandStillAgent(self.ego_vehicle)
        # destination = carla.Location(x=-77.6, y=60, z=0)
        # self.agent.set_destination(destination)

        # spawn the sensor and attach to vehicle.
        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1216')
        cam_bp.set_attribute('image_size_y', '1216')
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))
        # self.sensor = self.world.spawn_actor(cam_bp, sensor_transform, attach_to=self.ego_vehicle)
        self.sensor = self.world.get_actors().filter("sensor.camera.rgb")[0]
        self.sensor.listen(lambda image: self.process_img(image))


        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.dummy_tick = True
            self.world.tick()
        else:
            self.world.wait_for_tick()

        self.ego_traffic_light, self.op_traffic_light = self.get_traffic_lights()
        affected_waypoints = self.op_traffic_light.get_affected_lane_waypoints()
        num_wp = 0
        self.junction_loc = affected_waypoints[0].transform.location
        # for wp in affected_waypoints:
        #     self.junction_loc += wp.transform.location
        #     num_wp += 1
        # self.junction_loc /= num_wp
        
        # set the spectator to ego vehicle
        spectator = self.world.get_spectator()
        transform_vehicle = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        time.sleep(1)


        # initialize object detector
        self.obj_detector = CarlaObjectDetector()
        self.junction_annotator = JunctionAnnotator(self.world, ego_vehicle=self.ego_vehicle, camera_bp=cam_bp, camera=self.sensor)
        self.state_extractor = StateExtractor()

        self.count = 0
    
    
    
    def process_img(self, image):
        # Convert the image from CARLA format to an OpenCV image (RGB)
        if self.dummy_tick:
            self.dummy_tick = False
            return
        img0 = np.array(image.raw_data).reshape((image.height, image.width, 4))
        img0 = img0[:, :, :3]
        self.camera_image_queue.append(np.array(img0))

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
                if vehicle.get_transform().location.z >= 0:
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
        at_junction = False
        other_vehicle_loc = self.active_scenario_vehicle.get_location()
        other_vehicle_wp = self.map.get_waypoint(other_vehicle_loc)
        if other_vehicle_wp.is_junction:
            at_junction = True
        distance = other_vehicle_loc.distance(self.junction_loc)
        return at_junction, distance
    

    def start_data_collection(self):
        # Setting the light to green is the trigger for scenario. See DatasetGenSurrogateModel scenario
        self.ego_traffic_light.set_state(carla.TrafficLightState.Green)
        print("Triggered the light")
        # self.dummy_tick = True
        # self.world.tick()
        # self.world.tick()
        # while self.camera_image_queue.empty():
        #     self.world.tick()
        #     time.sleep(0.1)
        #     print("empty queue. ticking world")

        count = 0
        while True:
            # while self.camera_image_queue.qsize() > 1:
            #     self.camera_image_queue.get()
            # print(f"{self.active_scenario_vehicle.get_transform().location.x}, {self.active_scenario_vehicle.get_transform().location.y}, {self.active_scenario_vehicle.get_transform().location.z}")
            if self.active_scenario_vehicle.get_transform().location.z < -4:
                if not self.set_active_vehicle():
                    print("no active non scenario vehicle for 100 ticks")
                    break
            
            if self.agent.done():
                print("The target has been reached, stopping the simulation")
                break
            
            at_junction = False
            if not len(self.camera_image_queue) == 0:
                camera_image = self.camera_image_queue.pop()
                detections, annotated_camera_image = self.obj_detector.detect(camera_image)
                # junction_bbox = self.junction_annotator.annotate()
                # print(junction_bbox)
                at_junction, distance_to_junction = self.get_state()
                # print(f"{at_junction}, {distance_to_junction}")
                text = f"At Junction: {at_junction}, Distance to Junction: {distance_to_junction}"
                cv2.putText(annotated_camera_image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                active_loc = self.active_scenario_vehicle.get_location()
                text2 = f"loc: {active_loc.x}, {active_loc.y}, {active_loc.z}"
                cv2.putText(annotated_camera_image, text2, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.imshow('Annotated Images',annotated_camera_image)
                # if cv2.waitKey(1) == ord('q'):
                #     break
                # cv2.imshow("annotated images", annotated_image_queue.pop())
                cv2.imwrite(f"./recording/{count}.png", annotated_camera_image)
                count += 1
            else:
                print("empty queue")
            self.ego_vehicle.apply_control(self.agent.run_step())
            self.world.tick()
            # time.sleep(0.1)
        self.sensor.destroy()
    
    def cleanup(self):
        CarlaDataProvider.cleanup()
        self.sensor.destroy()

if __name__ == "__main__":
    datasetgenerator = DatasetGenSurrogateModel()
    try:
        datasetgenerator.start_data_collection()
    except KeyboardInterrupt:
        datasetgenerator.cleanup()
