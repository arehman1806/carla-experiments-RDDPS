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


        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        # Wait for the world to be ready
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()


        self.camera_image_queue = Queue()
        self.ego_vehicle = self.get_ego_vehicle()
        self.agent = StandStillAgent(self.ego_vehicle)
        # destination = carla.Location(x=-77.6, y=60, z=0)
        # self.agent.set_destination(destination)

        self.ego_traffic_light = self.ego_vehicle.get_traffic_light()
        self.op_traffic_light = CarlaDataProvider.annotate_trafficlight_in_group(self.ego_traffic_light)["opposite"][0]

        # set the spectator to ego vehicle
        spectator = self.world.get_spectator()
        transform_vehicle = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
        time.sleep(1)

        # spawn the sensor and attach to vehicle.
        blueprint_library = self.world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1216')
        cam_bp.set_attribute('image_size_y', '1216')
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))
        sensor = self.world.spawn_actor(cam_bp, sensor_transform, attach_to=self.ego_vehicle)
        sensor.listen(lambda image: self.process_img(image))

        # initialize object detector
        self.obj_detector = CarlaObjectDetector()
        self.junction_annotator = JunctionAnnotator(self.world, ego_vehicle=self.ego_vehicle, camera_bp=cam_bp, camera=sensor)
        self.state_extractor = StateExtractor()
    
    
    
    def process_img(self, image):
        # Convert the image from CARLA format to an OpenCV image (RGB)
        img0 = np.array(image.raw_data).reshape((image.height, image.width, 4))
        img0 = img0[:, :, :3]
        self.camera_image_queue.put(np.array(img0))

    
    def get_ego_vehicle(self) -> carla.Vehicle:
        ego_vehicle = None

        # Get the ego vehicle
        while ego_vehicle is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == 'hero':
                    print("Ego vehicle found")
                    ego_vehicle = vehicle
                    break
        return ego_vehicle

    def annotatate_with_junction(self, camera_image, bbox):
        for i in range(len(bbox) - 1):
            p1 = bbox[i]
            p2 = bbox[i+1]
            cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
        p1 = bbox[0]
        p2 = bbox[len(bbox) - 1]
        cv2.line(camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
        return camera_image



    def start_data_collection(self):
        count = 0

        self.ego_traffic_light.set_state(carla.TrafficLightState.Green)
        print("Triggered the light")

        while True:
            if self.agent.done():
                print("The target has been reached, stopping the simulation")
                break
            at_junction = False
            if not self.camera_image_queue.empty():
                camera_image = self.camera_image_queue.get()
                detections, annotated_camera_image = self.obj_detector.detect(camera_image)
                junction_bbox = self.junction_annotator.annotate()
                # print(junction_bbox)
                if junction_bbox is not None:
                    at_junction, distance_to_junction = self.state_extractor.check_overlap(detections, junction_bbox)
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
            self.ego_vehicle.apply_control(self.agent.run_step(non_ego_at_junction=at_junction))
            self.world.tick()
        self.sensor.destroy()

if __name__ == "__main__":
    datasetgenerator = DatasetGenSurrogateModel()
    datasetgenerator.start_data_collection()