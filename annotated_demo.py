import carla
import cv2
import numpy as np
import time
from math import tan, radians
import queue


class IntersectionAnnotator:

    def __init__(self) -> None:

        # Connect to the server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # seconds

        # Get the world
        self.world = self.client.get_world()

        # Set the world in synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.05  # 0.05 seconds (20 FPS)
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()

        # Set up the vehicle blueprint and select Tesla model 3
        vehicle_bp = blueprint_library.filter('model3')[0]

        # Specify the start location and orientation for the vehicle
        spawn_point = carla.Transform(carla.Location(x=106, y=58.9, z=2), carla.Rotation(yaw=-90))

        # Spawn the vehicle
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # get the camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=3))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.camera.listen(lambda image: self.process_image(image))

        # Get the world to camera matrix
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = self.build_projection_matrix(image_w, image_h, fov)

        self.image_queue = queue.Queue()

        spectator = self.world.get_spectator()
        transform_vehicle = self.ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform_vehicle.location + carla.Location(z=10), carla.Rotation(pitch=-90)))
        time.sleep(1)

    
    def run_main_loop(self):
        while True:
            img = self.image_queue.get()

            # Get the camera matrix 
            world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

            bb = self.get_bounding_box()

            for i in range(len(bb) - 1):
                p1 = self.get_image_point(bb[i], self.K, world_2_camera)
                p2 = self.get_image_point(bb[i+1],  self.K, world_2_camera)
                print(f"{p1}, {p2}")
                # Draw the edges into the camera output
                cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
            p1 = self.get_image_point(bb[0], self.K, world_2_camera)
            p2 = self.get_image_point(bb[len(bb) - 1],  self.K, world_2_camera)
            print(f"{p1}, {p2}")
            # Draw the edges into the camera output
            cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
            
            
            cv2.imshow('ImageWindowName',img)
            if cv2.waitKey(1) == ord('q'):
                break



            self.world.tick()
        self.world.destroy_all_actors()

    def build_projection_matrix(self,w, h, fov):
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


    def draw_bounding_box(self, image, bbox):
        """
        Draw a bounding box on the image.
        bbox is a list of 2D points representing the bounding box.
        """
        color = (255, 0, 0)
        thickness = 2

        points = [tuple(map(int, point)) for point in bbox]
        print(points)

        # Add a check for number of points
        if len(points) < 4:
            print("Fewer than 4 points in the bounding box, skipping drawing. ", len(points))
            return

        for i in range(4):
            cv2.line(image, points[i], points[(i + 1) % 4], color, thickness)
        cv2.imwrite("asdas.png", image)

    def project_to_image(self, points_3d, intrinsic):
        """
        Project 3D points to 2D using camera intrinsic parameters.
        points_3d is a list of carla.Location objects.
        intrinsic is the camera intrinsic matrix.
        """
        points_2d = []
        for point_3d in points_3d:
            x, y, z = point_3d.x, point_3d.y, point_3d.z
            point_3d_vector = np.array([x, y, z]).reshape(3, 1)
            
            point_2d_homogeneous = np.matmul(intrinsic, point_3d_vector)

            # Check if z-coordinate of the point in camera frame is zero or very close to zero
            if np.isclose(point_2d_homogeneous[2][0], 0, atol=1e-6):
                continue  # Skip this point, it cannot be correctly projected to 2D

            point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
            points_2d.append(point_2d.flatten())
        return points_2d

    def get_junction_waypoint(self):
        # get the waypoint ahead of the vehicle
        waypoint_ahead = self.map.get_waypoint(self.ego_vehicle.get_location())

        # get the next waypoints
        next_waypoints = waypoint_ahead.next(50)

        # next_waypoint = waypoint_ahead.next_until_lane_end()

        # find the first junction waypoint among the next waypoints
        junction_waypoint = None
        for wp in next_waypoints:
            if wp.is_junction:
                junction_waypoint = wp
                break
        # print(junction_waypoint)
        return junction_waypoint

    def get_bounding_box(self):
        # get the junction waypoint
        junction_waypoint = self.get_junction_waypoint()

        if junction_waypoint is None:
            return None

        junction = junction_waypoint.get_junction()

        intersection_waypoints_list = junction.get_waypoints(carla.LaneType.Any)

        intersection_waypoint = []

        for sublist in intersection_waypoints_list:
            intersection_waypoint.extend(sublist)


        # create a list to store the bounding box vertices
        bounding_box_3d = []

        # find the maximum and minimum x and y coordinates to create a bounding box
        min_x = min([wp.transform.location.x for wp in intersection_waypoint])
        max_x = max([wp.transform.location.x for wp in intersection_waypoint])
        min_y = min([wp.transform.location.y for wp in intersection_waypoint])
        max_y = max([wp.transform.location.y for wp in intersection_waypoint])

        # print(f"{min_x} {min_y}, {max_x}, {max_y}\n\n\n\n\n\n")

        # z = junction_waypoint.transform.location.z  # get the Z-coordinate of the junction waypoint
        z = 0

        # create the bounding box vertices
        bounding_box_3d.append(carla.Location(x=min_x, y=min_y, z=z))  # bottom-left corner
        bounding_box_3d.append(carla.Location(x=min_x, y=max_y, z=z))  # top-left corner
        bounding_box_3d.append(carla.Location(x=max_x, y=max_y, z=z))  # top-right corner
        bounding_box_3d.append(carla.Location(x=max_x, y=min_y, z=z))  # bottom-right corner

        return bounding_box_3d

    def process_image(self, image):
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 
        self.image_queue.put(np.array(img))






if __name__ == "__main__":
    ia = IntersectionAnnotator()
    ia.run_main_loop()



