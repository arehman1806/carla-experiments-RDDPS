import carla
import cv2
import numpy as np
import time
from math import tan, radians
import queue


class JunctionAnnotator:

    def __init__(self, world, ego_vehicle, camera_bp, camera) -> None:

        # Get the world
        self.world = world

        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()

    
        # Spawn the vehicle
        self.ego_vehicle = ego_vehicle

        # get the camera sensor
        self.camera = camera

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = self.build_projection_matrix(image_w, image_h, fov)
    
    def annotate(self):

        # Get the camera matrix 
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())

        bb = self.get_bounding_box()

        if bb is None:
            return None

        points = []
        for i in range(len(bb)):
            points.append(self.get_image_point(bb[i], self.K, world_2_camera))
        
        return points


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

    def get_junction_waypoint(self):
        # get the waypoint ahead of the vehicle
        waypoint_ahead = self.map.get_waypoint(self.ego_vehicle.get_location())
        # if waypoint_ahead.is_junction:
        #     return waypoint_ahead

        # get the next waypoints
        next_waypoints = waypoint_ahead.next(10)

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

        z = junction_waypoint.transform.location.z  # get the Z-coordinate of the junction waypoint
        # z = 2

        # create the bounding box vertices
        # bounding_box_3d.append(carla.Location(x=min_x, y=min_y, z=z))  # bottom-left corner
        # bounding_box_3d.append(carla.Location(x=min_x, y=max_y, z=z))  # top-left corner
        bounding_box_3d.append(carla.Location(x=max_x, y=max_y, z=z))  
        bounding_box_3d.append(carla.Location(x=max_x, y=min_y, z=z)) 
        # bounding_box_3d.append(carla.Location(x=max_x-20, y=min_y, z=z)) 
        # bounding_box_3d.append(carla.Location(x=max_x-20, y=max_y, z=z))  

        return bounding_box_3d




