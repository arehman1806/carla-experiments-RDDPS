import carla
import random
import py_trees
import time

from basic_scenario import BasicScenario

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      StopVehicle,
                                                                      ActorDestroy,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      Idle,
                                                                      ChangeWeather)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
import srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions as conditions
from srunner.tools.scenario_helper import choose_at_junction
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance, generate_target_waypoint_list
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.scenariomanager.weather_sim import Weather, WeatherBehavior


class DatasetGenSurrogateModel(BasicScenario):
    """
    Some documentation on NewScenario
    :param world is the CARLA world
    :param ego_vehicles is a list of ego vehicles for this scenario
    :param config is the scenario configuration (ScenarioConfiguration)
    :param randomize can be used to select parameters randomly (optional, default=False)
    :param debug_mode can be used to provide more comprehensive console output (optional, default=False)
    :param criteria_enable can be used to disable/enable scenario evaluation based on test criteria (optional, default=True)
    :param timeout is the overall scenario timeout (optional, default=60 seconds)
    """

    # some ego vehicle parameters
    # some parameters for the other vehicles

    timeout = 12000000

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=6000000):
        """
        Initialize all parameters required for NewScenario
        """
        
        self._world = world
        
        self._ego_vehicle = ego_vehicles[0]
        

        # Call constructor of BasicScenario
        super(DatasetGenSurrogateModel, self).__init__(
          "DatasetGenSurogateModel",
          ego_vehicles,
          config,
          world,
          debug_mode,
          criteria_enable=criteria_enable)
        

        
    def _initialize_actors(self, config):
        self._ego_vehicle.set_simulate_physics(enabled=False)
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self._ego_vehicle, False)
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_green_time(1e9)
        ego_stop_pts = self._traffic_light.get_stop_waypoints()
        ego_new_tranform = carla.Transform(
            carla.Location(ego_stop_pts[0].transform.location.x,
                           ego_stop_pts[0].transform.location.y,
                           ego_stop_pts[0].transform.location.z+1),
            ego_stop_pts[0].transform.rotation
        )
        self._ego_vehicle.set_transform(ego_new_tranform)
        self._ego_vehicle.set_simulate_physics(enabled=True)
        other_tls = CarlaDataProvider.annotate_trafficlight_in_group(self._traffic_light)
        opp = other_tls["opposite"][0]
        self._opp_stop_points = opp.get_stop_waypoints()
        
        self._active_other_vehicle = None

        self._other_actor_reserve_locations = {}

        z = -5

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            if vehicle is None:
                raise Exception(f"Error adding other actor: {actor}")
            vehicle.rolename = "idle"
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)
            transform = carla.Transform(
                carla.Location(vehicle.get_transform().location.x,
                               vehicle.get_transform().location.y,
                               z),
                vehicle.get_transform().rotation
            )
            vehicle.set_transform(transform)
            
            self._other_actor_reserve_locations[vehicle.id] = transform
            z -= 5
        
        # blueprint_library = self._world.get_blueprint_library()
        # cam_bp = blueprint_library.find('sensor.camera.rgb')
        # cam_bp.set_attribute('image_size_x', '1216')
        # cam_bp.set_attribute('image_size_y', '1216')
        # sensor_transform = carla.Transform(carla.Location(x=2.5, z=2))
        # sensor = self._world.spawn_actor(cam_bp, sensor_transform, attach_to=self._ego_vehicle)
        # CarlaDataProvider.register_actor(sensor)
        
        
        self._other_spawn_points = self.generate_other_spawn_points(debug=True)


    def _setup_scenario_trigger(self, config):
        return conditions.WaitForTrafficLightState(self._traffic_light, carla.TrafficLightState.Green)
    

    def _create_behavior(self):
        """
        Ideally this should spawn one of the other actors to a possible spawn location, take a picture, and record 
        the necassary data
        """

        sequence = py_trees.composites.Sequence("Sequence Behavior")

        # Behavior tree
        weather = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        carla_weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=90.0,
            fog_density=0
        )
        weather_updater_bb = Weather(carla_weather=carla_weather)
        
        weather.add_child(WeatherBehavior())
        weather.add_child(ChangeWeather(weather_updater_bb))

        
        
        sequence.add_child(weather)
        for actor in self.other_actors:
            actor.rolename = "abc"
            for spawn_point in self._other_spawn_points:
                sequence.add_child(ActorTransformSetter(actor, spawn_point, physics=False))
            actor.attributes["role_name"] = "idle"
            sequence.add_child(ActorTransformSetter(actor, self._other_actor_reserve_locations[actor.id], physics=False))

        # sequence.add_child(root)
        
        for actor in self.other_actors:
            sequence.add_child(ActorDestroy(actor))

        return sequence


    def _create_test_criteria(self):
        """
        Setup the evaluation criteria for NewScenario
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def get_previous_wps(self, until=100):
        ego_loc = self.other_actors[0].get_transform().location
        waypoint = CarlaDataProvider.get_map().get_waypoint(ego_loc)
        list_of_waypoints = waypoint.previous_until_lane_start(1)[:-1]
        # list_of_waypoints = []
        # for i in range(until):
        #     waypoint = waypoint.previous(1)[0]
        #     list_of_waypoints.append(waypoint)

        return list_of_waypoints
    
    def generate_other_spawn_points(self, debug=False):
        self._active_other_vehicle = self.other_actors[0]
        possible_spawn_points = []
        self.other_actors[0].set_transform(self._opp_stop_points[0].transform)
        time.sleep(0.1)
        possible_spawn_points.extend(self.generate_spawn_points_for_lane_and_turn(turn=-1, debug=debug))
        possible_spawn_points.extend(self.generate_spawn_points_for_lane_and_turn(debug=debug))
        self.other_actors[0].set_transform(self._opp_stop_points[1].transform)
        time.sleep(0.1)
        possible_spawn_points.extend(self.generate_spawn_points_for_lane_and_turn(debug=debug))
        possible_spawn_points.extend(self.generate_spawn_points_for_lane_and_turn(turn=1, debug=debug))

        self._active_other_vehicle.set_transform(self._other_actor_reserve_locations.get(self._active_other_vehicle.id))

        return possible_spawn_points

    def generate_spawn_points_for_lane_and_turn(self, turn=0, debug=False):
        other_loc = self.other_actors[0].get_transform().location
        waypoint = CarlaDataProvider.get_map().get_waypoint(other_loc)

        plan, _ = generate_target_waypoint_list(waypoint, turn)
        _, route = interpolate_trajectory(CarlaDataProvider._world, [point[0] for point in plan], 1)
        possible_spawn_transforms = [point[0] for point in route]
        prev_wps = self.get_previous_wps()
        possible_spawn_transforms.extend([point.transform for point in prev_wps])

        if debug:
            for transform in possible_spawn_transforms:
                CarlaDataProvider._world.debug.draw_string(transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1e9,
                                        persistent_lines=True)

        return possible_spawn_transforms