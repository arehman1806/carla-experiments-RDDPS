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
from srunner.tools.route_manipulation import interpolate_trajectory, interpolate_wp_trajectory
from srunner.scenariomanager.weather_sim import Weather, WeatherBehavior
from agents.navigation.local_planner import RoadOption


class DatasetGenPerceptionModel(BasicScenario):
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
        super(DatasetGenPerceptionModel, self).__init__(
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
        right_tl = other_tls["right"][0]
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

        
        
        self._other_routes = self.generate_actor_start_end_points()
        self._ego_points = self.generate_ego_points(self._traffic_light, right_tl)


    def _setup_scenario_trigger(self, config):
        return conditions.WaitForTrafficLightState(self._traffic_light, carla.TrafficLightState.Green)
        return conditions.StandStill(self._ego_vehicle, "lora mera", duration=0)
    

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

            for route in self._other_routes:

                for ego_spawn_point in self._ego_points:
                    ego_transform_setter = ActorTransformSetter(self._ego_vehicle, ego_spawn_point.transform, False)
                    actor_transform_setter = ActorTransformSetter(actor, route[0].transform)
                    wpf = WaypointFollower(actor, 25, [(route[1], RoadOption.LANEFOLLOW)])
                    sequence.add_child(ego_transform_setter)
                    sequence.add_child(actor_transform_setter)
                    sequence.add_child(wpf)
            rp_actor = ActorTransformSetter(actor, self._other_actor_reserve_locations[actor.id], physics=False)
            sequence.add_child(rp_actor)

        return sequence


    def _create_test_criteria(self):
        """
        Setup the evaluation criteria for NewScenario
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria
    
    def generate_ego_points(self, current_tl, right_tl):
        ego_points = []
        current_tl_stop_point = current_tl.get_stop_waypoints()[0]
        intersection = self.get_next_intersection(current_tl_stop_point)
        entry_exit_pts = intersection.get_waypoints(carla.LaneType.Any)
        
        right_tl_stop_points = right_tl.get_stop_waypoints()
        pt_i = 0
        if len(right_tl_stop_points) == 2:
            pt_i = 1
        right_tl_stop_point = right_tl_stop_points[pt_i]
        plan, _ = generate_target_waypoint_list(right_tl_stop_point, 0)
        left_tl_stop_point = plan[-1][0]
        CarlaDataProvider._world.debug.draw_string(current_tl_stop_point.transform.location, 'turn_left_start', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1e9,
                                        persistent_lines=True)
        CarlaDataProvider._world.debug.draw_string(left_tl_stop_point.transform.location, 'turn_left_stop', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=1e9,
                                        persistent_lines=True)
        
        route = interpolate_wp_trajectory(CarlaDataProvider._world, [current_tl_stop_point, left_tl_stop_point], 4)[:5]
        ego_points.extend([point[0] for point in route])
        
        previous_pts = current_tl_stop_point.previous(20)
        previous_pts.extend([current_tl_stop_point])
        route = interpolate_wp_trajectory(CarlaDataProvider._world, previous_pts,4 )
        ego_points.extend([point[0] for point in route])
        for point in ego_points:
            CarlaDataProvider._world.debug.draw_string(point.transform.location, 'tlj', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=255), life_time=1e9,
                                        persistent_lines=True)
        
        return ego_points

        # intersection = self.get_next_intersection(current_tl_stop_point)
        # ent_exit_pts = intersection.get_waypoints(carla.LaneType.Driving)
        # junction_pts = []
        # for sublish in ent_exit_pts:
        #     junction_pts.extend(sublish)
        # yaw_target = current_tl_stop_point.transform.rotation.yaw
        # loc_target = current_tl_stop_point.transform.location
        # stop_point = None
        # min_distance = 1e9
        # for point in junction_pts:
        #     yaw = point.transform.rotation.yaw
        #     loc = point.transform.location
        #     dist = loc.distance(loc_target)
            
        #     if dist < min_distance:
        #         stop_point = point
        #         min_distance = dist
        # CarlaDataProvider._world.debug.draw_string(stop_point.transform.location, 'main_stop_point', draw_shadow=False,
        #                                 color=carla.Color(r=255, g=0, b=255), life_time=1e9,
        #                                 persistent_lines=True)
        # for point in junction_pts:
        #     CarlaDataProvider._world.debug.draw_string(point.transform.location, 'jp', draw_shadow=False,
        #                                 color=carla.Color(r=162, g=25, b=255), life_time=1e9,
        #                                 persistent_lines=True)
    
    def get_next_intersection(self, waypoint):
        list_of_waypoints = []
        while waypoint and not waypoint.is_intersection:
            list_of_waypoints.append(waypoint)
            waypoint = waypoint.next(2.0)[0]

        # If the list is empty, the actor is in an intersection
        if not list_of_waypoints:
            return None
        else:
            return waypoint.get_junction()

    def get_start_end_wp_traj(self, waypoint):
        list_of_waypoints = waypoint.previous_until_lane_start(1)[:-1]
        first_wp = list_of_waypoints[-1]
        plan, _ = generate_target_waypoint_list(waypoint, 0)
        end_wp = [point[0] for point in plan][-1]
        return [first_wp, end_wp]
    
    def generate_actor_start_end_points(self):
        routes = []
        for stop_point in self._opp_stop_points:
            routes.append(self.get_start_end_wp_traj(stop_point))
        
        for route in routes:
            start, end  = route
            CarlaDataProvider._world.debug.draw_string(start.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=1e9,
                                        persistent_lines=True)
            CarlaDataProvider._world.debug.draw_string(end.transform.location, 'x', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1e9,
                                        persistent_lines=True)
        return routes

    
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
        if turn == 0:
            until = -6
        else:
            until = -6

        plan, _ = generate_target_waypoint_list(waypoint, turn)
        _, route = interpolate_trajectory(CarlaDataProvider._world, [point[0] for point in plan[:until]], 1)
        possible_spawn_transforms = [point[0] for point in route]
        prev_wps = self.get_previous_wps()
        possible_spawn_transforms.extend([point.transform for point in prev_wps])

        if False:
            for transform in possible_spawn_transforms:
                CarlaDataProvider._world.debug.draw_string(transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=1e9,
                                        persistent_lines=True)

        return possible_spawn_transforms