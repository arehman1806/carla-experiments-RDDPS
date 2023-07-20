import carla
import random
import py_trees
import time

from basic_scenario import BasicScenario

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      StopVehicle,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, StandStill
from srunner.tools.scenario_helper import choose_at_junction
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance, generate_target_waypoint_list
from srunner.tools.route_manipulation import interpolate_trajectory


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

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        """
        Initialize all parameters required for NewScenario
        """

        # Call constructor of BasicScenario
        super(DatasetGenSurrogateModel, self).__init__(
          "DatasetGenSurogateModel",
          ego_vehicles,
          config,
          world,
          debug_mode,
          criteria_enable=criteria_enable)
        
        self._ego_vehicle = ego_vehicles[0]
        self._ego_vehicle.set_simulate_physics(enabled=False)
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self._ego_vehicle, False)
        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_red_time(1e9)
        ego_stop_pts = self._traffic_light.get_stop_waypoints()
        print(f"length:{len(ego_stop_pts)}\n\n\n")
        self._ego_vehicle.set_transform(ego_stop_pts[0].transform)
        other_tls = CarlaDataProvider.annotate_trafficlight_in_group(self._traffic_light)
        opp = other_tls["opposite"][0]
        opp_stop_points = opp.get_stop_waypoints()
        print(f"length:{len(opp_stop_points)}\n\n\n")
        self.other_actors[0].set_transform(opp_stop_points[0].transform)
        self.simulate_random_wp_spawn(turn=-1)
        self.simulate_random_wp_spawn()
        self.other_actors[0].set_transform(opp_stop_points[1].transform)
        self.simulate_random_wp_spawn()
        self.simulate_random_wp_spawn(turn=1)

        
    def _initialize_actors(self, config):

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            if vehicle is None:
                raise Exception(f"Error adding other actor: {actor}")
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # get the minimum x, maximum x, min y, and max y for spawning other vehicle



    def _create_behavior(self):
        """
        Setup the behavior for NewScenario
        """

    def _create_test_criteria(self):
        """
        Setup the evaluation criteria for NewScenario
        """
        pass

    def get_wps_from_stop_pt(self):
        ego_loc = self.other_actors[0].get_transform().location
        waypoint = CarlaDataProvider.get_map().get_waypoint(ego_loc)

        list_of_waypoints = []
        for i in range(200):
            waypoint = waypoint.previous(1)[0]
            list_of_waypoints.append(waypoint)
        return list_of_waypoints
    
    def simulate_random_wp_spawn(self, turn=0):
        time.sleep(5)
        ego_loc = self.other_actors[0].get_transform().location
        waypoint = CarlaDataProvider.get_map().get_waypoint(ego_loc)
        plan, _ = generate_target_waypoint_list(waypoint, turn)
        
        list_of_wps = []
        for i in range(len(plan)):
            list_of_wps.append(plan[i][0])
        
        _, route = interpolate_trajectory(CarlaDataProvider._world, list_of_wps, 1)

        for i in range(len(route)):
            CarlaDataProvider._world.debug.draw_string(route[i][0].location, 'O', draw_shadow=False,
                                       color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                       persistent_lines=True)
        prev_wps = self.get_wps_from_stop_pt()
        for wp in prev_wps:
            CarlaDataProvider._world.debug.draw_string(wp.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

            
        print(f"len of points: {len(route)}")

        # for i in range(len(route)):
        #     self.other_actors[0].set_transform(route[i][0])
        #     time.sleep(1)
