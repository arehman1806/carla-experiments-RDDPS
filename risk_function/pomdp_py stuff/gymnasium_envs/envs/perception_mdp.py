import numpy as np

import gymnasium as gym
from gymnasium import spaces


class SimplePerceptionMDP(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, state_space_limits={}, 
                 safety_threshold=15, controller_stop_threshold=55, ego_distance_max=30,
                 speed_limit=45, dt=0.01):

        self._dt= dt
        self._speed_limit = speed_limit
        self._safety_threshold = safety_threshold
        self._ego_distance_max = ego_distance_max
        self._controller_stop_threshold = controller_stop_threshold

        self.observation_space = self._get_observation_space()

        # We have 2 actions, corresponding to "non-detected" and "detected"
        self.action_space = spaces.Discrete(2)
        self._state = None

        self.division_factor = 100

    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        actor_location = self.np_random.integers(self._safety_threshold + 2, 10000)
        ego_distance_travelled = 0
        self._state = np.array([actor_location, ego_distance_travelled, 0], dtype=np.uint16)
        
        info = {"actor_location": actor_location/10}

        return self._state, info
    
    def step(self, detected):
        terminated = False
        cost = 0

        controller_action = self._naive_controller_step(detected)
        # print(f"{self._state}, {controller_action}")
        state = self._get_scaled_state()
        distance_actor = state[0] - self._speed_limit * self._dt
        ego_move_distance = 0
        if controller_action == 1:
            ego_move_distance = state[1] + self._speed_limit * self._dt*1
        collision_course = 0
        if distance_actor < self._safety_threshold:
            collision_course = 1
            terminated = True
            cost = 10
        if ego_move_distance >= self._ego_distance_max:
            terminated = True
        
        self._state = np.array([distance_actor*self.division_factor, ego_move_distance*self.division_factor, collision_course], dtype=np.uint16)
        info = {}


        return self._state, cost, terminated, False, info


    def _get_observation_space(self):
        min_obs = np.array([0, 0, 0])
        max_obs = np.array([1000,1000,1])
        return spaces.Box(low=min_obs, high=max_obs, dtype=np.uint16)
    
    def _naive_controller_step(self, car_detected):
        """
        Returns:
            0: stop
            1: move
        """
        car_stop = car_detected == 1 and self._state[0] < self._controller_stop_threshold
        # print(f"non-ego at {self._state[0]}. car detected: {car_detected}. car should stop: {car_stop}")
        if car_detected == 1 and self._state[0] < self._controller_stop_threshold:
            return 0
        else:
            return 1
        
    def _get_scaled_state(self):
        return [self._state[0]/self.division_factor, self._state[1] / self.division_factor, self._state[2]]