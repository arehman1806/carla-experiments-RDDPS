from gymnasium.envs.registration import register

register(
     id="perception_mdp/simple-v0",
     entry_point="gymnasium_envs.envs:SimplePerceptionMDP"
)