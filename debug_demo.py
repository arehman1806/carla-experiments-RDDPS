import carla
import os
from random import randint, seed


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()


spawn_points = world.get_map().get_spawn_points()

# Route 1
spawn_point_1 =  spawn_points[32]
# Create route 1 from the chosen spawn points
route_1_indices = [129, 28, 124, 33, 97, 119, 58, 154, 147]
route_1 = []
for ind in route_1_indices:
    route_1.append(spawn_points[ind].location)

# Route 2
spawn_point_2 =  spawn_points[149]
# Create route 2 from the chosen spawn points
route_2_indices = [21, 76, 38, 34, 90, 3]
route_2 = []
for ind in route_2_indices:
    route_2.append(spawn_points[ind].location)

# Now let's print them in the map so we can see our routes
world.debug.draw_string(spawn_point_1.location, 'Spawn point 1', life_time=30, color=carla.Color(255,0,0))
world.debug.draw_string(spawn_point_2.location, 'Spawn point 2', life_time=30, color=carla.Color(0,0,255))

for ind in route_1_indices:
    spawn_points[ind].location
    world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(255,0,0))

for ind in route_2_indices:
    spawn_points[ind].location
    world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(0,0,255))

while True:
    world.tick()