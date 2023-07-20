import carla
import os
from random import randint, seed

def save_image(image):
    try:
        image.save_to_disk('./instance_segmentation_demo/%06d.png' % image.frame)
    except Exception as e:
        print(f'Error saving image: {e}')

def spawn_vehicle(world, blueprint_library, vehicle_model, transform):
    blueprint = blueprint_library.filter(vehicle_model)[0]
    return world.try_spawn_actor(blueprint, transform)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    spawn_points = world.get_map().get_spawn_points()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    tm_port = traffic_manager.get_port()

    seed(0) # For reproducibility
    num_vehicles = 100
    for _ in range(num_vehicles):
        spawn_transform = spawn_points[randint(0, len(spawn_points)-1)]
        vehicle = spawn_vehicle(world, blueprint_library, 'vehicle.*', spawn_transform)
        if vehicle is not None:
            vehicle.set_autopilot(True, tm_port)

    # Ego vehicle with semantic segmentation camera
    ego_vehicle_transform = spawn_points[randint(0, len(spawn_points)-1)]
    ego_vehicle = spawn_vehicle(world, blueprint_library, 'vehicle.*', ego_vehicle_transform)
    if ego_vehicle is not None:
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        camera.listen(lambda image: save_image(image))
    else:
        pass

    while True:
        world.tick()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExiting...')
