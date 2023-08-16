def calculate_required_acceleration(v, d):
    if d == 0:
        return float('-inf')  # Immediate stop, though it means a collision
    return -v**2 / (2*d)  # Negative because we want to decelerate

def update_state(v, a, dt, d):
    v_next = v + a * dt
    d_next = d - v * dt - 0.5 * a * dt**2
    return v_next, d_next

def simulate(initial_velocity, initial_distance, a_max, a_min, dt, max_timesteps=1000):
    v = initial_velocity
    d = initial_distance
    
    for t in range(max_timesteps):
        a_required = calculate_required_acceleration(v, d)
        
        # Check constraints and adjust acceleration
        if a_required > a_max:
            a = a_max
        elif a_required < a_min:
            a = a_min
        else:
            a = a_required
        
        # Update state
        v, d = update_state(v, a, dt, d)
        
        print(f"Time: {t*dt:.2f}s | Velocity: {v:.2f}m/s | Distance: {d:.2f}m | Acceleration: {a:.2f}m/s^2")
        
        # Check stopping condition
        if v <= 0 and d >= 0:
            print("Ego vehicle stopped safely!")
            return
        elif d < 0:
            print("Collision occurred!")
            return

    print("Simulation ended without stopping or collision.")

# Test
simulate(20, 44, 2, -4.6, 0.1)
