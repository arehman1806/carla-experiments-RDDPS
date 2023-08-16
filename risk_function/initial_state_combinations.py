def compute_stopping_distance(v, a_min):
    """Compute the stopping distance for given velocity and deceleration."""
    return v**2 / (2 * abs(a_min))

def max_speed_distance_combinations(a_min, max_speed, step=0.1):
    """Compute the combinations of maximum possible initial speed and distance."""
    combinations = []
    
    v = 0
    while v <= max_speed:
        d = compute_stopping_distance(v, a_min)
        combinations.append((v, d))
        v += step
        
    return combinations

# Given parameters
a_min = -4.6
max_speed = 20  # Or any other desired maximum speed to check until

combinations = max_speed_distance_combinations(a_min, max_speed)

# Display the results
for v, d in combinations:
    print(f"Initial Speed: {v:.2f} m/s | Required Distance: {d:.2f} m")

