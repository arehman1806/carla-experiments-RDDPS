using Flux
using Flux: crossentropy, @epochs, params
using DataFrames
using CSV
using Random
Random.seed!(0) # Set random seed

include("./surrogate_nn_architecture.jl")

# Load the dataset using DataFrames and CSV.jl
df = CSV.File("./yolov7_carla_object_detection/surrogate_model_dataset.csv") |> DataFrame

# Extract features and target
features = Matrix{Float32}(df[:, [:ego_distance, :ego_velocity, :actor_distance]])
target = reshape(Vector{Float32}(df[:, :detected]), :, 1)

# Create a SurrogateNN instance
net = SurrogateNN(3, 100)

# Loss function
loss(x, y) = Flux.Losses.binarycrossentropy(net(x'), y')


# Define an optimizer
opt = Descent(0.01)

# Training loop
epochs = 10000

@epochs epochs for epoch in 1:epochs
    Flux.train!(loss, params(net), [(features, target)], opt)
    if epoch % 1000 == 0
        println("Epoch: $epoch, Loss: $(loss(features, target))")
    end
end


bson("./risk_function/surrogate_nn_model.bson", model=model)

using Plots

# Predict with the model
# Define the range and resolution of your grid
num_points = 100  # the number of points along each axis
ego_distance_range = (minimum(df.ego_distance), maximum(df.ego_distance))
actor_distance_range = (minimum(df.actor_distance), maximum(df.actor_distance))

# Create a 2D grid
ego_distance_grid = range(ego_distance_range[1], stop=ego_distance_range[2], length=num_points)
actor_distance_grid = range(actor_distance_range[1], stop=actor_distance_range[2], length=num_points)

grid_ego, grid_actor = meshgrid(ego_distance_grid, actor_distance_grid)

# Velocities to iterate over
velocities = [10, 15, 25]

# Loop through velocities
for specified_ego_velocity in velocities
    # Create grid with the specified ego_velocity value
    grid_data = hcat(reshape(grid_ego, :, 1), fill(specified_ego_velocity, size(grid_ego)...), reshape(grid_actor, :, 1))
    grid_tensor = Array{Float32}(grid_data)
    
    # Evaluate the model on the grid
    predictions = net(grid_tensor)
    
    # Reshape predictions to have the same shape as the grid
    pred_reshaped = reshape(predictions, size(grid_ego))
    
    # Create a 2D heatmap
    heatmap(ego_distance_grid, actor_distance_grid, pred_reshaped, color=:coolwarm, xlabel="Ego Distance", ylabel="Actor Distance", title="Heatmap for Ego Velocity = $specified_ego_velocity")
    
    # Display the plot
    display(current())
end