using Flux
using Flux: logitcrossentropy, params, gradient, update!
using CSV
using DataFrames
using Random
using BSON

Random.seed!(0) # Equivalent to setting random seed

include("./surrogate_nn_architecture.jl")

# Read data from CSV
df = CSV.File("./risk_function/datasets/surrogate_EM_dataset.csv") |> DataFrame

# Modify distance_to_junction based on at_junction value
transform!(df, :at_junction => ByRow(x -> x ? -1 : 1) => :sign)
df[!, :distance_to_junction] .= df[!, :sign] .* df[!, :distance_to_junction]

# Extract data and labels
train_data = reshape(Float32.(df[!, :distance_to_junction]), 1, :)
train_labels = reshape(Float32.(df[!, :has_detected]), 1, :)



# Neural network and training setup
model = SurrogateNN(1, 10)
loss(x, y) = logitcrossentropy(model(x), y)
opt = Descent(0.001)

# Training loop
epochs = 100000
for epoch in 1:epochs
    grads = gradient(params(model)) do
        loss(train_data, train_labels)
    end
    update!(opt, params(model), grads)
    if epoch % 10000 == 0
        @info "Epoch $epoch/$epochs, Loss: $(loss(train_data, train_labels))"
    end
end


bson("./risk_function/surrogate_nn_model.bson", model=model)


using Plots

# Predict with the model
predictions = model(train_data)

# Convert predictions to the same shape as train_labels for plotting
predictions = reshape(predictions, size(train_labels)...)

# Plot
scatter(train_data', train_labels', label="Training Data", alpha=0.5, legend=:bottomright)
scatter!(train_data', predictions', label="Model Predictions", color=:red, alpha=0.5)
xlabel!("distance_to_junction")
ylabel!("has_detected")
