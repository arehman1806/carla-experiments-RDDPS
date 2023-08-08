using GridInterpolations, Distributions
test_values = [-10.0, 0.0, 25.0, 50.0, 75.0]
for val in test_values
    indices, weights = interpolants(test_grid, [val])
    println("Value: ", val, " | Indices: ", indices, " | Weights: ", weights)
end
