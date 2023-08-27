# Define the neural network architecture
struct SurrogateNN
    layer1
    layer2
    layer3
end

SurrogateNN(input_dim::Int, hidden_dim::Int) = SurrogateNN(
    Dense(input_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, 1, σ)
)

function (m::SurrogateNN)(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    return x
end