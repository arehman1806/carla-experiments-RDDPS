# Neural network model
struct SurrogateNN
    layer1
    layer2
end

SurrogateNN(input_dim::Int, hidden_dim::Int) = SurrogateNN(Dense(input_dim, hidden_dim, relu), Dense(hidden_dim, 1, σ))

function (m::SurrogateNN)(x)
    x = m.layer1(x)
    x = m.layer2(x)
    return x
end