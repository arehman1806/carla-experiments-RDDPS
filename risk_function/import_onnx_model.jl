using ONNX
using Images
import Umlaut: Tape, play!
using Plots


function surrogate_model_pass(tape::Tape, sample_vector::Array{Float32,1})
    # Process the input data
    x = reshape(sample_vector, length(sample_vector), 1)
    y = play!(tape, x)
    return y
end

function main()
    # Update the path to your ONNX model
    path = "surrogate_model.onnx"
    
    # Dummy input for your surrogate model
    sample_input = rand(Float32, 3)  # Adjust as per your actual input size

    # Load the model as a Umlaut.Tape
    println("Loading the model")
    surrogate_model = ONNX.load(path, sample_input)
    
    p_detect(s) = surrogate_model_pass(surrogate_model, s)[1]
    ego_junction_end = -20
    actor_junction_end = -15
    start_detect_ego = 100
    ds_ego = sort(collect(range(50, (ego_junction_end - 1), 70)))
    vs_ego = sort(collect(range(25, 10, 15)))
    ds_actor = sort(collect(range(50, actor_junction_end - 10, 65)))

    display(heatmap(ds_ego, ds_actor, (d_ego, d_actor)->p_detect(Float32[d_ego, 10, d_actor])))
    println("Output: ", output)
end

main()
