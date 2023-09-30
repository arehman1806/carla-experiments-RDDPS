using POMDPs, POMDPGym, Crux, Distributions, Random, GridInterpolations, POMDPTools
using StatsBase
using Plots
using BSON
using ONNX
using Images
import Umlaut: Tape, play!
using DataFrames, CSV
include("./risk_mdp.jl")
include("./sjtl_mdp.jl")
include("./risk_solvers.jl")
include("./surrogate_nn_architecture.jl")
include("./traffic_parameters.jl")

function sample_random_state()
    # Ownship state
    scenario = rand(SparseCat([1,2,3,4], [0.25, 0.25, 0.25, 0.25]))
    aje = scenario_params[scenario][2]
    actor_distance = rand(Distributions.Uniform(aje, 60))
    ego_distance = rand(Distributions.Uniform(-10, 50))
    lane_actor = rand(SparseCat([1, 2], [0.5, 0.5]))
    actor = rand(SparseCat([1,2,3,4], [0.25, 0.25, 0.25, 0.25]))

    return actor_distance, ego_distance, lane_actor, actor, scenario
end

function rejection_sample_states(N; baseline=0.2, α=0.0)
    # Store samples in dataframe
    samples = DataFrame(d_actor=Float64[], d_ego=Float64[], lane_actor=Float64[], actor=Float64[], scenario=Float64[])

    ind = 1
    while ind ≤ N
        actor_distance, ego_distance, lane_actor, actor, scenario = sample_random_state()
        rw = risk_weight([actor_distance, ego_velocity, ego_distance], α=α)
        if rand() < rw + baseline
            # Store the sample
            push!(samples, [actor_distance, ego_distance, lane_actor, actor, scenario])
            ind += 1
        end
        ind % 500 == 0 ? println(ind) : nothing
    end

    return samples
end

function baseline_states(N; baseline=0, α=0.0)
    # Store samples in dataframe
    samples = DataFrame(d_actor=Float64[], d_ego=Float64[], lane_actor=Float64[], actor=Float64[], scenario=Float64[])

    ind = 1
    while ind ≤ N
        actor_distance, ego_distance, lane_actor, actor, scenario = sample_random_state()
        
        push!(samples, [actor_distance, ego_distance, lane_actor, actor, scenario])
        ind += 1

        ind % 500 == 0 ? println(ind) : nothing
    end

    return samples
end

samples = baseline_states(2600)



CSV.write("baseline_states.csv", samples)