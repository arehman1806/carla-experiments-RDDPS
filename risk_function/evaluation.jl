using POMDPs, POMDPGym, Crux, Distributions, Random, GridInterpolations, POMDPTools, POMDPSimulators
using StatsBase
using Plots
using BSON
using ONNX
using Images
import Umlaut: Tape, play!
using DataFrames, CSV
using Statistics
include("./risk_mdp.jl")
include("./sjtl_mdp.jl")
include("./risk_solvers.jl")
include("./surrogate_nn_architecture.jl")
include("./traffic_parameters.jl")


function get_initial_state(d_actor0=Distributions.Uniform(30, 60), speed_limit=11.0)
    d_actor =  rand(d_actor0)
    v_actor = speed_limit
    t_coll = (d_actor - (-1)) / speed_limit
    abs_d_ego = speed_limit * t_coll
    d_ego = ego_junction_end + abs_d_ego
    return [d_actor, speed_limit, d_ego]
end



env = SignalizedJunctionTurnLeftMDP(junction_end_ego=ego_junction_end, junction_end_actor=actor_junction_end, start_detect_ego=start_detect_ego, collision_d_ul = 5, collision_d_ll = -15, 
                                    collision_d = -1, in_junction_stop_th=-8, speed_limit=11, max_accel=4.6, max_decel=-4.6, dt=0.1,
                                    d_ego0=Distributions.Uniform(10, 60), v_ego0=Distributions.Uniform(10,11), d_actor0=Distributions.Uniform(10, 50))


ds_ego = sort(collect(range(50, (ego_junction_end - 1), 32)))
# vs_ego = [10, 15, 25]
ds_actor = sort(collect(range(50, actor_junction_end - 10, 16)))
detects = [0, 1]

# display(scatter(all_d, zeros(length(all_d))))

policy = GetNaivePolicy(env)

function costfn(m, s, sp)
    d_actor, v_ego, d_ego = sp
    cost = 0
    if isterminal(m, sp)
        extent = check_violation_extent(m, sp)
        if extent > 0.8
            return 1
        else
            return 0
        end
        # cost += extent * 20
    end
    
    return cost
end
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

function surrogate_model_pass(tape::Tape, sample_vector::Array{Float32,1})
    x = reshape(sample_vector, length(sample_vector), 1)
    y = play!(tape, x)
    return y
end
path = "./risk_function/models/surrogate_baseline_old_model.onnx"
sample_input = rand(Float32, 3)

surrogate_model_baseline = ONNX.load(path, sample_input)
p_detect(s) = surrogate_model_pass(surrogate_model_baseline, [s[3], s[2], s[1]])[1]

path = "./risk_function/models/surrogate_risk_old_model.onnx"
surrogate_model_risk = ONNX.load(path, sample_input)
p_detect_risk(s) = surrogate_model_pass(surrogate_model_risk, [s[3], s[2], s[1]])[1]

# path = "./risk_function/surrogate_risk_model.onnx"
# surrogate_model_baseline = ONNX.load(path, sample_input)
# p_detect_rs(s) = surrogate_model_pass(surrogate_model_baseline, [s[3], s[2], s[1]])[1]



# display(heatmap(ds_ego, ds_actor, (d_ego, d_actor)->p_detect([d_ego, 15, d_actor])))

function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

function get_detect_dist_risk(s)
    pd = p_detect_risk(s)
    noises = [[ϵ, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

function get_detect_dist_rs(s)
    pd = p_detect_rs(s)
    noises = [[ϵ, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

noises_detect = [0, 1]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0, 0.0] for ϵ in ϵ_grid]
# probs = probs_detect # NOTE: will need to change once also have additive noise

px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))
px_risk = StateDependentDistributionPolicy(get_detect_dist_risk, DiscreteSpace(noises))
px_rs = StateDependentDistributionPolicy(get_detect_dist_rs, DiscreteSpace(noises))

# sim = RolloutSimulator()
# r = simulate(sim, rmdp, px, Float64[20, 25, 5])
# println("cost total: $r")



# Get the distribution of returns and plot
# N = 1000
# D = episodes!(Sampler(rmdp, px), Neps=N)
# samples = D[:r][1, D[:done][:]]

# p1 = histogram(samples, title="Costs BASELINE", bins=range(1, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")
# display(p1)
# # Calculate the 75th percentile
# q75 = quantile(samples, 0.75)

# # Filter values that are greater than or equal to the 75th percentile
# top_25_percent_values = samples[samples .≥ q75]

# # Calculate the mean of the top 25% values
# mean_top_25 = mean(top_25_percent_values)

# println("Mean of top 25% values for baseline: $mean_top_25")



# N = 1000
# D = episodes!(Sampler(rmdp, px_risk), Neps=N)
# samples = D[:r][1, D[:done][:]]

# p1 = histogram(samples, title="Costs RISK", bins=range(1, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")
# display(p1)
# # Calculate the 75th percentile
# q75 = quantile(samples, 0.75)

# # Filter values that are greater than or equal to the 75th percentile
# top_25_percent_values = samples[samples .≥ q75]

# # Calculate the mean of the top 25% values
# mean_top_25 = mean(top_25_percent_values)

# println("Mean of top 25% values for baseline: $mean_top_25")


bl = []
risk = []
sim = RolloutSimulator()
for i in 1:100000
    if i % 500 == 0
        println("i = $i")
    end
    s0 = get_initial_state()
    # println(s0)
    r_baseline = simulate(sim, rmdp, px, s0)
    push!(bl, r_baseline)
    r_risk = simulate(sim, rmdp, px_risk, s0)
    push!(risk, r_risk)
end
# Count the number of 1s in each list
count_bl = sum(bl)
count_risk = sum(risk)

# Plot the bar chart
bar(["Baseline", "Risk"], [count_bl, count_risk], legend=false, title="Number of collisions", ylabel="Count")#

print("collisions:\nbaseline: $count_bl\nrisk:$count_risk")