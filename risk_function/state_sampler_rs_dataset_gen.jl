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

s2pt(s) = s

function compute_risk_weights(eje, aje)

    env = SignalizedJunctionTurnLeftMDP(junction_end_ego=eje, junction_end_actor=aje, start_detect_ego=start_detect_ego, collision_d_ul = 5, collision_d_ll = -15, 
                                        collision_d = -1, in_junction_stop_th=-8, speed_limit=25, max_accel=4.6, max_decel=-4.6, dt=0.1,
                                        d_ego0=Distributions.Uniform(30, 100), v_ego0=Distributions.Uniform(5, 40), d_actor0=Distributions.Uniform(10, 100))

    ds_ego = sort(collect(range(50, (eje - 1), 32)))
    vs_ego = sort(collect(range(25, 0, 5)))
    ds_actor = sort(collect(range(50, aje - 10, 16)))

    detects = [0, 1]

    policy = GetNaivePolicy(env)

    function costfn(m, s, sp)
        d_actor, v_ego, d_ego = sp
        cost = 0
        if isterminal(m, sp)
            extent = check_violation_extent(m, sp)
            cost += extent * 20
        end

        return cost
    end
    rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

    path = "./risk_function/surrogate_model.onnx"
    sample_input = rand(Float32, 3)
    println("Loading the model")
    surrogate_model = ONNX.load(path, sample_input)
    function surrogate_model_pass(tape::Tape, sample_vector::Array{Float64,1})
        x = reshape(sample_vector, length(sample_vector), 1)
        y = play!(tape, x)
        return y
    end
    p_detect(s) = surrogate_model_pass(surrogate_model, [s[3], s[2], s[1]])[1]

    function get_detect_dist(s)
        pd = p_detect(s)
        noises = [[ϵ, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
        dist = ObjectCategorical(noises, [1 - pd, pd])
        return dist
    end

    noises_detect = [0, 1]
    ϵ_grid = RectangleGrid(noises_detect)
    noises = [[ϵ[1], 0.0, 0.0, 0.0] for ϵ in ϵ_grid]

    px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

    cost_points = collect(range(0, 20, 21))
    s_grid = RectangleGrid(ds_actor, vs_ego, ds_ego)
    𝒮 = [[d_actor, v_ego, d_ego] for d_actor in ds_actor, v_ego in vs_ego, d_ego in ds_ego]
    s2pt(s) = s

    @time Uw, Qw, us = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
        cost_points, mdp_type=:exp)

    calc_CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

    riskmin(x; α) = minimum([calc_CVaR(x, [noise], α) for noise in noises_detect])
    riskmax(x; α) = maximum([calc_CVaR(x, [noise], α) for noise in noises_detect])
    risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)

    return risk_weight
end


function sample_random_state(scenario)
    # Ownship state
    aje = scenario_params[scenario][2]
    actor_distance = rand(Distributions.Uniform(aje, 60))
    ego_velocity = rand(SparseCat([10, 15, 25], [1/3, 1/3, 1/3]))
    ego_distance = rand(Distributions.Uniform(-10, 50))
    lane_actor = rand(SparseCat([1, 2], [0.5, 0.5]))
    actor = rand(SparseCat([1,2,3,4], [0.25, 0.25, 0.25, 0.25]))

    return actor_distance, ego_velocity, ego_distance, lane_actor, actor, scenario
end

function rejection_sample_states(N, risk_weight, sc; baseline=0.2, α=0.0)
    # Store samples in dataframe
    samples = DataFrame(d_actor=Float64[], d_ego=Float64[], lane_actor=Float64[], actor=Float64[], scenario=Float64[])

    ind = 1
    while ind ≤ N
        actor_distance, ego_velocity, ego_distance, lane_actor, actor, scenario = sample_random_state(sc)
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

merged_samples = DataFrame(d_actor=Float64[], d_ego=Float64[], lane_actor=Float64[], actor=Float64[], scenario=Float64[])

for (scenario, params) in scenario_params
    eje, aje = params
    risk_weights = compute_risk_weights(eje, aje)
    samples = rejection_sample_states(650, risk_weights, scenario; baseline=0)
    global merged_samples = vcat(merged_samples, samples)
end


# samples = rejection_sample_states(2600)



CSV.write("rs_states.csv", merged_samples)