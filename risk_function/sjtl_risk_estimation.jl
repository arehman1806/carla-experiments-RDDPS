# ----------------------------------------------------------------------------------
# This code is an extension, modification, and adaptation of the original code found at:
# https://github.com/sisl/RiskDrivenPerception
#
# Original work by the authors and contributors of the RiskDrivenPerception repository.
# Please refer to the original repository for more details on the foundational research.
#
# All modifications and adaptations made to this code are intended for further research 
# and exploration, and full credit goes to the original authors for their foundational work.
#
# If you have any concerns or queries regarding the changes made to this code, please reach out.
# ----------------------------------------------------------------------------------


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



env = SignalizedJunctionTurnLeftMDP(junction_end_ego=ego_junction_end, junction_end_actor=actor_junction_end, start_detect_ego=start_detect_ego, collision_d_ul = 5, collision_d_ll = -15, 
                                    collision_d = -1, in_junction_stop_th=-8, speed_limit=25, max_accel=4.6, max_decel=-4.6, dt=0.1,
                                    d_ego0=Distributions.Uniform(30, 100), v_ego0=Distributions.Uniform(5, 40), d_actor0=Distributions.Uniform(10, 100))


ds_ego = sort(collect(range(50, (ego_junction_end - 1), 70)))
vs_ego = sort(collect(range(25, 0, 26)))
ds_actor = sort(collect(range(50, actor_junction_end - 10, 65)))
# ds_ego = sort(collect(range(50, (ego_junction_end - 1), 10)))
# vs_ego = sort(collect(range(25, 0, 3)))
# ds_actor = sort(collect(range(50, actor_junction_end - 10, 10)))
detects = [0, 1]

policy = GetNaivePolicy(env)

# anim = @animate for v_ego in range(0, 25, 26)
# #     heatmap(all_d, all_v, (x, y) -> CVaR([x, y], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="distance ego (m)", ylabel="distance actor (m)")
#     heatmap(ds_ego, ds_actor, (d_ego, d_actor) -> action(policy, [d_actor, v_ego, d_ego, 1]), xlabel="d_ego", ylabel="d_actor", title="SJTL Policy. Ego_Vel = $v_ego")
#     vline!([-20, -8, 0], color=:black, lw=2, label=true)
# end
# Plots.gif(anim, "./risk_function/figures/sjtl_policy.gif", fps=6)

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

function surrogate_model_pass(tape::Tape, sample_vector::Array{Float64,1})
    x = reshape(sample_vector, length(sample_vector), 1)
    y = play!(tape, x)
    return y
end

path = "./risk_function/models/surrogate_baseline_old_model.onnx"
sample_input = rand(Float32, 3)
println("Loading the model")

surrogate_model = ONNX.load(path, sample_input)
p_detect(s) = surrogate_model_pass(surrogate_model, [s[3], s[2], s[1]])[1]
# p_detect(s) = 0

display(heatmap(ds_ego, ds_actor, (d_ego, d_actor)->p_detect([d_ego, 5, d_actor]), title="v_ego = 5", xlabel="d_ego (m)", ylabel="d_actor (m)"))
savefig("./risk_function/figures/detection_probability_baseline_5.png")

function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

noises_detect = [0, 1]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0, 0.0] for ϵ in ϵ_grid]

px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

sim = RolloutSimulator()
r = simulate(sim, rmdp, px, Float64[20, 25, 5])
println("cost total: $r")



# Get the distribution of returns and plot
# N = 10000
# D = episodes!(Sampler(rmdp, px), Neps=N)
# samples = D[:r][1, D[:done][:]]

# p1 = histogram(samples, title="Costs", bins=range(0, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")
# display(p1)

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 20, 21))
# cost_points = [0, 50]
s_grid = RectangleGrid(ds_actor, vs_ego, ds_ego)
𝒮 = [[d_actor, v_ego, d_ego] for d_actor in ds_actor, v_ego in vs_ego, d_ego in ds_ego];
s2pt(s) = s

# # Solve for distribution over costs
@time Uw, Qw, us = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);
    
# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

# si, wi = GridInterpolations.interpolants(s_grid, s2pt([20, 25, 5]))
# si = si[argmax(wi)]
# s = [20, 25, 15]
# println("risk of detection: $(CVaR(s, [1], 0)). risk of not detection: $(CVaR(s, [0], 0))")

# Plot one sample
display(heatmap(ds_ego, ds_actor, (d_ego, d_actor) -> CVaR([d_actor, 10, d_ego], [0], 0), title="α = 0, ego_vel = 5", xlabel="d_ego (m)", ylabel="d_actor (m)"))


# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt([30, 25, 25]))
si = si[argmax(wi)]
println(cost_points)
println(Uw[si])
println(s_grid[si])
p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 51, 50), normalize=true, alpha=1, label="DP")
display(p2)

# Plot one sample
display(heatmap(ds_ego, ds_actor, (d_ego, d_actor) -> CVaR([d_actor, 25, d_ego], [0], 0.9), title="α = 0.9, ego_vel = speed_limit", xlabel="d_ego (m)", ylabel="d_actor (m)"))
savefig("./risk_function/figures/cvar_sjtl_alpha_0p9.png")

# anim = @animate for α in range(0, 1.0, length=51)
#     heatmap(ds_ego, ds_actor, (x, y) -> CVaR([y, 5, x], [0], α), title="CVaR (α = $α)", xlabel="distance ego (m)", ylabel="distance actor (m)", clims=(0, 20))
# end
# Plots.gif(anim, "./risk_function/figures/sjtl_CVaR.gif", fps=6)

# Most important states
riskmin(x; α) = minimum([CVaR(x, [noise], α) for noise in noises_detect])
riskmax(x; α) = maximum([CVaR(x, [noise], α) for noise in noises_detect])
risk_weight(x; α) = riskmax(x; α) - riskmin(x; α)

display(heatmap(collect(range(-25, 50, 100)), collect(range(-25, 50, 100)), (x, y) -> risk_weight([y, 25, x], α=00), xlabel="d_ego (m)", ylabel="d_actor (m)", title="Risk Wights", colorbar=false))#, clims=(0, 0.05))
savefig("./risk_function/figures/risk_weights_alpha_0")

# anim = @animate for α in range(-1.0, 1.0, length=51)
#     heatmap(collect(range(-20, 30, 100)), collect(range(-20, 30, 100)), (x, y) -> risk_weight([y, 25, x], α=α), xlabel="τ (s)", ylabel="h (m)", title="Risk of Perception Errors: α = $(α)", clims=(0, 6))
# end
# Plots.gif(anim, "risk_function/figures/daa_risk_weights.gif", fps=6)