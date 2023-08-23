using POMDPs, POMDPGym, Crux, Distributions, Random, GridInterpolations, POMDPTools
using StatsBase
using Plots
using BSON

include("./risk_mdp.jl")
include("./sjtl_mdp.jl")
include("./risk_solvers.jl")
include("./surrogate_nn_architecture.jl")

ego_junction_end = -20
actor_junction_end = -15
start_detect_ego = 100

env = SignalizedJunctionTurnLeftMDP(junction_end_ego=ego_junction_end, junction_end_actor=actor_junction_end, start_detect_ego=start_detect_ego, collision_d_ul = 2, collision_d_ll = -10, 
                                    collision_d = -1, in_junction_stop_th=-8, speed_limit=25, max_accel=4.6, max_decel=-4.6, dt=0.1,
                                    d_ego0=Distributions.Uniform(30, 100), v_ego0=Distributions.Uniform(5, 40), d_actor0=Distributions.Uniform(10, 100))


ds_ego = sort(collect(range(50, (ego_junction_end - 1), 200)))
vs_ego = sort(collect(range(25, 10, 4)))
ds_actor = sort(collect(range(50, actor_junction_end - 10, 200)))
detects = [0, 1]

# display(scatter(all_d, zeros(length(all_d))))

policy = GetNaivePolicy(env)

# anim = @animate for v_ego in range(0, 25, 26)
# #     heatmap(all_d, all_v, (x, y) -> CVaR([x, y], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="distance ego (m)", ylabel="distance actor (m)")
#     heatmap(ds_ego, ds_actor, (d_ego, d_actor) -> action(policy, [d_ego, v_ego, d_actor, 1]), xlabel="d_ego", ylabel="d_actor", title="SJTL Policy. Ego_Vel = $v_ego")
#     vline!([-20, -8, 0], color=:black, lw=2, label=true)
# end
# Plots.gif(anim, "./risk_function/figures/sjtl_policy.gif", fps=6)

function costfn(m, s, sp)
    d_ego, v_ego, d_actor, a_detected = sp
    cost = 0
    if isterminal(m, sp)
        extent = check_violation_extent(m, sp)
        cost += extent * 20
    # elseif is_inside_junction(m, sp)
    #     cost += 1
        # println("terminal state is $sp")
        # println("returning cost $cost")
    end
    
    return cost
end
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

model_data = BSON.load("./risk_function/surrogate_nn_model.bson")
model = model_data[:model]
# p_detect(s) = model([s[2]])[1]
p_detect(s) = 0

# function p_detect(s)
#     distance_ego, distance_actor = s
#     # clip the distance_actor to be within 20 and 100
#     distance_actor = max(min(distance_actor, 100), 20)
#     # Linear interpolation between 20 and 100.
#     # pd will be 1 when distance_actor is 20, and 0 when it's 100.
#     pd = (100 - distance_actor) / 80.0
#     return 1
# end

# display(heatmap(ds_ego, ds_actor, (d_ego, d_actor)->p_detect([d_ego, 10, d_actor])))

function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

noises_detect = [0, 1]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0, 0.0, 0.0] for ϵ in ϵ_grid]
# probs = probs_detect # NOTE: will need to change once also have additive noise

px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

sim = RolloutSimulator()
r = simulate(sim, rmdp, px, [0, 25, 10, 0.0])
println("cost total: $r")



# Get the distribution of returns and plot
# N = 10000
# D = episodes!(Sampler(rmdp, px), Neps=N)
# samples = D[:r][1, D[:done][:]]

# p1 = histogram(samples, title="Costs", bins=range(0, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")


# display(p1)

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 100, 11))
# cost_points = [0, 50]
s_grid = RectangleGrid(ds_ego, vs_ego, ds_actor, detects)
𝒮 = [[d_ego, v_ego, d_actor, detect] for d_ego in ds_ego, v_ego in vs_ego, d_actor in ds_actor, detect in [0, 1]];
s2pt(s) = s

# # Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);


# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt([30, 25, 25, 0]))
si = si[argmax(wi)]
println(cost_points)
println(Uw[si])
println(s_grid[si])
p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 51, 50), normalize=true, alpha=1, label="DP")
display(p2)
# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

# Plot one sample
display(heatmap(ds_ego, ds_actor, (d_ego, d_actor) -> CVaR([d_ego, 25, d_actor, 0], [0], 0), title="α = 0, ego_vel = 25", xlabel="ego displacement from junction (m)", ylabel="actor displacement from junction (m)"))# plot(all_distance_actor, (y) -> CVaR([20, y], [0], 0.0), title="α = 0", xlabel="actor_distance_junction (m)", ylabel="Risk")
# hline!([0], color=:white, lw=2, label=false)
# vline!([0], color=:white, lw=2, label=false)
# savefig("./risk_function/figures/cvar_sjtl.png")

# # anim = @animate for α in range(-1.0, 1.0, length=51)
# #     heatmap(all_d, all_v, (x, y) -> CVaR([x, y], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="distance ego (m)", ylabel="distance actor (m)")
# # end
# # Plots.gif(anim, "./risk_function/figures/sfo_CVaR_v3.gif", fps=6)

# # function test_value_function(state)
# #     si, wi = GridInterpolations.interpolants(s_grid, s2pt(state))
# #     si = si[argmax(wi)]
# #     println(cost_points)
# #     println(Uw[si])
# #     println(s_grid[si])
# #     return cost_points, Uw[si]
# # end