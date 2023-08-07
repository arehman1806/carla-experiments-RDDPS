using POMDPs, POMDPGym, Crux, Distributions, Random, GridInterpolations, POMDPTools
using StatsBase
using Plots

include("./risk_mdp.jl")
include("signalized_junction_turn_left_mdp.jl")
include("./risk_solvers.jl")

distance_junction = 20

env = SignalizedJunctionTurnLeftMDP(distance_junction=distance_junction, safety_threshold=5, speed_limit=40.0, yield_threshold=5, reward_safety_violation=-100, dt=0.1, 
        ego_distance0=Deterministic(distance_junction), actor_distance0=Distributions.Uniform(distance_junction/2, distance_junction*5))

distance_actor_max = 100
all_distance_actor = sort(distance_actor_max .- (collect(range(0, stop=distance_actor_max^(1/0.5), length=40))).^0.5)
all_distance_ego = collect(range(0, 30, 40))
all_distance_ego = collect(range(0, 30, 40))

display(scatter(all_distance_actor, zeros(length(all_distance_actor))))

policy = GetNaivePolicy()

display(heatmap(all_distance_ego, all_distance_actor, (distance_ego, distance_actor) -> action(policy, [distance_ego, distance_actor, 0.0]), xlabel="τ (s)", ylabel="h (m)", title="CAS Policy"))
# costfn(m, s, sp) = isterminal(m, sp) ? abs(s[1]) : 0.0
function costfn(m, s, sp)
    ego_distance, actor_distance, collision_occured = s
    # println("Ego Distance: $ego_distance, Actor Distance: $actor_distance, Collision Occurred: $collision_occured")
    cost = 1
    if isfailure(m, sp)
        # High cost for safety violation
        cost = cost + 80
    end
    return cost

end
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

function p_detect(s)
    distance_ego, distance_actor, collision_occured = s
    # clip the distance_actor to be within 20 and 100
    distance_actor = max(min(distance_actor, 100), 20)
    # Linear interpolation between 20 and 100.
    # pd will be 1 when distance_actor is 20, and 0 when it's 100.
    pd = (100 - distance_actor) / 80.0
    return pd
end

display(heatmap(all_distance_ego, all_distance_actor, (ego_distance, actor_distance)->p_detect([ego_distance, actor_distance, 0.0])))

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
#probs = probs_detect # NOTE: will need to change once also have additive noise

px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

sim = RolloutSimulator()
r = simulate(sim, rmdp, px, [20, 18, 0.0])

# s0 = [20, 90]
# r_total = 0.0
# d = 1.0
# while !isterminal(rmdp, s0)
#     a = action(px, s0)
#     s0, r = @gen(:sp,:r)(rmdp, s0, a)
#     r_total += d*r
#     d *= discount(mdp)
# end
# println("reward_total:}"+ string(r_total))


# # Get the distribution of returns and plot
# N = 1000
# D = episodes!(Sampler(rmdp, px), Neps=N)
# samples = D[:r][1, D[:episode_end][:]]

# p1 = histogram(samples, title="CAS Costs", bins=range(0, 100, 50), normalize=true, alpha=0.3, xlabel="cost", label="MC")

# print(length(samples))

# display(p1)

# # Set up cost points, state grid, and other necessary data
# cost_points = collect(range(0, 100, 50))
# s_grid = RectangleGrid(all_distance_ego, all_distance_actor)
# 𝒮 = [[distance_ego, distance_actor] for distance_ego in all_distance_ego, distance_actor in all_distance_actor];
# s2pt(s) = s

# # Solve for distribution over costs
# @time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
#     cost_points, mdp_type=:exp);

# # Grab the initial state
# si, wi = GridInterpolations.interpolants(s_grid, s2pt([20, 30]))
# si = si[argmax(wi)]
# println(cost_points)
# println(Uw[si])
# println(s_grid[si])
# p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 100, 50), normalize=true, alpha=0.4, label="DP")
# display(p2)
# # # Create CVaR convenience functions
# # CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

# # # Plot one sample
# # # heatmap(τs, hs, (x, y) -> CVaR([y, 0.0, 0.0, x], [0], 0.0), title="α = 0")

# # anim = @animate for α in range(-1.0, 1.0, length=51)
# #     heatmap(all_distance_ego, all_distance_actor, (x, y) -> CVaR([x, y], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="τ (s)", ylabel="h (m)")
# # end
# # Plots.gif(anim, "./risk_function/figures/daa_CVaR_v3.gif", fps=6)

