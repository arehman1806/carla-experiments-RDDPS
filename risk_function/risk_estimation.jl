using POMDPs, POMDPGym, Crux, Distributions, Random, GridInterpolations, POMDPTools
using StatsBase
using Plots
using BSON

include("./risk_mdp.jl")
include("signalized_junction_turn_left_mdp.jl")
include("./risk_solvers.jl")
include("./surrogate_nn_architecture.jl")

distance_junction = 20

env = SignalizedJunctionTurnLeftMDP(distance_junction=distance_junction, safety_threshold=1, speed_limit=30, yield_threshold=5, reward_safety_violation=-100, dt=0.1, 
        ego_distance0=Deterministic(distance_junction), actor_distance0=Distributions.Uniform(1, 50))

distance_actor_max = 50
positive_range = distance_actor_max .- (collect(range(0, stop=distance_actor_max^(1/0.5), length=20))).^0.5
negative_range = -positive_range # Create the negative mirror of the positive range
all_distance_actor = sort(vcat(negative_range, positive_range))
# all_distance_actor = sort(distance_actor_max .- (collect(range(0, stop=distance_actor_max^(1/0.5), length=40))).^0.5)
all_distance_ego = sort(reverse(collect(range(0, 20, 40))))

display(scatter(all_distance_actor, zeros(length(all_distance_actor))))

policy = GetNaivePolicy()

# display(heatmap(all_distance_ego, all_distance_actor, (distance_ego, distance_actor) -> action(policy, [distance_ego, distance_actor, 0.0]), xlabel="τ (s)", ylabel="h (m)", title="CAS Policy"))


function costfn(m, s, sp)
    ego_distance, actor_distance = s
    dist_min = 5
    dist_max = 10 + distance_junction

    cost = 0
    if isterminal(m, sp)
        # println("is terminal")
        if abs(actor_distance) < dist_min
            cost += 50
        elseif abs(actor_distance) < dist_max
            cost += 100/abs(actor_distance)
        end
    end
    println("returning cost $cost")
    return cost

end
rmdp = RMDP(env, policy, costfn, false, 1.0, 40.0, :both)

model_data = BSON.load("./risk_function/surrogate_nn_model.bson")
model = model_data[:model]
p_detect(s) = model([s[2]])[1]
# p_detect(s) = 0.5

# function p_detect(s)
#     distance_ego, distance_actor = s
#     # clip the distance_actor to be within 20 and 100
#     distance_actor = max(min(distance_actor, 100), 20)
#     # Linear interpolation between 20 and 100.
#     # pd will be 1 when distance_actor is 20, and 0 when it's 100.
#     pd = (100 - distance_actor) / 80.0
#     return 1
# end

# display(heatmap(all_distance_ego, all_distance_actor, (ego_distance, actor_distance)->p_detect([ego_distance, actor_distance, 0.0])))

function get_detect_dist(s)
    pd = p_detect(s)
    noises = [[ϵ, 0.0, 0.0] for ϵ in [0, 1]]
    dist = ObjectCategorical(noises, [1 - pd, pd])
    # display(dist)
    return dist
end

noises_detect = [0, 1]

ϵ_grid = RectangleGrid(noises_detect)
noises = [[ϵ[1], 0.0, 0.0] for ϵ in ϵ_grid]
#probs = probs_detect # NOTE: will need to change once also have additive noise

px = StateDependentDistributionPolicy(get_detect_dist, DiscreteSpace(noises))

# sim = RolloutSimulator()
# r = simulate(sim, rmdp, px, [20.0, 100])
# println("cost total: $r")



# Get the distribution of returns and plot
N = 10000
D = episodes!(Sampler(rmdp, px), Neps=N)
samples = D[:r][1, D[:done][:]]


# print(samples)

p1 = histogram(samples, title="Costs", bins=range(0, 100, 100), normalize=true, alpha=0.3, xlabel="cost", label="MC")

print(length(samples))

display(p1)

# Set up cost points, state grid, and other necessary data
cost_points = collect(range(0, 100, 11))
# cost_points = [0, 50]
s_grid = RectangleGrid(all_distance_ego, all_distance_actor)
𝒮 = [[distance_ego, distance_actor] for distance_ego in all_distance_ego, distance_actor in all_distance_actor];
s2pt(s) = s

# Solve for distribution over costs
@time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
    cost_points, mdp_type=:exp);

function test_value_function(state)
    si, wi = GridInterpolations.interpolants(s_grid, s2pt(state))
    si = si[argmax(wi)]
    println(cost_points)
    println(Uw[si])
    println(s_grid[si])
    return cost_points, Uw[si]
end
# Grab the initial state
si, wi = GridInterpolations.interpolants(s_grid, s2pt([20, 15]))
si = si[argmax(wi)]
println(cost_points)
println(Uw[si])
println(s_grid[si])
p2 = histogram!(cost_points, weights=Uw[si], bins=range(0, 100, 50), normalize=true, alpha=1, label="DP")
display(p2)
# Create CVaR convenience functions
CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

# Plot one sample
# heatmap(all_distance_ego, all_distance_actor, (x, y) -> CVaR([x, y], [0], 0.0), title="α = 0", xlabel="ego_distance_left (m)", ylabel="actor_distance_left (m)")
plot(all_distance_actor, (y) -> CVaR([20, y], [0], 0.0), title="α = 0", xlabel="actor_distance_junction (m)", ylabel="Risk")

# hline!([0], color=:black, lw=2, label=false)
# anim = @animate for α in range(-1.0, 1.0, length=51)
#     heatmap(all_distance_ego, all_distance_actor, (x, y) -> CVaR([x, y], [0], α), title="CVaR (α = $α)", clims=(0, 150), xlabel="distance ego (m)", ylabel="distance actor (m)")
# end
# Plots.gif(anim, "./risk_function/figures/daa_CVaR_v3.gif", fps=6)

# -------------------------------------------------------

# function total_rewards_per_episode(D)
#     rewards = D[:r]
#     episode_ends = D[:episode_end]
    
#     total_rewards = Float32[]  # to store summed rewards for each episode
#     episode_reward = 0.0
    
#     for i in 1:length(rewards)
#         episode_reward += rewards[i]
#         if episode_ends[i]
#             push!(total_rewards, episode_reward)
#             episode_reward = 0.0
#         end
#     end

#     return total_rewards
# end

# episode_rewards = total_rewards_per_episode(D)
