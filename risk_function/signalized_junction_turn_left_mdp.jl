using Distributions, Parameters, Random
using POMDPTools, POMDPGym, POMDPs

@with_kw struct SignalizedJunctionTurnLeftMDP <: MDP{Array{Float32},Float32}
    safety_threshold::Float64 = 15..0 # if the ego_vehicle is inside the junction while non ego is 20m away, this will be considered a safety violation
    speed_limit::Float64 = 40.0 # applies to both ego vehicle and other actors.
    yield_threshold::Float64 = 50.0 # if the oncoming traffic < yield_threshold away, yield!
    actions = [0, 1]
    reward_safety_violation = -100
    dt = 0.1 # the time step

    distance_junction::Float64 = 30.0 # distance ego vehicle has to travel to successfully complete the task.
    ego_distance0 = Deterministic(distance_junction) # distance travelled by ego vehicle
    actor_distance0 = Distributions.Uniform(10, 100)

#--------------------------------------------------------------------------------------------------#

    ddh_max::Float64 = 1.0 # vertical acceleration limit [m/s²]
    collision_threshold::Float64 = 50.0 # collision threshold [m]
    reward_collision::Float64 = -100.0 # reward obtained if collision occurs
    reward_change::Float64 = -1 # reward obtained if action changes
    px = DiscreteNonParametric([2.0, 0.0, -2.0], [0.25, 0.5, 0.25])
#---------------------------------------------------------------------------------------------------#
end

# unchanged - not needed
function POMDPs.gen(mdp::SignalizedJunctionTurnLeftMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    t = transition(mdp, s, a, x)
    (sp=rand(t), r=reward(mdp, s, a))
end

# done - LOOK AT THE LOGIC FOR NOT DETECTED AGAIN. JUST FOLLOWED THE EXAMPLE FROM DAAMDP
function POMDPs.transition(mdp::SignalizedJunctionTurnLeftMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    not_detected = x == 0
    ego_distance, actor_distace = s

    actor_distace -= mdp.speed_limit * mdp.dt
    if (a == 1 || not_detected)
        ego_distance -= mdp.speed_limit * mdp.dt
    end

    return Deterministic(Float32[ego_distance, actor_distace])
    # a = x == 0 ? 0.0 : a # COC if don't detect

    # h, dh, a_prev, τ = s

    # # Update the dynamics
    # h = h + dh
    # #if a != 0.0
    # if abs(a - dh) < mdp.ddh_max
    #     dh += a - dh
    # else
    #     dh += sign(a - dh) * mdp.ddh_max
    # end
    # #end
    # a_prev = a
    # τ = max(τ - 1.0, -1.0)
    # SparseCat([Float32[h, dh+x, a_prev, τ] for x in mdp.px.support], mdp.px.p)
end

# done
function POMDPs.reward(mdp::SignalizedJunctionTurnLeftMDP, s, a)
    ego_distance, actor_distance = s

    r = 0.0
    if isfailure(mdp, s)
        # We collided
        r += mdp.reward_safety_violation
    end
    r
end

POMDPs.convert_s(::Type{Array{Float32}}, v::V where {V<:AbstractVector{Float64}}, ::SignalizedJunctionTurnLeftMDP) = Float32.(v)
POMDPs.convert_s(::Type{V} where {V<:AbstractVector{Float64}}, s::Array{Float32}, ::SignalizedJunctionTurnLeftMDP) = Float64.(s)

# double check what variables are part of state space
function POMDPs.initialstate(mdp::SignalizedJunctionTurnLeftMDP)
    ImplicitDistribution((rng) -> Float32[rand(mdp.ego_distance0), rand(mdp.actor_distance0)])
end

# done
POMDPs.actions(mdp::SignalizedJunctionTurnLeftMDP) = mdp.actions
POMDPs.actionindex(mdp::SignalizedJunctionTurnLeftMDP, a) = findfirst(mdp.actions .== a)

# unchanged - not needed
disturbanceindex(mdp::SignalizedJunctionTurnLeftMDP, x) = findfirst(mdp.px.support .== x)
disturbances(mdp::SignalizedJunctionTurnLeftMDP) = mdp.px.support

# done
function isfailure(mdp::SignalizedJunctionTurnLeftMDP, s)
    ego_distance, actor_distance = s
    actor_distance < mdp.safety_threshold && ego_distance > eps()
end

# done
function POMDPs.isterminal(mdp::SignalizedJunctionTurnLeftMDP, s)
    ego_distance, actor_distance = s
    ego_distance < eps() && actor_distance > eps()
end

# unchanged - not needed
POMDPs.discount(mdp::SignalizedJunctionTurnLeftMDP) = 0.99


## Hard Coded Naive Controller Policy

struct NaiveControlPolicy <: Policy
    𝒜
end

function NaivePolicy()
    return NaiveControlPolicy([0, 1])
end

function POMDPs.action(Policy:: NaiveControlPolicy, s)
    ego_distance, actor_distance = s
    action = actor_distance < 50 ? 0 : :stop
    return action
end

# Stuff from DetecAndAvoidMDP:

# ## Here is a solver that gives the optimal policy
# struct OptimalDetectAndAvoidPolicy <: Policy
#     𝒜
#     grid
#     Q
# end

# # my understanding is that this function calculates the optimal policy for for CONTROL problem, not the risk perception
# function OptimalDetectAndAvoidPolicy(mdp::SignalizedJunctionTurnLeftMDP, hs=range(-200, 200, length=21), dhs=range(-10, 10, length=21), τs=range(0, 40, length=41))
#     grid = RectangleGrid(hs, dhs, actions(mdp), τs)

#     𝒮 = [[h, dh, a_prev, τ] for h in hs, dh in dhs, a_prev in actions(mdp), τ in τs]

#     # State value function
#     U = zeros(length(𝒮))

#     # State-action value function
#     Q = [zeros(length(𝒮)) for a in actions(mdp)]

#     # Solve with backwards induction value iteration
#     for (si, s) in enumerate(𝒮)
#         for (ai, a) in enumerate(actions(mdp))
#             Tsa = transition(mdp, s, a, 1.0)
#             Q[ai][si] = reward(mdp, s, a)
#             Q[ai][si] += sum(isterminal(mdp, s′) ? 0.0 : Tsa.probs[j] * GridInterpolations.interpolate(grid, U, vec(s′)) for (j, s′) in enumerate(Tsa.vals))
#         end
#         U[si] = maximum(q[si] for q in Q)
#     end
#     return OptimalDetectAndAvoidPolicy(actions(mdp), grid, Q)
# end

# function POMDPs.action(policy::OptimalDetectAndAvoidPolicy, s)
#     a_best = first(policy.𝒜)
#     q_best = -Inf
#     for (a, q) in zip(policy.𝒜, policy.Q)
#         q = GridInterpolations.interpolate(policy.grid, q, s)
#         if q > q_best
#             a_best, q_best = a, q
#         end
#     end
#     return a_best
# end
