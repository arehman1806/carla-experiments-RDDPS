# Wrote this MDP for practice. This MDP models a scenario where 
# Ego vehicle has to detect and obstacle and stop before colliding into it.

using Distributions, Parameters, Random
using POMDPTools, POMDPGym, POMDPs

@with_kw struct StopForObstacleMDP <: MDP{Array{Float32},Float32}
    safe_stop_distance::Float64 = 5.0 # If ego vehicle crosses this threshold, it will be a collision.
    collision_threshold = 5.0
    actions = [0, 1]
    reward_safety_violation = -100
    dt = 0.1 # the time step
    a_min = -4.6
    d0 = Distributions.Uniform(10, 300)
    v0 = Distributions.Uniform(10, 40) # ego initial velocity

end

# unchanged - not needed
function POMDPs.gen(mdp::StopForObstacleMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    t = transition(mdp, s, a, x)
    (sp=rand(t), r=reward(mdp, s, a))
end

function POMDPs.transition(mdp::StopForObstacleMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    not_detected = x == 0
    # println(x)
    d, v = s
    # a_required = -(v^2) / (2*d)
    a_required = a
    # println("original: $a_required")
    a_required = a_required < mdp.a_min ? mdp.a_min : a_required
    # println("after check 1: $a_required")
    a_required = not_detected ? 0 : a_required
    # println("after_check3: $a_required")

    v_next = v + a_required * mdp.dt
    d_next = d
    if v >= 0
        d_next = d - v * mdp.dt - (0.5 * a * mdp.dt^2)
    end
    # println("d: $d, v: $v, d_next: $d_next, v_next: $v_next")
    return SparseCat([Float32[d_next, v_next]], [1])
end


# done
function POMDPs.reward(mdp::StopForObstacleMDP, s, a)
    d, v = s
    return 0
end

POMDPs.convert_s(::Type{Array{Float32}}, v::V where {V<:AbstractVector{Float64}}, ::StopForObstacleMDP) = Float32.(v)
POMDPs.convert_s(::Type{V} where {V<:AbstractVector{Float64}}, s::Array{Float32}, ::StopForObstacleMDP) = Float64.(s)

# double check what variables are part of state space
function POMDPs.initialstate(mdp::StopForObstacleMDP)
    ImplicitDistribution((rng) -> Float32[rand(mdp.d0), rand(mdp.v0)])
end

# done
POMDPs.actions(mdp::StopForObstacleMDP) = mdp.actions
POMDPs.actionindex(mdp::StopForObstacleMDP, a) = findfirst(mdp.actions .== a)

# unchanged - not needed
disturbanceindex(mdp::StopForObstacleMDP, x) = findfirst(mdp.px.support .== x)
disturbances(mdp::StopForObstacleMDP) = mdp.px.support

# done
function isfailure(mdp::StopForObstacleMDP, s)
    return false
end

# done
function POMDPs.isterminal(mdp::StopForObstacleMDP, s)
    d, v = s

    result = d <= 0.1 || v <= 0
    # println("Ego Distance: $ego_distance, Actor Distance: $actor_distance, isterminal: $result")
    # println("is terminal is $result")
    return result
end

function check_safety_condition(mdp:: StopForObstacleMDP, s)
    ego_distance, actor_distance = s
    result = !(actor_distance > mdp.safety_threshold || actor_distance <= -10 - mdp.distance_junction)
    # println("safety condition violated: $result. state: $s")
    return result
end


# unchanged - not needed
POMDPs.discount(mdp::StopForObstacleMDP) = 0.99


## Hard Coded Naive Controller Policy

struct NaiveControlPolicy <: Policy
    𝒜
end

function GetNaivePolicy()
    return NaiveControlPolicy([0, 1])
end

function POMDPs.action(Policy:: NaiveControlPolicy, s)
    d, v = s
    if d < eps() || v < eps()
        println("returning 0 accel")
        return 0
    end
    a_required = -(v^2) / (2*d + 1e-6)
    # a_required = a_required < -4.6 ? -4.6 : a_required
    return a_required
end