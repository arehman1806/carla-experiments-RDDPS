using Distributions, Parameters, Random
using POMDPTools, POMDPGym, POMDPs

@with_kw struct SignalizedJunctionTurnLeftMDP <: MDP{Array{Float32},Float32}
    junction_end_ego::Float64 = -20
    junction_end_actor::Float64 = -15
    start_detect_ego = 100
    actor_collision_threshold = -10 # if actor is in the junction until point when ego leaves, a collision has occured
    in_junction_stop_th::Float64 = -8 # max distance ego can stop inside the junction
    speed_limit::Float64 = 40 # applies to both ego and actor
    max_accel::Float64 = 4.6
    max_decel::Float64 = -4.6
    actions = [0, 1]
    reward_safety_violation = -100
    dt = 0.1 # the time step
    a_min = -4.6
    d0 = Distributions.Uniform(10, 300)
    v0 = Distributions.Uniform(10, 40) # ego initial velocity

end

# unchanged - not needed
function POMDPs.gen(mdp::SignalizedJunctionTurnLeftMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    t = transition(mdp, s, a, x)
    (sp=rand(t), r=reward(mdp, s, a))
end

function POMDPs.transition(mdp::SignalizedJunctionTurnLeftMDP, s, a_required, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    detected = x != 0
    # println(x)
    d_ego, v_ego, d_actor, a_detected = s
    v_ego_next = max((v_ego + a_required * mdp.dt), 0) # its speed not vel so cant get below 0
    d_ego_next = d_ego - (v_ego * mdp.dt + (0.5 * a_required * mdp.dt^2))
    d_actor_next = d_actor - mdp.speed_limit*mdp.dt
    # println("d: $d, v: $v, d_next: $d_next, v_next: $v_next")
    return SparseCat([Float32[d_ego_next, v_ego_next, d_actor_next, detected]], [1])
end


# done
function POMDPs.reward(mdp::SignalizedJunctionTurnLeftMDP, s, a)
    d, v = s
    return 0
end

POMDPs.convert_s(::Type{Array{Float32}}, v::V where {V<:AbstractVector{Float64}}, ::SignalizedJunctionTurnLeftMDP) = Float32.(v)
POMDPs.convert_s(::Type{V} where {V<:AbstractVector{Float64}}, s::Array{Float32}, ::SignalizedJunctionTurnLeftMDP) = Float64.(s)

# double check what variables are part of state space
function POMDPs.initialstate(mdp::SignalizedJunctionTurnLeftMDP)
    ImplicitDistribution((rng) -> Float32[rand(mdp.d0), rand(mdp.v0)])
end

# done
POMDPs.actions(mdp::SignalizedJunctionTurnLeftMDP) = mdp.actions
POMDPs.actionindex(mdp::SignalizedJunctionTurnLeftMDP, a) = findfirst(mdp.actions .== a)

# unchanged - not needed
disturbanceindex(mdp::SignalizedJunctionTurnLeftMDP, x) = findfirst(mdp.px.support .== x)
disturbances(mdp::SignalizedJunctionTurnLeftMDP) = mdp.px.support

# done
function isfailure(mdp::SignalizedJunctionTurnLeftMDP, s)
    return false
end

# done
function POMDPs.isterminal(mdp::SignalizedJunctionTurnLeftMDP, s)
    d, v = s

    result = d <= 0.1 || v <= 0
    # println("Ego Distance: $ego_distance, Actor Distance: $actor_distance, isterminal: $result")
    # println("is terminal is $result")
    return result
end

function check_safety_condition(mdp:: SignalizedJunctionTurnLeftMDP, s)
    ego_distance, actor_distance = s
    result = !(actor_distance > mdp.safety_threshold || actor_distance <= -10 - mdp.distance_junction)
    # println("safety condition violated: $result. state: $s")
    return result
end


# unchanged - not needed
POMDPs.discount(mdp::SignalizedJunctionTurnLeftMDP) = 0.99


## Hard Coded Naive Controller Policy

struct NaiveControlPolicy <: Policy
    𝒜
    max_accel::Float64
    max_decel::Float64
    speed_limit::Float64
    detection_start::Float64
    in_junction_stop_th::Float64
    actor_d_ul::Float64
    actor_d_ll::Float64
    dt::Float64
end

function GetNaivePolicy(mdp::SignalizedJunctionTurnLeftMDP)
    return NaiveControlPolicy(mdp.actions, mdp.max_accel, mdp.max_decel, mdp.speed_limit, mdp.in_junction_stop_th, mdp.start_detect_ego, mdp.actor_collision_threshold, mdp.dt)
end

function POMDPs.action(Policy:: NaiveControlPolicy, s)
    d_ego, v_ego, d_actor, a_detected = s
    if d_ego > 0 && d_ego <= Policy.detection_start
        if a_detected == 1 && (d_actor <= Policy.actor_d_ul && d_actor >= Policy.actor_d_ll)
            # decel to stop at junction
            return required_decel_to_stop_in_d(Policy, d_ego, v_ego)
        else
            # accel or decel to meet speed limit
            return accel_to_meet_speed_limit(Policy, v_ego)
        end
    elseif d_ego <= 0 && d_ego > Policy.in_junction_stop_th
        if a_detected == 1 && (d_actor <= Policy.actor_d_ul && d_actor >= Policy.actor_d_ll)
            # max decel
            return Policy.max_decel
        else
            # accel or decel to meet speed limit
            return accel_to_meet_speed_limit(Policy, v_ego)
        end
    else
        #accel or decel to meet speed limit
        return accel_to_meet_speed_limit(Policy, v_ego)
    end
    
end

function required_decel_to_stop_in_d(policy::NaiveControlPolicy, d, v)
    if d >=0 && d <= eps()
        return policy.max_decel
    end
    return max(policy.max_decel, -v^2/(2*d))
end

function accel_to_meet_speed_limit(Policy::NaiveControlPolicy, v)
    return min(Policy.max_accel, max(Policy.max_decel, (Policy.speed_limit - v) / Policy.dt))
end
