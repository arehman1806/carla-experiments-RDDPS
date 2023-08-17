using Distributions, Parameters, Random
using POMDPTools, POMDPGym, POMDPs

@with_kw struct SignalizedJunctionTurnLeftMDP <: MDP{Array{Float32},Float32}
    junction_end_ego::Float64 = -20
    junction_end_actor::Float64 = -15
    start_detect_ego = 100
    collision_d_ul = 2
    collision_d_ll = -10 # if actor is in the junction until point when ego leaves, a collision has occured
    in_junction_stop_th::Float64 = -8 # max distance ego can stop inside the junction
    speed_limit::Float64 = 40 # applies to both ego and actor
    max_accel::Float64 = 4.6
    max_decel::Float64 = -4.6
    actions = [0, 1]
    reward_safety_violation = -100
    dt = 0.1 # the time step
    d_ego0 = Distributions.Uniform(10, 300)
    v_ego0 = Distributions.Uniform(10, 40) # ego initial velocity
    d_actor0 = Distributions.Uniform(15, 200)
    detected0 = Deterministic(0)

end

# unchanged - not needed
function POMDPs.gen(mdp::SignalizedJunctionTurnLeftMDP, s, a, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    t = transition(mdp, s, a, x)
    (sp=rand(t), r=reward(mdp, s, a))
end

function POMDPs.transition(mdp::SignalizedJunctionTurnLeftMDP, s, a_required, x, rng::AbstractRNG=Random.GLOBAL_RNG)
    detected = x != 0
    d_ego, v_ego, d_actor, a_detected = s
    v_ego_next = max((v_ego + a_required * mdp.dt), 0) # its speed not vel so cant get below 0
    d_ego_next = d_ego - (v_ego * mdp.dt + (0.5 * a_required * mdp.dt^2))
    d_actor_next = d_actor - mdp.speed_limit*mdp.dt
    # println("d_ego: $d_ego, v_ego: $v_ego, d_actor: $d_actor, accel: $a_required, d_ego_next: $d_ego_next, v_ego_next: $v_ego_next, detected: $detected")
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
    ImplicitDistribution((rng) -> Float32[rand(mdp.d_ego0), rand(mdp.v_ego0), rand(mdp.d_actor0), rand(mdp.detected0)])
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
    d_ego, v_ego, d_actor, a_detected = s
    return d_ego <= mdp.junction_end_ego
end

function check_safety_condition(mdp:: SignalizedJunctionTurnLeftMDP, s)
    d_ego, v_ego, d_actor, a_detected = s
    return d_actor > mdp.collision_d_ll && d_actor < mdp.collision_d_ul
end

function is_inside_junction(mdp::SignalizedJunctionTurnLeftMDP, s)
    d_ego, v_ego, d_actor, a_detected = s
    return d_ego <=0 && d_ego >= mdp.junction_end_ego
end

# unchanged - not needed
POMDPs.discount(mdp::SignalizedJunctionTurnLeftMDP) = 0.99


## Hard Coded Naive Controller Policy

struct NaiveControlPolicy <: Policy
    𝒜
    max_accel::Float64
    max_decel::Float64
    speed_limit::Float64
    in_junction_stop_th::Float64
    start_detection_d::Float64
    actor_d_ul::Float64
    actor_d_ll::Float64
    dt::Float64
end

function GetNaivePolicy(mdp::SignalizedJunctionTurnLeftMDP)
    return NaiveControlPolicy(mdp.actions, mdp.max_accel, mdp.max_decel, mdp.speed_limit, mdp.in_junction_stop_th, mdp.start_detect_ego, mdp.collision_d_ul, mdp.collision_d_ll, mdp.dt)
end

function POMDPs.action(Policy:: NaiveControlPolicy, s)
    d_ego, v_ego, d_actor, a_detected = s
    if d_ego > 0 && d_actor <= Policy.start_detection_d
        if a_detected == 1 && is_on_collision_course(Policy, s)
            # decel to stop at junction
            return required_decel_to_stop_in_d(Policy, d_ego, v_ego)
        else
            # accel or decel to meet speed limit
            return accel_to_meet_speed_limit(Policy, v_ego)
        end

    elseif d_ego <= 0 && d_ego > Policy.in_junction_stop_th
        if a_detected == 1 && is_on_collision_course(Policy, s)
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

function is_on_collision_course(Policy::NaiveControlPolicy, s)
    d_ego, v_ego, d_actor, a_detected = s
    t = abs(-20 - d_ego) / Policy.speed_limit
    d_actor_final = d_actor - Policy.speed_limit * t

    # println("d_actor_final is $d_actor_final while ul is $(Policy.actor_d_ul) and ll is $(Policy.actor_d_ll)")
    return d_actor_final >= Policy.actor_d_ll && d_actor_final <= Policy.actor_d_ul
end
