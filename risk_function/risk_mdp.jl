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


using POMDPGym, POMDPs, POMDPTools
using Parameters, Random

@with_kw mutable struct RMDP{S,A} <: MDP{S,A}
    amdp::MDP{S,A}
    π
    cost_fn = POMDPS.reward
    include_time_in_state = false
    dt = 0.1
    maxT = 100*dt
    disturbance_type=:arg #:arg if passed as argument, :noise if used as noise on the state
end

function action_s_s′(mdp::RMDP, s, s′)
    return action(mdp.π, s)
end

function POMDPs.initialstate(mdp::RMDP)
    s0 = initialstate(mdp.amdp)
    if mdp.include_time_in_state
        return ImplicitDistribution((rng) -> [0, rand(s0)...])
    else
        return s0
    end
end

get_s(mdp::RMDP, s) = mdp.include_time_in_state ? s[2:end] : s
    

POMDPs.actions(mdp::RMDP) = disturbances(mdp.amdp)
POMDPs.actionindex(mdp::RMDP, a) = disturbanceindex(mdp.amdp, a)

function POMDPs.isterminal(mdp::RMDP, s)
    isterm = isterminal(mdp.amdp, get_s(mdp, s))
    if mdp.include_time_in_state
        isterm = isterm || (s[1] > (mdp.maxT + mdp.dt/2))
    end
    isterm
end

function check_safety_condition(mdp:: RMDP, s)
    result = check_safety_condition(mdp.amdp, get_s(mdp, s))
    return result
end

function check_violation_extent(mdp:: RMDP, s)
    result = check_violation_extent(mdp.amdp, get_s(mdp, s))
    return result
end

function is_inside_junction(mdp::RMDP, s)
    return is_inside_junction(mdp.amdp, get_s(mdp, s))
end

function isfailure(mdp::RMDP, s)
    isfailure(mdp.amdp, get_s(mdp,s))
end
    
POMDPs.discount(mdp::RMDP) = 1f0

render(mdp::RMDP, s, a = nothing; kwargs...) = render(mdp.amdp, get_s(mdp,s), a; kwargs...)
render(mdp::RMDP; kwargs...) = render(mdp.amdp; kwargs...)

function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    if mdp.disturbance_type == :arg
        sp, r = gen(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
    elseif mdp.disturbance_type == :noise
        sp, r = gen(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
    elseif mdp.disturbance_type == :both
        sp, r = gen(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
    else
        @error "Unrecognized disturbance type $(mdp.disturbance_type)"
    end
        
    if mdp.include_time_in_state
        t = s[1]
        sp = [t+mdp.dt, sp...]
    end
    (sp=sp, r=mdp.cost_fn(mdp, s, sp))
end

function POMDPs.transition(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    if mdp.disturbance_type == :arg
        t = transition(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s)), x, rng; kwargs...)
    elseif mdp.disturbance_type == :noise
        t = transition(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x), rng; kwargs...)
    elseif mdp.disturbance_type == :both
        t = transition(mdp.amdp, get_s(mdp, s), action(mdp.π, get_s(mdp, s) .+ x[2:end]), x[1], rng; kwargs...)
    else
        @error "Unrecognized disturbance type $(mdp.disturbance_type)"
    end
    t
end

POMDPs.reward(mdp::RMDP, s, sp) = mdp.cost_fn(mdp.amdp, s, sp)