# Calculate Risk
using POMDPs, POMDPGym, Crux, Flux, Distributions, BSON, GridInterpolations
using DataFrames, LinearAlgebra, CSV
using StatsBase
using DelimitedFiles
using ONNX
using Images
import Umlaut: Tape, play!
include("./risk_solvers.jl")
include("./sjtl_mdp.jl")

s2pt(s) = s

function get_CVaR_items(ego_junction_end, actor_junction_end)
    # Load the environment and policy
    println("Loading environment and getting optimal policy...")
    # ego_junction_end = -20
    # actor_junction_end = -15
    start_detect_ego = 100

    env = SignalizedJunctionTurnLeftMDP(junction_end_ego=ego_junction_end, junction_end_actor=actor_junction_end, start_detect_ego=start_detect_ego, collision_d_ul = 2, collision_d_ll = -10, 
                                        collision_d = -1, in_junction_stop_th=-8, speed_limit=25, max_accel=4.6, max_decel=-4.6, dt=0.01,
                                        d_ego0=Distributions.Uniform(30, 100), v_ego0=Distributions.Uniform(5, 40), d_actor0=Distributions.Uniform(10, 100))


    ds_ego = sort(collect(range(50, (ego_junction_end - 1), 70)))
    vs_ego = sort(collect(range(25, 0, 26)))
    ds_actor = sort(collect(range(50, actor_junction_end - 10, 65)))
    detects = [0, 1]

    policy = GetNaivePolicy(env)

    # Set up the cost function and risk mdp
    println("Setting up cost function and solving for risk...")

    function costfn(m, s, sp)
        d_actor, v_ego, d_ego = sp
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

    function surrogate_model_pass(tape::Tape, sample_vector::Array{Float64,1})
        # Process the input data
        x = reshape(sample_vector, length(sample_vector), 1)
        y = play!(tape, x)
        return y
    end
    path = "surrogate_model.onnx"
    sample_input = rand(Float64, 3)
    println("Loading the model")

    surrogate_model = ONNX.load(path, sample_input)
    p_detect(s) = surrogate_model_pass(surrogate_model, [s[3], s[2], s[1]])[1]
    # p_detect(s) = 0

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

    # Set up cost points, state grid, and other necessary data
    cost_points = collect(range(0, 100, 11))
    s_grid = RectangleGrid(ds_actor, vs_ego, ds_ego)
    𝒮 = [[d_actor, v_ego, d_ego] for d_actor in ds_actor, v_ego in vs_ego, d_ego in ds_ego];
    s2pt(s) = s

    # # Solve for distribution over costs
    @time Uw, Qw = solve_cvar_fixed_particle(rmdp, px, s_grid, 𝒮, s2pt,
        cost_points, mdp_type=:exp);


        return s_grid, ϵ_grid, Qw, cost_points, px, policy, (s, ϵ, α) -> CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

end

# s_grid, ϵ_grid, Qw, cost_points, px, policy = get_CVaR_items()
# CVaR(s, ϵ, α) = CVaR(s, ϵ, s_grid, ϵ_grid, Qw, cost_points; alphaa=α)

# Start labeling the data
state_data_file = "./dataset/states.csv"
df = DataFrame(CSV.File(state_data_file))
n_rows = nrow(df)
# Group the dataframe by ego_junction_end and actor_junction_end
grouped_df = groupby(df, [:ego_junction_distance, :actor_junction_distance])


training_dir = "./dataset/labels/train"
files_in_training_dir = Set([replace(f, ".txt" => "") for f in readdir(training_dir)])

val_dir = "./dataset/labels/val"
files_in_val_dir = Set([replace(f, ".txt" => "") for f in readdir(val_dir)])

for group in grouped_df
    ego_end = group[1, "ego_junction_distance"]
    actor_end = group[1, "actor_junction_distance"]
    # println(group[1, :])
    # continue

    s_grid, ϵ_grid, Qw, cost_points, px, policy, cvar_func = get_CVaR_items(-ego_end, -actor_end)

    for i in 1:nrow(group)
        s = [group[i, "actor_distance"], group[i, "ego_velocity"], group[i, "ego_distance"]]
        detect_risk = round(cvar_func(s, [1], 0.9), digits=6)
        no_detect_risk = round(cvar_func(s, [0], 0.9), digits=6)

        f_id = group[i, "image_file_name"]
        if f_id in files_in_val_dir
            text_file_name = "./dataset/labels/val/$(f_id).txt"
        elseif f_id in files_in_training_dir
            text_file_name = "./dataset/labels/train/$(f_id).txt"
        else
            continue
        end

        labels = readdlm(text_file_name)
        # println(labels)
        new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) $(detect_risk) $(no_detect_risk)"
        
        io = open(text_file_name, "w")
        write(io, new_string)
        close(io)
    end
end

# # Loop through files
# for i = 1:9500
#     # Get the mdp state
#     s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
#     # Get the detect risk
#     detect_risk = round(CVaR(s, [1], 0.0), digits=6)
#     # Get the no detect risk
#     no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
#     # Get name of text file
#     fn = df[i, "filename"]
#     text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v3_rl/train/labels/$(fn).txt"
#     # Write the risks to it
#     io = open(text_file_name, "r")
#     temp = read(io, String)
#     close(io)

#     new_string = "$(temp[1:end-1]) $(detect_risk) $(no_detect_risk)"

#     io = open(text_file_name, "w")
#     write(io, new_string)
#     close(io)
# end

# # Loop through files
# for i = 9501:10000
#     # Get the mdp state
#     s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
#     # Get the detect risk
#     detect_risk = round(CVaR(s, [1], 0.0), digits=6)
#     # Get the no detect risk
#     no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
#     # Get name of text file
#     fn = df[i, "filename"]
#     text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v1_rl/valid/labels/$(fn).txt"
#     # Write the risks to it
#     io = open(text_file_name, "r")
#     temp = read(io, String)
#     close(io)

#     new_string = "$(temp[1:end-1]) $(detect_risk) $(no_detect_risk)"

#     io = open(text_file_name, "w")
#     write(io, new_string)
#     close(io)
# end

# """
# Overwrite Existing
# """

# 

# text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v1_rl/train/labels/0.txt"
# io = open(text_file_name, "r")
# temp = read(io, String)
# close(io)

# labels = readdlm(text_file_name)
# new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) 0 0"

# # Loop through files
# for i = 1:9500
#     # Get the mdp state
#     s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
#     # Get the detect risk
#     detect_risk = round(CVaR(s, [1], 0.0), digits=6)
#     # Get the no detect risk
#     no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
#     # Get name of text file
#     fn = df[i, "filename"]
#     text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v3_rl/train/labels/$(fn).txt"
#     # Write the risks to it
#     labels = readdlm(text_file_name)
#     new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) $(detect_risk) $(no_detect_risk)"

#     io = open(text_file_name, "w")
#     write(io, new_string)
#     close(io)
# end

# # Loop through files
# for i = 9501:10000
#     # Get the mdp state
#     s = mdp_state(df[i, "e0"], df[i, "n0"], df[i, "u0"], df[i, "e1"], df[i, "n1"], df[i, "u1"])
#     # Get the detect risk
#     detect_risk = round(CVaR(s, [1], 0.0), digits=6)
#     # Get the no detect risk
#     no_detect_risk = round(CVaR(s, [0], 0.0), digits=6)
#     # Get name of text file
#     fn = df[i, "filename"]
#     text_file_name = "/scratch/smkatz/yolo_data/yolo_format/uniform_v1_rl/valid/labels/$(fn).txt"
#     # Write the risks to it
#     labels = readdlm(text_file_name)
#     new_string = "$(labels[1]) $(labels[2]) $(labels[3]) $(labels[4]) $(labels[5]) $(detect_risk) $(no_detect_risk)"

#     io = open(text_file_name, "w")
#     write(io, new_string)
#     close(io)
# end