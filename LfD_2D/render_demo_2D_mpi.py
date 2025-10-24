import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

# from joblib import Parallel, delayed

import pickle
import h5py
import re
import os
import argparse
import json
import traceback

#=================================
# PARAMETERS
#=================================

DEBUG=False 
class Params:
    dataset_type = "single_render"
    dataset_name = "dynamic"                #  one of ["osa", "static", "dynamic", "only_close", "dynamic_accel"], accepted by LfD dataloader
    LfH_dir = "10obs_TS2+_lammr0_model_3090_drop_FIXED"
    traj_length_in_time = 1.0               # UNUSED generally set --seq_len_idx argument -> L=5 unused now
    index_from_sample = 0                   # 0 means sample everything   -> for DEBUGGING
    index_upto_sample = 10000000            # large number means sample everything
    downsample = 4                          # 1 = no downsampling
    demo_dir_suffix = "_4d_20u_cG_0velTime_12Vel"
    calculate_goals_from_traj = True        # pretty much fixed
    laser_max_range = 5.0                   # pretty much fixed

    # feeding in previous commands to LfD!
    commands_as_inputs = True              # UNUSED generate data as autoregressive for LfD training
    cmd_prediction = 1                     # UNUSED cmd_prediction includes current + future commands. so it has to be >= 1. if >1 then it will predict future commands., unused now

    # OSA:
    only_noobs = False                       # to make a seperate open space augmentation dataset where there are no obstacles in the entire dataset
    osa_cosine_thresh = -1                  # if cosine(cmd, goal) > osa_cosine_thresh, osa_cosine_thresh_proportion% of the trajectory, then mark the trajectory as no_obstactle_trajectory (ie. it is part of OSA)
    osa_cosine_thresh_proportion = -1
    # osa_linear_vel_thresh = 0.7             # in addition to the cosine requirement, now the mean linear velocity of the trajectory needs to be > osa_linear_vel_thresh

    # for generating obstacle trajectories
    min_valid_samples_per_obs = 20
    clearance = 0.15                        # static obstacle clearance scale = 0.15, dynamic obstacle clearance scale = 0.15 (DAGGER - was 0.3), fixed
    max_clearance = -1                      # if max_cearance > -1, then obstacles have to be this close at least once in their trajectory. max_clearance > -1 will override clearance, for only_close dataset

    obs_vel_min = 1
    obs_vel_max = 2
    obs_fixed_vel = 0                     # if fixed_vel, vel ~ Uniform(Circle of radius fixed_vel). stdDev ignored
    obs_vel_stddev = 0                      # dynamic obstacle vel_stdDev = 1 static = 0, onlyClose = 2
    obs_acc_stddev =  0                   # dynamic obstacle acc_stdDev = 1 for dynamic_accel, rest = 0
    obs_jerk_stddev = 0                     # dynamic obstacle jerk_stdDev = 1 for dynamic_jerk, rest = 0
    S_samples = 100                        # for every critical point/robot trajectory pair, sample S_samples obstacle trajectories.
    max_sample_trials = 10                   # try for max_sample_trials before giving up on finding valid obstacle trajectories.

    # for random obstable generation
    min_additional_obs = -1                  # add. static obstacle = [10,20], add. dynamic obstacles = [5,10], onlyClose = [-1,-1]
    max_additional_obs = -1
    loc_radius = 3.0                        # add. static obstacle radius = 2.0, add. dynamic obstacle radius = 3
    ro_vel_min = 1                         # add. dynamic obstacle vel max = +-2, accel: +-2, same as dynamic_vel 1e-5 for static
    ro_vel_max = 2                         # add. dynamic obstacle vel max = +-2, accel: +-2, same as dynamic_vel 1e-5 for static
    random_obs_clearance = 0.3              # clearance for random obstacles, same as 
    # size_min, size_max = 0.2, 0.5
    max_linear_velocity = -1                # if -1 then no max, linear vel. else sqrt(vx**2 + vy**2) = max_linear_velocity. For random Obs Augmentation
    
    # obstacle collision checking:
    vel_time = 0.0

    # unchanged below
    vel_forward_span = 120
    n_pts_to_consider = 10 # for plotting - unused
    gen_obst_size = 0.5
    loc_span = 270

    # pretty much constant params:::
    # for laser scan rendering
    laser_span = 270
    n_scan = 720
    plot_freq = -1

    #dynamic LfLH params -> usually unchanged
    odom_freq = 50
    laser_scan_freq = 10
    cmd_vel_freq = 10
    split_demo = True
    debug_data = True
    drop_obstacle_perc = 0              # unused
    traj_knot_start = -3                # Todo: change to load from eval
    traj_knot_end = 35                  # Todo: change to load from eval

    save_to_scratch = True
    load_from_scratch = True
    n_render_per_hallucination = 1

    # post process params
    loc_span = loc_span * np.pi / 180
    vel_forward_span = vel_forward_span * np.pi / 180
    laser_span = laser_span * np.pi / 180


#=================================
# Helper methods
#=================================
# rotates xy coordinates around origin - for rendering laser scans 
def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = np.take(xy, 0, -1), np.take(xy, 1, -1)
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    return np.stack([xx, yy], axis=-1)
# converts quaternion to radians
def quat_to_psi(ori):
    q1, q2, q3, q0 = ori[...,0], ori[...,1], ori[...,2], ori[...,3]
    PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
    return PSI
# for generating obstacle trajectories
def setup_timesteps_traj(knot_start=-3, knot_end=18):
    ref_pt_ts = np.arange(knot_start+2,  knot_end -2)* 0.15
    reference_pt_idx = np.round(ref_pt_ts * 50).astype(int)

    ref_pt_idx0 = reference_pt_idx - reference_pt_idx[0]
    full_traj_ts = []
    for i in range(len(ref_pt_idx0) -2):
        x = np.linspace(ref_pt_ts[i], ref_pt_ts[i+1],ref_pt_idx0[i+1]-ref_pt_idx0[i] +1)
        full_traj_ts.extend(x[:-1])
    x = np.linspace(ref_pt_ts[len(ref_pt_idx0) -2], 
                    ref_pt_ts[len(ref_pt_idx0) -1], 
                    ref_pt_idx0[len(ref_pt_idx0) -1] -ref_pt_idx0[len(ref_pt_idx0) -2] +1)
    full_traj_ts.extend(x)
    full_traj_ts = np.array(full_traj_ts)
    traj_start = reference_pt_idx[1] - reference_pt_idx[0] # 8
    traj_end = reference_pt_idx[-2] - reference_pt_idx[0] + 1   # +1: for exclusion # 114
    timesteps_traj = full_traj_ts[traj_start: traj_end]
    full_traj_ref_pt_ts = full_traj_ts[reference_pt_idx - reference_pt_idx[0]]

    return reference_pt_idx, timesteps_traj, full_traj_ts, full_traj_ref_pt_ts
# for aggregating data
def reshape_obstacle_trajectory(sampled_obs_traj_to_add, total_obs):
    num_trajs, num_samples, num_obs, traj_len, Dy = sampled_obs_traj_to_add.shape
    sample_obs = np.ones((num_trajs, num_samples, total_obs, traj_len, Dy)) * 1e+10
    sample_obs[:,:, :num_obs] = sampled_obs_traj_to_add
    return sample_obs
# for aggregating data
def reshape_sequenced_obstacle_trajectory(sampled_obs_traj_to_add, total_obs):
    num_trajs, num_samples, num_sequences, num_obs, traj_len, Dy = sampled_obs_traj_to_add.shape
    sample_obs = np.ones((num_trajs, num_samples, num_sequences, total_obs, traj_len, Dy)) * 1e+10
    sample_obs[:, :, :, :num_obs] = sampled_obs_traj_to_add
    return sample_obs
# converts cmd's linear & angular velocity to a euclidean vector
def velocity_to_euclidean_vector(v,w,theta, dt=0.1):
    """
    Convert (v, w) to a direction vector considering angular velocity over time.

    Parameters:
        v (float): Linear velocity (m/s)
        w (float): Angular velocity (rad/s)
        theta (float): Initial heading angle (radians)
        dt (float): Time step (seconds, default 1.0)

    Returns:
        (float, float, float): New (x->, y->, updated theta)
    """
    theta_new = theta + w * dt  # Update heading based on angular velocity
    x_vec = v * np.cos(theta_new)
    y_vec = v * np.sin(theta_new)
    return x_vec, y_vec
# calculate 
def calculate_correct_goal(given_goal, traj, raw_ori, params):
    if params.calculate_goals_from_traj:
        psis = quat_to_psi(raw_ori)
        psi_diff_ = psis - psis[0]                                                                  # the difference between psi @ robot position and psi @ traj start

        calculated_goals = traj[0, -1, :2] - traj[0, :, :2]       
        calculated_goals = rotate_origin_only(calculated_goals, psi_diff_)                                                               # goal in robocentric frame. rotate around every single point using the diff. this will get the goal in 
        calculated_goals = calculated_goals / (np.linalg.norm(calculated_goals, axis=-1, keepdims=True)+1e-9)
        
        return calculated_goals
    else:
        return given_goal

# check if current trajectory should be considered for open space augmentation
def no_obst_augmentation(cmd, goal, params):
    if params.osa_cosine_thresh > 0:
        # cmd = np.array(eval_data["open_space_augmentation_correctedGoal"]["cmd"])[:,:, ::1]
        x_, y_ = velocity_to_euclidean_vector(cmd[0], cmd[1], 0, dt=1)                                          # theta = 0 -> robocentric frame
        cmd_euclidean = np.stack((x_, y_), axis=-1) 

        cosine_diff = (cmd_euclidean * goal).sum(-1) / (np.linalg.norm(cmd_euclidean, axis=-1)* np.linalg.norm(goal, axis=-1) + 1e-9)
        traj_mean_cosine_agreement = (cosine_diff > params.osa_cosine_thresh).mean(-1)
        if traj_mean_cosine_agreement > params.osa_cosine_thresh_proportion:

            return True
        
    return False

# aggregates all seperate .p files into single .h5 file
def dataset_aggregator(params, total_original_obs=10):

    total_length = 0
    try:
        demo_dir = os.path.join(params.demo_dir, "LfH_demo")

        # from seperate .p files to h5 file
        files =[f for f in  os.listdir(demo_dir) if "agg" not in f and "demo" in f]
        files.sort(key = lambda x: int(re.search(r"\d+", x)[0]))
        try:
            with open(os.path.join(demo_dir, files[0]), "rb") as f:
                first_data = pickle.load(f)
        except Exception as e1:
            print("error in loading first pickle file")
            print(traceback.format_exc())
        for k in first_data:
            first_data[k] = np.array(first_data[k])

        total_length = 0
        with h5py.File(os.path.join(params.demo_dir,"LfH_demo.h5"), "w") as h5_file:

            datasets = {k: h5_file.create_dataset(k, data=first_data[k], maxshape=tuple([None] + list(first_data[k].shape)[1:])) for k in first_data if k not in ["sd_obs_loc", "generated_obs_trajs"]}
        
            if params.debug_data and "sd_obs_loc" in first_data:
                obs_reshaped = reshape_sequenced_obstacle_trajectory(first_data["sd_obs_loc"], total_original_obs)
                datasets["sd_obs_loc"] = h5_file.create_dataset("sd_obs_loc", data=obs_reshaped, maxshape=tuple([None] + list(obs_reshaped.shape)[1:]))

            if params.debug_data and "generated_obs_trajs" in first_data:
                obs_reshaped = reshape_obstacle_trajectory(first_data["generated_obs_trajs"], total_original_obs)
                datasets["generated_obs_trajs"] = h5_file.create_dataset("generated_obs_trajs", data=obs_reshaped, maxshape=tuple([None] + list(obs_reshaped.shape)[1:]))
            
            for file in files[1:]:
                try:
                    with open(os.path.join(demo_dir, file), "rb") as f:
                        data = pickle.load(f)
                except Exception as e1:
                    print("error in loading pickle file "+str(file))
                    print(traceback.format_exc())
                    continue
                for k in datasets:
                    data[k] = np.array(data[k])
                    if params.debug_data and k=="sd_obs_loc":
                        obs_reshaped = reshape_sequenced_obstacle_trajectory(data[k], total_original_obs)
                        datasets[k].resize(datasets[k].shape[0] + obs_reshaped.shape[0], axis=0)
                        datasets[k][-obs_reshaped.shape[0]:] = obs_reshaped
                        continue
                    elif params.debug_data and k=="generated_obs_trajs":
                        obs_reshaped = reshape_obstacle_trajectory(data[k], total_original_obs)
                        datasets[k].resize(datasets[k].shape[0] + obs_reshaped.shape[0], axis=0)
                        datasets[k][-obs_reshaped.shape[0]:] = obs_reshaped
                        continue
                    datasets[k].resize(datasets[k].shape[0] + data[k].shape[0], axis=0)
                    datasets[k][-data[k].shape[0]:] = data[k]
            total_length = len(datasets["laser"])

    except Exception as e:
        print("error in dataset_aggregator:", e)
        print(traceback.format_exc())

    finally:
        return total_length
  

#=================================
# Main methods
#=================================
# takes in unbatched data to run - a
def demo_helper(i, raw_traj, raw_ori, traj, cmd, goal, obs_loc, obs_size, obs_time_idx, params):
    if os.path.exists(os.path.join(params.demo_dir, "LfH_demo", "LfH_demo"+str(i)+".p")):
        return 0
    
    demo_ = {"laser": [], "goal": [], "cmd": []}

    calculated_correct_goal = calculate_correct_goal(goal, traj, raw_ori, params)
    no_obs_traj = no_obst_augmentation(cmd, calculated_correct_goal, params)

    if params.only_noobs and not no_obs_traj:
        return len(demo_["laser"])
    if params.min_additional_obs > 0:
        add_obs_loc, add_obs_size = generate_random_dynamic_obs(traj, params) 
    else:
        add_obs_loc, add_obs_size = [], []
        
    output = prepare_data(raw_traj, raw_ori, traj, cmd, calculated_correct_goal, obs_loc, obs_size, obs_time_idx, add_obs_loc, add_obs_size, params, i, no_obs_traj)
    
    if isinstance(output, int):
        return len(demo_["laser"])
    elif no_obs_traj:
        seq_cmd, seq_goal, len_of_one_seq, _ = output
        laser_span_num = 720
        laser_scan = np.ones((len_of_one_seq, laser_span_num))*params.laser_max_range
        for cmd, goal in zip(seq_cmd, seq_goal):
            demo_["laser"].append(laser_scan)
            demo_["goal"].append(goal)
            demo_["cmd"].append(cmd)

        if params.debug_data:
            demo_["robot_traj"] = traj[0][None]
            demo_["sd_goal"] = seq_goal[None]
            demo_["sd_cmd"] = seq_cmd[None]

        for k in demo_:
            demo_[k] = np.array(demo_[k])
            print(k, demo_[k].shape)
            print("---------")
            
        with open(os.path.join(params.demo_dir, "LfH_demo", "LfH_demo"+str(i)+".p"), "wb") as f:
            pickle.dump(demo_, f)

        return len(demo_["laser"])
    else:
        seq_traj, seq_psi, seq_goal, seq_cmd, sampled_seq_obs_loc, seq_obs_size, seq_addobs_loc, seq_addobs_size, generated_obs_trajs = output 
    
    num_samples = sampled_seq_obs_loc.shape[0]

    psi_0 = seq_psi[0,0]
    if params.debug_data:
        demo_["sd_psi"] = seq_psi

    seq_psi = seq_psi - psi_0
    
    skipped_sequences_across_samples = []
    for si in range(num_samples):
        skipped_sequence = [False]*seq_traj.shape[0]
        break_in_laser_scan_sequence = True
        for k, (traj_, psi_, goal_, cmd_, loc, size, addloc, addsize) in enumerate(zip(seq_traj, seq_psi, seq_goal, seq_cmd, sampled_seq_obs_loc[si], seq_obs_size, seq_addobs_loc, seq_addobs_size)):
            for j in range(params.n_render_per_hallucination):
                if k==0 or break_in_laser_scan_sequence:
                    laser_scan = render_laser_scan(True, traj_, psi_, loc, size, addloc, addsize, params)
                    break_in_laser_scan_sequence = False
                else:
                    laser_scan = render_laser_scan(False, traj_, psi_, loc, size, addloc, addsize, params)
                    laser_scan = np.concatenate((demo_["laser"][-1][1:], laser_scan), axis=0)

                if (laser_scan == params.laser_max_range).all():    # obstacles are too far
                    skipped_sequence[k]=True
                    break_in_laser_scan_sequence = True
                    continue                                        # skip as ambiguous 
                
                demo_["laser"].append(laser_scan)
                demo_["goal"].append(goal_)
                demo_["cmd"].append(cmd_)
        skipped_sequences_across_samples.append(skipped_sequence)
                
    if params.debug_data:
        demo_["sd_traj"] = seq_traj
        demo_["robot_traj"] = traj[0]
        demo_["skipped_sequences"] = skipped_sequences_across_samples
        # demo_["sd_goal"] = seq_goal
        # demo_["sd_cmd"] = seq_cmd
        demo_["sd_obs_loc"] = sampled_seq_obs_loc
        # demo_["sd_obs_size"] = seq_obs_size
        # demo_["sd_addloc"] = seq_addobs_loc
        # demo_["sd_addsize"] = seq_addobs_size
        demo_["generated_obs_trajs"] = generated_obs_trajs
    
    for k in demo_:
        if k not in ["laser","cmd", "goal"]:
            demo_[k] = np.array(demo_[k])[None]

    if params.split_demo:
        with open(os.path.join(params.demo_dir, "LfH_demo", "LfH_demo"+str(i)+".p"), "wb") as f:
            pickle.dump(demo_, f)
        
        return len(demo_["laser"])
    else:
        return len(demo_["laser"])
    
# prepare sequence data - ab
def prepare_data(raw_traj, raw_ori, traj, cmd, goal, obs_loc, obs_size, obs_time_idx, add_obs_loc, add_obs_size, params, batch_i, no_obs_traj=False):
    """
    removing the time dimension for all things
    :param raw_traj     (2, 106, 3)                     2 - loc,vel. 3 - x,y,z
    :param raw_ori      (106, 4)                        4 - quaternion orientation
    :param traj         (2, 106, 3) / (B, 2, 106, 3)    2 - loc,vel 3 - x,y,t
    :param cmd          (2, 106) / (B, 2, 106)          2 - linear vel. angular vel.
    :param goal         (106, 2) / (B, 106, 2)
    :param obs_loc      (N, 2)                          N - |obst| 
    :param obs_size     (N, 2) / (B, N, 2)
    :param obs_vel      (N, 2) / (B, N, 2) or (N,) / (B, N,) if vx only
    :param obs_acc      (N, 2) or None
    
    # returns (L, seq_len, Dy) / (B, L, seq_len, Dy)                  for trajs, goals
    # returns (L, 2, seq_len) / (B, L, 2, seq_len)                    for cmd (2- ang. and linear vel)
    # returns (L, n_obs, seq_len, Dy) / (B, L, n_obs, seq_len, Dy)    for obs_loc, obs_size
    # returns (L, seq_len)                                            for orientation

    """

    laser_scan_freq = params.laser_scan_freq
    odom_freq = params.odom_freq
    cmd_vel_freq = params.cmd_vel_freq
    # max_linear_velocity_sq = -1 #if params.max_linear_velocity < 0 else  params.max_linear_velocity**2

    sequenced_trajs = [] 
    sequenced_goals = []
    sequenced_cmds = []
    # sequenced_obs_locs = []
    sequenced_obs_sizes = []
    sequences_psis = []
    sequenced_add_obs_locs = []
    sequenced_add_obs_sizes = []
    sequenced_sampled_obs_trajs = []

    if cmd_vel_freq <= odom_freq:
        di = laser_scan_freq/cmd_vel_freq
        # pass
    else:
        print(f"desired cmd_vel freq too high. max. freq is odom_freq of {odom_freq}")
        di = laser_scan_freq/odom_freq

    if params.dataset_type == "single_render":
        # honestly this is very easy. just find the odom idxs once and get the data of those indices and return. No need for di or anything.
        L_sequences = 1
        len_of_one_seq = int(raw_traj.shape[-2]*laser_scan_freq/params.odom_freq)+1
        odom_idxs = np.arange(0, len_of_one_seq)*params.odom_freq/laser_scan_freq
        odom_idxs = odom_idxs.astype(int)
        odom_idxs = odom_idxs[None]
        sequenced_cmds.append(np.transpose(cmd[:, odom_idxs].squeeze(), (1,0)))
    else:
        len_of_one_seq = max(int(laser_scan_freq*params.traj_length_in_time), 1)
        total_required_len_in_timesteps = len_of_one_seq
        future_cmd_horizon = 0
        if params.cmd_prediction > 1:
            # The horizon in timesteps is the number of predictions times the step interval
            future_cmd_horizon = (params.cmd_prediction -1) * di
            
            # Calculate the total length needed for one full sample
            total_required_len_in_timesteps += future_cmd_horizon

        # Calculate the number of valid sequences that can be extracted
        L_sequences = int(np.floor((raw_traj.shape[-2]/odom_freq - total_required_len_in_timesteps/laser_scan_freq)*params.cmd_vel_freq)+1) +1 -1 # +1 because the first position is 0 -1 because this would go out of range due to odom_idxs including index 106. also, makes sense.
    
        odom_idxs = []
        for i in range(int(1/di), L_sequences):
            start = i*di
            idxs = np.arange(start, start+len_of_one_seq)*params.odom_freq/laser_scan_freq
            idxs = idxs.astype(int)
            odom_idxs.append(idxs)
            if params.commands_as_inputs:
                # Adjust the end of the arange to include the future commands
                cmd_idxs_end = start + len_of_one_seq + future_cmd_horizon
                cmd_idxs = (np.arange(start-1, cmd_idxs_end)*params.odom_freq/laser_scan_freq).astype(int)
                sequenced_cmds.append(np.transpose(cmd[:, cmd_idxs], (1,0)))
            else:
                sequenced_cmds.append(cmd[:, idxs[-1]])
        odom_idxs = np.array(odom_idxs).astype(int)

    psis = quat_to_psi(raw_ori)
    sequences_psis = psis[odom_idxs]

    sequenced_cmds = np.array(sequenced_cmds)

    # goal is incorrect. goal needs to be dynamically set. so it should just be the goal[odom_idxs, :]
    sequenced_goals = goal[odom_idxs, :]
    if no_obs_traj:
        return sequenced_cmds, sequenced_goals, len_of_one_seq, L_sequences

    generated_obstacle_trajectories = generate_obstacle_trajectories(obs_loc, obs_time_idx, traj, params, batch_i)
    if isinstance(generated_obstacle_trajectories, int) and generated_obstacle_trajectories < 0:
        return -1

    sequenced_trajs = np.stack((traj[0,odom_idxs,:2], traj[1,odom_idxs,:2]), axis=1)

    for idxs in odom_idxs:
        sequenced_sampled_obs_trajs.append(generated_obstacle_trajectories[:, :, idxs])
    sequenced_sampled_obs_trajs = np.stack(sequenced_sampled_obs_trajs, axis=1)

    sequenced_obs_sizes = (obs_size[None].repeat(L_sequences, axis=0)[:, :, None]).repeat(len_of_one_seq, axis=2) 

    if len(add_obs_loc) > 0:
        # raise NotImplementedError(":D")
        seq_add_obs_loc = np.transpose(add_obs_loc[:, odom_idxs], axes=(1,0,2,3))
        seq_add_obs_size = np.transpose(add_obs_size[:, odom_idxs], axes=(1,0,2,3))
    else:
        seq_add_obs_loc = np.array([None]*L_sequences)
        seq_add_obs_size = np.array([None]*L_sequences)

    return sequenced_trajs, sequences_psis, sequenced_goals, sequenced_cmds, sequenced_sampled_obs_trajs, sequenced_obs_sizes, seq_add_obs_loc, seq_add_obs_size, generated_obstacle_trajectories

# generate obstacle trajectories (need to move to eval) - abc
def generate_obstacle_trajectories(locs, time_idxs, traj, params, batch_i):
    reference_pt_idx, traj_ts, fulltraj_ts, fulltraj_ref_pt_ts = setup_timesteps_traj(params.traj_knot_start, params.traj_knot_end)
    traj_reference_pt_idx = reference_pt_idx[1:-1]

    num_obs, _ = locs.shape
    vel_std = params.obs_vel_stddev
    acc_std = params.obs_acc_stddev
    jerk_std = params.obs_jerk_stddev
    S = params.S_samples 

    fulltraj_ref_pt_ts_tiled = np.tile(fulltraj_ref_pt_ts, (num_obs, 1))
    t0 = fulltraj_ref_pt_ts_tiled[time_idxs.astype(bool)]
    adjusted_timesteps_fulltraj = fulltraj_ts - t0.reshape(num_obs, 1)
    adjusted_timesteps_traj = adjusted_timesteps_fulltraj[:, reference_pt_idx[1]-reference_pt_idx[0]: reference_pt_idx[-2]-reference_pt_idx[0]+1]
    adjusted_timesteps_traj = adjusted_timesteps_traj[None, :, :, None]
    
    valid_sample = np.array([0]*num_obs)
    sample_trials = 0

    robot_poss, robot_vels = traj[:, traj_reference_pt_idx]
    
    while valid_sample.min() < params.min_valid_samples_per_obs:
        if sample_trials > params.max_sample_trials:
            failing_obs = np.where(valid_sample < params.min_valid_samples_per_obs)[0]
            print(f"{batch_i}: obstacles", *failing_obs, f"< {params.min_valid_samples_per_obs}, the min_valid_samples_per_obs.")
            return -1

        if params.obs_vel_min > 0 and params.obs_vel_max > params.obs_vel_min:
            print(f"rendering fixed speeds in [{params.obs_vel_min}, {params.obs_vel_max}].")
            theta = np.random.uniform(-np.pi, np.pi, size=(S, num_obs))
            speed = np.random.uniform(params.obs_vel_min, params.obs_vel_max, size=(S, num_obs))
            vx = np.cos(theta)*speed
            vy = np.sin(theta)*speed
            vel = np.stack((vx, vy), axis=-1)[:, :, None]

        elif params.obs_fixed_vel > 0:
            print(f"rendering fixed speeds at { params.obs_fixed_vel}.")
            theta = np.random.uniform(-np.pi, np.pi, size=(S, num_obs))
            radius = params.obs_fixed_vel
            vx = np.cos(theta)*radius
            vy = np.sin(theta)*radius
            vel = np.stack((vx, vy), axis=-1)[:, :, None]
        else:
            print(f"rendering variable speeds from N(0, {vel_std}).")
            vel = np.random.normal(scale=vel_std, size=(S, num_obs, 2))[:, :, None]
            
        acc = np.random.normal(scale=acc_std, size=(S, num_obs, 2))[:, :, None]
        jerk = np.random.normal(scale=jerk_std, size=(S, num_obs, 2))[:, :, None]
        locs_ = locs[None, :, None]

        rollout = locs_ + vel *  adjusted_timesteps_traj + acc * (adjusted_timesteps_traj**2)/2 + jerk*(adjusted_timesteps_traj**3)/6
        
        rollout_ref_pts = rollout[:,:,traj_reference_pt_idx]
        
        # ref_pts = traj[0, traj_reference_pt_idx
        obs_size = np.ones_like(rollout_ref_pts)*0.5
        
        ref_collisions = is_colliding_w_traj(robot_poss[None, None], robot_vels[None, None], rollout_ref_pts, obs_size, params.vel_time, params.vel_forward_span, params.clearance)

        if DEBUG:
            ref_collisions = np.zeros_like(ref_collisions)

        sample_idx, obs_idx = np.where(~ref_collisions)

        collision_free_samples = []
        # total_configs = 1
        for obs_i in range(num_obs):
            subsample_idx = obs_idx == obs_i
            collision_free_sample_i = rollout[sample_idx[subsample_idx], obs_idx[subsample_idx]][:params.min_valid_samples_per_obs]
            collision_free_samples.append(collision_free_sample_i)
            valid_sample[obs_i] = len(collision_free_sample_i)
        sample_trials += 1

    for collfree in collision_free_samples:
        if len(collfree) == 0:
            return -1
    all_sample_configs = np.stack(collision_free_samples, axis=1)
    return all_sample_configs

# render laser scans - ad
def render_laser_scan(start_scan, robot_traj, robot_oris, obs_loc, obs_size, add_obs_loc, add_obs_size, params):
    """
    :param obs_loc: (N, 2)
    :param obs_size: (N, 2)
    :param add_obs_loc: (M, 2)
    :param add_obs_size: (M, 2)
    :param params:
    :return:
    """
    
    laser_max_range = params.laser_max_range
    laser_span = params.laser_span
    n_scan = params.n_scan
    num_samples = obs_loc.shape[0]

    if params.min_additional_obs > 0:
        obs_loc = np.concatenate([obs_loc, add_obs_loc], axis=0)
        sizes = np.concatenate([obs_size, add_obs_size], axis=0)


        #ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 3 dimension(s) and the array at index 1 has 4 dimension(s)

    else:
        obs_loc, sizes = obs_loc, obs_size

    if start_scan:
        start = 0
    else:
        start = len(robot_traj[0])-1
    scan_sequence = []


    for t in range(start, len(robot_traj[0])):
        robot_pos = robot_traj[0, t]
        psi_diff = robot_oris[t] # robot_ori @ [0,0] (beginning of traj ori) has already been subtracted out

        locs = obs_loc[:, t]-robot_pos
        locs = rotate_origin_only(locs, psi_diff)
        psi_diff = 0 #robot_oris[t] - robot_oris[0] this is now happening outside but with odom at 0 index in R0 frame 
        robot_pos = robot_pos-robot_pos
        
        scans = []
        for theta in np.linspace(psi_diff-laser_span / 2, psi_diff+laser_span / 2, n_scan):
            scan = laser_max_range
            for loc, size in zip(locs, sizes): 
                # solve (c * l - x)^2 / a^2 + (s * l - y)^2 / b^2 = 1
                l = find_raycast(robot_pos, theta, loc, size[t]) 
                if l is not None:
                    scan = np.minimum(scan, l)
            scans.append(scan)
        scan_sequence.append(scans)

    return np.array(scan_sequence)

# calculate particular laser ray - ade
def find_raycast(pos, ang, loc, size):
    x, y = pos
    c, s = np.cos(ang), np.sin(ang)
    h, v = loc
    a, b = size
    assert a > 0, f"value of a <= 0: {a}"
    assert b > 0, f"value of b <= 0: {b}"

    A = c ** 2 / a ** 2 + s ** 2 / b ** 2
    B = (2 * c * (x - h) / a ** 2 + 2 * s * (y - v) / b ** 2)
    C = (x - h) ** 2 / a ** 2 + (y - v) ** 2 / b ** 2 - 1
    delta = B ** 2 - 4 * A * C

    assert A > 0, f"value of A <= 0: {A}"

    if delta >= 0:
        delta = np.maximum(delta, 0)                
        l1 = (-B - np.sqrt(delta)) / (2 * A)
        l2 = (-B + np.sqrt(delta)) / (2 * A)
        if l1 < 0 and l2 < 0:
            return None
        if l1 > 0 and l2 > 0:
            return np.minimum(l1, l2)
        return np.maximum(l1, l2)
    else:
        return None


#=================================
# plotting methods (unused)
#=================================
def plot_render_rslts(obs_loc, obs_size, traj, goal, cmd, add_obs_loc, add_obs_size, laser_scan, params, fig_name):
    plt.figure()

    # plot collision detection
    pos, vel = traj
    vel_span = params.vel_span
    vel_time = params.vel_time
    pt_indexes = np.round(np.linspace(0, len(pos) - 1, params.n_pts_to_consider)).astype(int)
    poses, vels = pos[pt_indexes], vel[pt_indexes]
    for pos, vel in zip(poses, vels):
        vel_norm = np.linalg.norm(vel)
        clearance = vel_norm * vel_time
        vel_ang = np.arctan2(vel[1], vel[0])
        vel_arc = np.linspace(vel_ang - vel_span / 2, vel_ang + vel_span / 2, 20)
        sector = pos + clearance * np.stack([np.cos(vel_arc), np.sin(vel_arc)], axis=1)   # (20, 2)
        sector = np.concatenate([pos[None], sector, pos[None]], axis=0)                # (22, 2)
        plt.plot(sector[:, 0], sector[:, 1], color="orange", linestyle="--")

    # plot laser scan
    laser_span = params.laser_span
    laser_angles = np.linspace(-laser_span / 2, laser_span / 2, len(laser_scan))
    for scan, theta in zip(laser_scan[::10], laser_angles[::10]):
        plt.plot([0, scan * np.cos(theta)], [0, scan * np.sin(theta)], color="cyan", alpha=0.5)

    # plot traj
    pos = traj[0]
    plt.plot(pos[:, 0], pos[:, 1], label="traj")
    cmd_vel, cmd_ang_vel = cmd[0], cmd[1]
    cmd_vel = cmd_vel * 0.4
    cmd_ang_vel = cmd_ang_vel * 0.3
    goal = goal * 0.2
    plt.arrow(0, 0, cmd_vel, 0, color="b", width=0.02, label="cmd_vel")
    plt.arrow(cmd[0] / 2.0, -cmd_ang_vel / 2.0, 0, cmd_ang_vel, color="b", width=0.02, label="cmd_ang_vel")
    plt.arrow(0, 0, goal[0], goal[1], color="g", width=0.02, label="goal")

    # plot obstacles
    obses = [Ellipse(xy=loc, width=2 * size[0], height=2 * size[1]) for loc, size in zip(obs_loc, obs_size)]
    for obs in obses:
        plt.gca().add_artist(obs)
        obs.set_alpha(0.5)
        obs.set_facecolor("red")

    add_obses = [Ellipse(xy=loc, width=2 * size[0], height=2 * size[1]) for loc, size in zip(add_obs_loc, add_obs_size)]
    for obs in add_obses:
        plt.gca().add_artist(obs)
        obs.set_alpha(0.5)
        obs.set_facecolor("violet")

    plt.legend(loc="upper left")
    plt.gca().axis("equal")
    plt.xlim([-params.laser_max_range * 1.5, params.laser_max_range * 1.5])
    plt.ylim([-params.laser_max_range * 1.5, params.laser_max_range * 1.5])

    image_dir = os.path.join(params.demo_dir, "plots")
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, fig_name))
    plt.close()


#==================================================================
# Generating Random Obstacles & Helper Methods
#==================================================================
# checks immediate collision, forward and backward clearance, and obstacle is in line-of-sight
def is_colliding_w_traj(poses, vels, loc, size, vel_time, vel_forward_span, clearance, max_clearance=-1):
    """
    :param poses: (S, N, C, 2) trajectory positions
    :param vels: (S, N, C, 2) trajectory velocities
    :param loc: (S, N, C, 2) obs location
    :param size: (S, N, C, 2) obs size
    :param vel_time, vel_span: colliding if obs within radius = vel * vel_time and angle in vel_span
    :return:
    """
    loc_x, loc_y = loc[..., 0], loc[..., 1]
    a, b = size[..., 0], size[..., 1]
    pos_x, pos_y = poses[..., 0], poses[..., 1]

    collisions = ((loc_x - pos_x) ** 2 / a ** 2 + (loc_y - pos_y) ** 2 / b ** 2 <= 1).any(-1)     #collisions with 0 clearance
    
    vel_norms = np.linalg.norm(vels, axis=-1)
    forward_clearance = vel_norms*vel_time + clearance
    backward_clearance = vel_norms*0 + clearance
    forward_clearance = np.tile(forward_clearance, (loc.shape[0], loc.shape[1], 1))
    backward_clearance = np.tile(backward_clearance, (loc.shape[0], loc.shape[1], 1))

    diff = (loc - poses)
    diff_norm = np.linalg.norm(diff, axis=-1)
    diff_dir = diff/diff_norm[..., None]
    radius = 1/np.sqrt((diff_dir**2/size**2).sum(axis=-1))

    collision_distance = diff_norm - radius             # SNC

    vel_angs = np.arctan2(vels[..., 1], vels[..., 0])
    loc_dirs = np.arctan2(loc_y - pos_y, loc_x - pos_x)


    # first split forward, and backward sections
    angle_diffs = angle_diff(loc_dirs, vel_angs)            # SNC
    forward_angles = angle_diffs < vel_forward_span / 2
    backward_angles = ~forward_angles

    if max_clearance < 0:
        # check forward collision
        forward_coll = (collision_distance <= forward_clearance) * forward_angles        # SNC
        any_ = forward_coll.any(-1)
        forward_coll = np.where(any_)
        collisions[forward_coll] = True
        
        # check backward collision
        backward_coll = (collision_distance <= backward_clearance) * backward_angles
        backward_coll = np.where(backward_coll.any(-1))
        collisions[backward_coll] = True
    else:
        obs_too_far_at_every_point = (collision_distance > max_clearance).all(-1)
        idxs = np.where(obs_too_far_at_every_point)
        collisions[idxs] = True

    # check if the angle is completely outside the vision of the robot
    completely_out_of_view = (~(angle_diffs < (270*np.pi/180)/2)).all(-1)
    collisions[completely_out_of_view] = True

    return collisions
# finding the angle between heading and obstacle direction
def angle_diff(angle1s, angle2s): 
    diffs = np.abs(angle1s - angle2s)
    whereTrue = (diffs > np.pi)
    diffs[whereTrue] = 2 * np.pi - diffs[whereTrue]
    return diffs
# for generating random additional obstacles
# this step needs improvement
def generate_random_dynamic_obs(traj, params):
    pos, vel = traj
    _, timesteps_traj, _, _ = setup_timesteps_traj(params.traj_knot_start, params.traj_knot_end)

    locs, sizes = [], []
    n_valid_obs = 0

    additional_obs = np.random.randint(params.min_additional_obs, params.max_additional_obs+1)
    # additional_obs = params.min_additional_obs
    print("augmenting by", additional_obs, "additional obs")
    while n_valid_obs < additional_obs:
        
        rad = np.random.uniform(0, params.loc_radius, size=(2*params.max_additional_obs, additional_obs-n_valid_obs)) 
        ang = np.random.uniform(-params.loc_span / 2, params.loc_span / 2, size=(2*params.max_additional_obs, additional_obs-n_valid_obs))
        loc = np.stack([np.cos(ang), np.sin(ang)], axis=-1) * rad[..., None]
        size = params.gen_obst_size #np.random.uniform(params.size_min, params.size_max, 2)

        if params.max_linear_velocity > 0:
            vx = np.random.uniform(-params.max_linear_velocity, params.max_linear_velocity, size=(2*params.max_additional_obs, additional_obs-n_valid_obs))
            vy = np.sqrt(params.max_linear_velocity**2 - vx**2)*np.random.choice([-1,1])
            obs_vel = np.stack((vx,vy), axis=-1) # (S, N, 2)
        else:
            theta = np.random.uniform(-np.pi, np.pi, size=(2*params.max_additional_obs, additional_obs-n_valid_obs))
            speed = np.random.uniform(params.ro_vel_min, params.ro_vel_max, size=(2*params.max_additional_obs, additional_obs-n_valid_obs))
            vx = np.cos(theta)*speed
            vy = np.sin(theta)*speed
            obs_vel = np.stack((vx, vy), axis=-1)
            # obs_vel = np.random.uniform(-params.obs_vel_max, params.obs_vel_max, size=(2*params.max_additional_obs, additional_obs-n_valid_obs, 2))

        temp_ = obs_vel[:, :, None] * timesteps_traj[None, None, :, None] #(S, N, T, 2)

        obs_loc = loc[:, :, None] + temp_ # (S, N, T, 2)
        obs_size = np.ones_like(obs_loc)*size
        
        ref_collisions = is_colliding_w_traj(pos[None, None], vel[None, None], obs_loc, obs_size, params.vel_time, params.vel_forward_span, params.random_obs_clearance)
        sample_idx, obs_idx = np.where(~ref_collisions)

        collision_free_obs = obs_loc[sample_idx, obs_idx][:additional_obs]
        if len(collision_free_obs) > 0:
            n_valid_obs += len(collision_free_obs)
            locs.append(collision_free_obs)
            sizes.append(np.ones_like(collision_free_obs)*params.gen_obst_size)

    return np.concatenate(locs, axis=0), np.concatenate(sizes, axis=0)


#=================================
# parse arguments
#=================================
def parse_args(params):
    seq_lens = [3,5,8,10]
    eval_dirs = []
    
    seq_choices = [i for i in range(len(seq_lens))]
    seq_choices.insert(0, -1)
    eval_choices = [i for i in range(len(eval_dirs))]
    eval_choices.insert(0, -1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len_idx", default=-1, type=int, choices=seq_choices, help=f"The Sequence length is set to {seq_lens}. seq_len_idx gets the corresponding value. Default: -1 which gets the sequence length in Params")
    parser.add_argument("--eval_dir_idx", default=-1, type=int, choices=eval_choices, help=f"The eval directory is set to {eval_dirs}. Default: -1 which gets the eval dir in Params")
    parser.add_argument("--accel_jerk", default=0, type=int, choices=[0,1,2], help=f"0 if we just want to simulate velocity, 1: +accel, 2: +jerk" )
    args = parser.parse_args()

    if args.seq_len_idx != -1:
        params.traj_length_in_time = float(seq_lens[args.seq_len_idx])/params.laser_scan_freq
    
    if args.eval_dir_idx != -1:
        params.LfH_dir = eval_dirs[args.eval_dir_idx]

    if args.accel_jerk == 1:
        params.obs_accel_stddev = 0.5
        params.dataset_name = "dynamic_accel"
        params.demo_dir_suffix += "_accel"
    elif args.accel_jerk == 2:
        params.obs_accel_stddev = 0.5
        params.obs_jerk_stddev = 0.1
        params.dataset_name = "dynamic_jerk"
        params.demo_dir_suffix += "_jerk"

    return args



if __name__ == '__main__':
    params = Params()
    args = parse_args(params)

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sub_dir_folder = "single_render" if params.dataset_type == "single_render" else str(int(params.traj_length_in_time*params.laser_scan_freq))+"_seq"

    folder_name = params.LfH_dir if not params.only_noobs else "open_space_augmentation"
    folder_name = folder_name + params.demo_dir_suffix
    
    if params.save_to_scratch:
        params.demo_dir = demo_dir = os.path.join("/scratch",os.getenv("USER"),"hallucination_new", "LfH_demo", sub_dir_folder, folder_name)
    else:
        params.demo_dir = demo_dir = os.path.join(repo_path, "LfH_demo", sub_dir_folder, folder_name+"_"+str(params.traj_length_in_time))

    if params.load_from_scratch:
        LfH_dir = os.path.join("/scratch", os.getenv("USER"), "hallucination_new", "LfH_eval", params.LfH_dir)
    else:
        LfH_dir = os.path.join(repo_path, "LfH_eval", params.LfH_dir)
    
    os.makedirs(os.path.join(demo_dir, "LfH_demo"), exist_ok=True)
    shutil.copyfile(os.path.join(LfH_dir, "params.json"), os.path.join(demo_dir, "eval_params.json"))
    shutil.copyfile(os.path.join(LfH_dir, "LfH_eval.p"), os.path.join(demo_dir, "LfH_eval.p"))
    shutil.copyfile(os.path.join(LfH_dir, "model"), os.path.join(demo_dir, "model"))
    render_params_dict = {k:v for k,v in dict(Params.__dict__).items() if "__" not in k}
    render_params_dict.update({k:v for k,v in dict(params.__dict__).items() if "__" not in k})
    
    with open(os.path.join(LfH_dir, "LfH_eval.p"), "rb") as f:
        data = pickle.load(f)

    print("Params:")
    for k in render_params_dict.keys():
        print(k, render_params_dict[k])
    print("----------")

    print("loaded data from LfH_dir:", LfH_dir)
    for key in data.keys():
        data[key] = data[key][params.index_from_sample:params.index_upto_sample:params.downsample]
        print(f"{key}: {len(data[key])}")

    num_trajectory_samples = len(data["cmd"])
    print("Total samples: {}".format(num_trajectory_samples))


    if not params.split_demo:
        raise Exception("params.split_demo = False No longer supported")

    task_args = [
        (i, raw_traj_, raw_ori_, traj_, cmd_, goal_, obs_loc_, obs_size_, obs_time_idx_, params)
        for i, (raw_traj_, raw_ori_, traj_, cmd_, goal_, obs_loc_, obs_size_, obs_time_idx_) 
        in enumerate(zip(data["raw_traj"], data["raw_ori"], data["traj"], data["cmd"], data["goal"], data["obs_loc"], data["obs_size"], data["obs_time_idx"]))
    ]
    # if DEBUG:
    #     sample_lengths = []
    #     for args in task_args:
    #         sample_lengths.append(demo_helper(*args))
    # else:
    with MPIPoolExecutor() as executor:
        sample_lengths = executor.map(lambda args: demo_helper(*args), task_args)


    print("sample lengths returned from MPIPoolExecutor:",*sample_lengths)
    total_original_obs = 0
    with open(os.path.join(LfH_dir, "params.json")) as f:
        eval_params = json.load(f)
        total_original_obs = eval_params["model_params"]["num_obs"]

    count = dataset_aggregator(params, total_original_obs)

    render_params_dict["dataset_size"] = count    
    print("dataset size:", count)

    json.dump(render_params_dict, open(os.path.join(demo_dir, "render_params.json"), "w"), indent=4)
    print("rendering complete")


