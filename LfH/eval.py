import os
import json
import pickle
import shutil
import numpy as np
from joblib import Parallel, delayed

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import TrainingParams, AttrDict, to_numpy

DEBUG = False

def repeat(input, repeat_time):
    """
    :param array/tensor: (A, B, C)
    :return: (A, A, A, B, B, B, C, C, C) if repeat_time == 3
    """
    if torch.is_tensor(input):
        input = to_numpy(input)
    array = np.stack([input] * repeat_time, axis=1)
    array = array.reshape(tuple((-1, *array.shape[2:])))
    return array


def repeat_tensor(input, repeat_time):
    """
    :param tensor: (A, B, C)
    :return: (A, A, A, B, B, B, C, C, C) if repeat_time == 3
    """
    input = torch.stack([input] * repeat_time, dim=1)
    output = input.view(tuple((-1, *input.size()[2:])))
    return output



def feature_selection(min_obs, max_obs, thresh, locs, size, time_idxs, reference_pts, model, input_traj, forward_feat=True):
    num_obs = locs.shape[0]
    
    obs_selected = []
    stats = {"loss":[], "pos_loss":[]}
    lookahead_counter = 0
    diff = 100
    for round in range(max_obs):
        loss_to_compare = 1e+9 
        associated_pos_loss = -1
        associated_obs = -1
        associated_recon_traj = None
        stats["round"+str(round)] = {}
        for obs_i in [i for i in range(num_obs) if i not in obs_selected]:
            # set all obs loc to very far except for obs_i
            locs_local = locs.clone()

            if forward_feat:
                masked_obs = [i for i in range(num_obs) if i not in obs_selected and i != obs_i]
            else:
                masked_obs = [i for i in obs_selected] + [obs_i]
            
            locs_local[masked_obs] = locs_local[masked_obs] + 1e+6

            recon_traj, _ = model.decode(reference_pts[None], locs_local[None], size[None], time_idxs[None])

            loss = ((input_traj - recon_traj[0]) ** 2).sum(dim=(0,1)).mean().item()
            stats["round"+str(round)]["obs_"+str(obs_i)] = loss

            if loss < loss_to_compare:
                loss_to_compare = loss
                associated_obs = obs_i
                associated_recon_traj = recon_traj
                associated_pos_loss = ((input_traj[0] - recon_traj[0,0]) ** 2).sum(dim=(0)).mean().item()

        if round > 0 and (round+1) >= min_obs :
            diff = (stats["loss"][-1] - loss_to_compare)/stats["loss"][-1] *100
            if diff < thresh:
                break
        

        stats["loss"].append(loss_to_compare)
        stats["pos_loss"].append(associated_pos_loss)
        obs_selected.append(associated_obs)
    stats["recon_traj"] = associated_recon_traj.detach().cpu().numpy()
    stats["loss"] = np.array(stats["loss"])
    stats["pos_loss"] = np.array(stats["pos_loss"])
    return obs_selected, stats


def reference_collision_checker(params,reference_pts_, loc_, time_idx_, size_=0.5): # no batch
        diff = reference_pts_[None, :] - loc_[:, None]
        diff_norm = np.linalg.norm(diff, axis=-1)
        diff_dir = diff / diff_norm[..., None]
        radius = 1 / np.sqrt((diff_dir ** 2 / size_ ** 2).sum(axis=-1))
        reference_collision = (diff_norm <= (radius + params.optimization_params.clearance * params.eval_params.clearance_scale)*time_idx_).any()
        return reference_collision

# input: unbatched trajectory, critical points, model.
# process:  1. select obstacles that are most important - 'feature_selection'.
#           2. check collision with selected obstacles' critical points - 'reference_collision_checker'.
# output: subset of obstacle CPs for a particular trajectory along with whether the CP is in collision or not.
def select_obstacles_and_check_collisions(locs, sizes, obs_time_idx, ref_pt_tensor, trajs_tensor, model, params):
    # obstacle/feature selection
    max_obs = params.eval_params.select_max_obs 
    min_obs = params.eval_params.select_min_obs 
    fs_thresh = params.eval_params.obs_selection_threshold
    obs_selected, fs_stats = feature_selection(min_obs, max_obs, fs_thresh, locs, sizes, obs_time_idx, ref_pt_tensor, model, trajs_tensor, forward_feat=True)

    collision = reference_collision_checker(params, to_numpy(ref_pt_tensor), to_numpy(locs[obs_selected]), to_numpy(obs_time_idx[obs_selected]))

    return obs_selected, fs_stats, collision

# input: batched trajectories, stats - {recon_loss, pos_recon_loss (pos only, no)}.
# process:  1. drop trajectories in which the ratio pos_recon_loss / straight_line_recon_loss is > ratio_threshold_loss
#           2. drop trajectories in which recon_loss > min_threshold_loss
# output: trajectory dropping mask that is True for trajectories to drop, False to keep.
def filter_trajectories(traj, feature_selection_stats, params):
    eval_params = params.eval_params
    final_pos_recon_losses = np.array([stat["pos_loss"][-1] for stat in feature_selection_stats])
    final_recon_losses = np.array([stat["loss"][-1] for stat in feature_selection_stats])
    traj_ = traj[:, 0]
    traj_len = traj_.shape[1]
    init_control_pts = traj_[:, None, 0] + \
                           np.linspace(0, 1, traj_len)[None, :, None] * \
                           (traj_[:, None, -1] - traj_[:, None, 0])
    straight_line_recon_loss = np.sum((init_control_pts - traj_)**2, axis=(1,2)) 

    # drop the trajectories in which the ratio: recon_loss/straight_line_loss > min_recon_perc (0.1)
    recon_loss_perc_drop = final_pos_recon_losses / straight_line_recon_loss > eval_params.min_recon_loss_perc_drop
   
    # drop the trajectories in which the final_recon_loss is above a certain threshold
    low_recon_loss = final_recon_losses > eval_params.min_recon_loss
    
    # # drop the trajectories in which the low recon loss is worse than the straight line one
    # mask = np.zeros_like(final_recon_losses, dtype=bool)  # Create a mask with the same shape
    # mask[low_recon_loss] = (final_recon_losses[low_recon_loss] / straight_line_recon_loss[low_recon_loss]) > 1
    # low_recon_loss[mask] = True


    return (recon_loss_perc_drop) & (low_recon_loss)


def eval(params):
    device = torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else "cpu")

    params.device = device
    eval_params = params.eval_params
    training_params = params.training_params
    sample_per_traj = eval_params.sample_per_traj
    downsample_traj = eval_params.downsample_traj
    n_traj_in_batch = 32 // sample_per_traj
    batch_size = n_traj_in_batch * downsample_traj

    dataset = HallucinationDataset(params, eval=True, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    params.model_params.gumbel_starting_tau = params.model_params.gumbel_final_tau

    model = Hallucination(params, None).to(device)
    assert os.path.exists(training_params.load_model)
    model.load_state_dict(torch.load(training_params.load_model, map_location=torch.device("cpu")))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.set_trainingstage(2) # so we get the non one-hot encoded outputs for the time distribution
    print(training_params.load_model, "loaded")

    rslts = {"obs_loc": [],
             "obs_size": [],
             "obs_time_idx":[],
             "traj": [],
             "goal": [],
             "raw_ori":[],
             "raw_traj":[],
             "recon_traj":[],
             "final_recon_losses":[]}
    
    if params.Dy == 2:
        rslts["cmd"] = []
    else:
        rslts["lin_vel"] = []
        rslts["ang_vel"] = []
        rslts["ori_base"] = []

    eval_dir = eval_params.eval_dir
    eval_plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_plots_dir, exist_ok=True)
    params.pop("device")
    json.dump(params, open(os.path.join(eval_dir, "params.json"), "w"), indent=4)
    shutil.copyfile(training_params.load_model, os.path.join(eval_dir, "model"))
    params.device = device
    for i_batch, sample_batched in enumerate(dataloader):
        for key, val in sample_batched.items():
            val = val[::downsample_traj]
            sample_batched[key] = val.to(device)

        full_traj_tensor = repeat_tensor(sample_batched["full_traj"], sample_per_traj)
        reference_pts_tensor = repeat_tensor(sample_batched["reference_pts"], sample_per_traj)

        reference_pts = to_numpy(reference_pts_tensor)
        trajs_tensor = repeat_tensor(sample_batched["traj"], sample_per_traj)

        if params.Dy == 2:
            cmds = repeat(sample_batched["cmd"], sample_per_traj)
        else:
            lin_vels = repeat(sample_batched["lin_vel"], sample_per_traj)
            ang_vels = repeat(sample_batched["ang_vel"], sample_per_traj)
            ori_bases = repeat(sample_batched["ori_base"], sample_per_traj)

        if (i_batch + 1) == len(dataloader):
            n_traj_in_batch = full_traj_tensor.shape[0] // sample_per_traj
        else:
            n_traj_in_batch = 32 // sample_per_traj

        n_samples = np.zeros(n_traj_in_batch).astype(int)
        print_n_samples = -1
        n_invalid_samples = np.zeros(n_traj_in_batch).astype(int)
        while (n_samples < sample_per_traj).any():
            if print_n_samples != n_samples.sum():
                print("{}/{} {}/{}".format(i_batch + 1, len(dataloader),
                                           n_samples.sum(), n_traj_in_batch * sample_per_traj))
                print_n_samples = n_samples.sum()

            # check if stuck for a long time
            need_break = True
            for n_valid, n_invalid in zip(n_samples, n_invalid_samples):
                if n_valid < sample_per_traj and n_invalid < sample_per_traj * eval_params.max_sample_trials:
                    need_break = False
                    break
            if need_break:
                print("give up after {} valid and {} invalid".format(n_samples, n_invalid_samples))
                break

            loc_tup, size_tup, _, obs_time_idx_tensor = model(full_traj_tensor, reference_pts_tensor, decode=False)
            
            locs_tensor, sizes_tensor = loc_tup[2], size_tup[2]

            # subset of obs CPs and collision checks.
            list_of_return = Parallel(n_jobs=os.cpu_count() if not DEBUG else 1)(
                delayed(select_obstacles_and_check_collisions)(locs_, sizes_, obs_time_idx_, ref_pt_tensor_, trajs_tensor_, model, params)
                for locs_, sizes_, obs_time_idx_, ref_pt_tensor_, trajs_tensor_ in zip(locs_tensor, sizes_tensor, obs_time_idx_tensor, reference_pts_tensor, trajs_tensor)
            )
            
            obs_selected_list, fs_stats, collision_list = zip(*list_of_return)

            trajs = to_numpy(trajs_tensor)

            # calculate straight line recon_loss and whether the trajectory with obstacles are accepted or not
            rejected_trajs = filter_trajectories(trajs, fs_stats)
            # combine accepted_trajs and collision_list
            collision_or_rejected = np.array(collision_list) | rejected_trajs
            recon_trajs = np.array([stat["recon_traj"] for stat in fs_stats])
            final_recon_losses = np.array([stat["loss"][-1] for stat in fs_stats])

            locs = to_numpy(locs_tensor)
            sizes = to_numpy(sizes_tensor)
            obs_time_idx = to_numpy(obs_time_idx_tensor)
            

            for j, (col, loc, size, traj, obs_selected, time_idx, recon_traj) in enumerate(zip(collision_or_rejected, locs, sizes, trajs, obs_selected_list, obs_time_idx, recon_trajs)):
                idx_in_batch = j // sample_per_traj
                if col:
                    n_invalid_samples[idx_in_batch] += 1
                    continue

                if n_samples[idx_in_batch] < sample_per_traj:
                    n_samples[idx_in_batch] += 1

                    loc = loc[obs_selected]
                    size = size[obs_selected]
                    time_idx = time_idx[obs_selected]
        
                    rslts["obs_loc"].append(loc)
                    rslts["obs_size"].append(size)
                    rslts["obs_time_idx"].append(time_idx)
                    rslts["traj"].append(traj)

                    # WARNING - goal set incorrectly!!!
                    goal = traj[0, -1] -(traj[0].copy()) 
                    # set the goal to be traj[0, -1] for every single point, minus the current point
                    goal /= (np.linalg.norm(goal, axis=-1, keepdims=True)+1e-9)             # adding 1e-9 as the last point will be 0
                    
                    # goal = to_numpy(sample_batched["goal"][idx_in_batch])

                    rslts["raw_ori"].append(to_numpy(sample_batched["odom_frame_ori_traj"][idx_in_batch]))
                    rslts["raw_traj"].append(to_numpy(sample_batched["odom_frame_traj"][idx_in_batch]))
                    rslts["recon_traj"].append(recon_traj)
                    rslts["final_recon_losses"].append(final_recon_losses[idx_in_batch])

                    cmd = lin_vel = None
                    if params.Dy == 2:
                        cmd = cmds[j]
                        rslts["goal"].append(goal)
                        rslts["cmd"].append(cmd)
                    else:
                        lin_vel = lin_vels[j]
                        rslts["goal"].append(goal)
                        rslts["lin_vel"].append(lin_vel)
                        rslts["ang_vel"].append(ang_vels[j])
                        rslts["ori_base"].append(ori_bases[j])


                    # sample_idx = i_batch * n_traj_in_batch + idx_in_batch
                    # if sample_idx % eval_params.plot_freq == 0 and n_samples[idx_in_batch] == 1:
                    #     fname = os.path.join(eval_plots_dir, str(sample_idx))
                    #     recon_traj, _ = model.decode(reference_pts_tensor[j:j + 1],
                    #                                  loc_tensors[j:j + 1], size_tensors[j:j + 1])
                    #     recon_traj = to_numpy(recon_traj)[0]
                    #     plot_eval_rslts(loc, size, traj, recon_traj, cmd, goal, lin_vel, fname=fname)

        if i_batch % 20 == 0:
            # rslts_ = {key: np.array(val) for key, val in rslts.items()}
            with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
                pickle.dump(rslts, f)
    
    print("---------------------------------------")
    print("Results")
    print(f"total # of trajs:\t{len(dataset)}")
    print(f"total # of trajs after downsampling:\t{len(dataset)/downsample_traj}")
    print(f"total # of uncollided samples:\t{len(rslts['traj'])}")
    print("---------------------------------------")


    # rslts = {key: np.array(val) for key, val in rslts.items()}
    with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
        pickle.dump(rslts, f)


if __name__ == "__main__":
    load_dir = "10obs_TS2+_lammr0"
    category_dir = "10obs"
    model_fname = "model_3090"
    sample_per_traj = 1
    downsample_traj = 4
    # plot_freq = 2000
    data_fnames = None
    clearance_scale = 0.5
    max_sample_trials = 10
    # new params:
    save_to_scratch=True
    select_max_obs = 7
    select_min_obs = 1
    obs_selection_threshold_perc = 1
    min_recon_loss = 5
    min_recon_loss_perc_drop = 0.1 # ratio of final / straight line < 0.05
    eval_dir_suffix = "_drop_FIXED"

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if save_to_scratch:
        eval_dir = os.path.join("/scratch/sghani2/hallucination_new", "LfH_eval", "{}_{}".format(load_dir, model_fname))
    else:
        eval_dir = os.path.join(repo_path, "LfH_eval", "{}_{}".format(load_dir, model_fname))
    eval_dir += eval_dir_suffix
    
    load_dir = os.path.join(repo_path, "rslts", "LfH_rslts", category_dir, load_dir)
    model_fname = os.path.join(load_dir, "trained_models", model_fname)
    params_fname = os.path.join(load_dir, "params.json")

    params = TrainingParams(params_fname, train=False)

    params.training_params.load_model = model_fname
    params.eval_params = eval_params = AttrDict()
    eval_params.eval_dir = eval_dir
    eval_params.params_fname = params_fname
    eval_params.sample_per_traj = sample_per_traj
    eval_params.downsample_traj = downsample_traj
    # eval_params.plot_freq = plot_freq
    eval_params.clearance_scale = clearance_scale
    eval_params.max_sample_trials = max_sample_trials
    eval_params.select_max_obs = select_max_obs
    eval_params.select_min_obs = select_min_obs
    eval_params.obs_selection_threshold = obs_selection_threshold_perc
    eval_params.min_recon_loss_perc_drop = min_recon_loss_perc_drop
    eval_params.min_recon_loss = min_recon_loss
    if data_fnames is not None:
        params.data_fnames = data_fnames

    eval(params)
