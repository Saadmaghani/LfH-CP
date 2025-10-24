import torch
import os
import numpy as np
import pickle

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from LfH_main import sample, TrainingParams
from eval import  select_obstacles_and_check_collisions, filter_trajectories

from saad_utils import setup_timesteps_traj, quat_to_psi, rotate_origin_only

from render_demo_2D_mpi import generate_obstacle_trajectories, Params as RenderParams


# GETTING DATA DIRECTLY FROM MODEL

#params:
model_dir = "10obs_TS2+_lammr0_model_3090_drop_FIXED"
model_cps = ["model", "model"]
training_stages = [1, 2]
num_samples = 12500
# end of params
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# code:

def get_straight_line(_trajs_pos):
    _traj_len = _trajs_pos.shape[-2]
    _init_control_pts = _trajs_pos[:, None, 0] + \
                           np.linspace(0, 1, _traj_len)[None, :, None] * \
                           (_trajs_pos[:, None, -1] - _trajs_pos[:, None, 0])
    
    _straight_line_recon_loss = np.sum((_init_control_pts - _trajs_pos)**2, axis=(1,2)) 

    return _init_control_pts, _straight_line_recon_loss

all_results = {1:{}, 2:{}}
base_path = os.path.join("/projects/RobotiXX/saad/hallucination/rslts/LfH_rslts", model_dir)
for i, cp in enumerate(model_cps):
    model_path = os.path.join(base_path, cp)
    params_path = os.path.join(base_path, "params.json") 

    params=TrainingParams(params_path, train=False)
    params.model_params.gumbel_starting_tau = 0.1
    params.model_params.lambda_time_idx_entropy = 0.5

    dict_out = sample(params, model_path, n_times=num_samples, training_stage=training_stages[i], seed=100)
    dict_out["training_stage"] = training_stages[i]


    min_obs = 1
    max_obs = 7
    thresh = 1
    locs = dict_out["loc"]
    sizes = dict_out["size"]
    time_idxs = dict_out["time_idx"]
    model = dict_out["model"]
    reference_pts = dict_out["reference_pts"]
    traj_tensor = dict_out["traj"]


    selected_obs, stats, in_collision = [], [], []
    for sample_i in range(num_samples):
        sel_obs, stats_, collision = select_obstacles_and_check_collisions(locs[sample_i], sizes[sample_i],  time_idxs[sample_i], reference_pts[sample_i], traj_tensor[sample_i], model, params)
        # sel_obs, stats_ = feature_selection(min_obs, max_obs, thresh, locs[sample_i], sizes[sample_i], time_idxs[sample_i], reference_pts[sample_i], model, traj_tensor[sample_i], forward_feat=True)
        selected_obs.append(sel_obs)
        stats.append(stats_)
        in_collision.append(collision)

    trajs = traj_tensor.detach().cpu().numpy()

    rejected_trajs = filter_trajectories(trajs, stats, params)
    # combine accepted_trajs and collision_list
    collision_or_rejected = np.array(in_collision) | rejected_trajs
    recon_trajs = np.array([stat["recon_traj"] for stat in stats])
    final_recon_losses = np.array([stat["loss"][-1] for stat in stats])

    not_rejected = np.where(~collision_or_rejected)[0]

    good_recon_trajs, good_recon_losses, good_trajs = recon_trajs[not_rejected], final_recon_losses[not_rejected], trajs[not_rejected]
    # print(f"training stage {k}")
    # # print(not_rejected, good_recon_losses)
    # print("-"*100)

    # dict_out["not_rejected_idx"] = not_rejected
    # dict_out["good_recon_losses"] = good_recon_losses
    # dict_out["good_recon_trajs"] = good_recon_trajs
    # dict_out["good_trajs"] = good_trajs
    # straight_line_traj, straight_line_loss = get_straight_line(good_trajs[:, 0])
    # dict_out["straight_line_traj"] = straight_line_traj
    # dict_out["straight_line_loss"] = straight_line_loss

    all_results[training_stages[i]]["not_rejected_idx"] = not_rejected
    all_results[training_stages[i]]["good_recon_losses"] = good_recon_losses
   

# calculate the difference between the static recon loss and cp recon loss
_, idx_common1, idx_common2 = np.intersect1d(all_results[1]["not_rejected_idx"] , all_results[2]["not_rejected_idx"], assume_unique=True, return_indices=True)

recon_loss1 = all_results[1]["good_recon_losses"][idx_common1]
recon_loss2 = all_results[2]["good_recon_losses"][idx_common2]


from scipy import stats
import numpy as np


t_stat, p_value = stats.ttest_rel(recon_loss1, recon_loss2)

diff = recon_loss1 - recon_loss2
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)  # sample std

print("mean:",mean_diff) 
print("std_dev:",std_diff)

alpha = 0.05
if p_value < alpha:
    print(f"Reject null hypothesis (p={p_value:.5f})")
else:
    print(f"Fail to reject null hypothesis (p={p_value:.5f})")
