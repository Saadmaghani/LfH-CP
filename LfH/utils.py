import os
import json
import shutil
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class AttrDict(dict):
    """A dictionary that allows attribute-style access to its items."""
    def __init__(self, *args, **kwargs):
        """
        Initialize an AttrDict.
        
        Args:
            *args: Variable length argument list passed to dict.
            **kwargs: Arbitrary keyword arguments passed to dict.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TrainingParams(AttrDict):
    """
    Configuration parameters for training loaded from a JSON file.
    
    This class extends AttrDict to provide convenient attribute access to training
    parameters and automatically handles directory setup for results storage.
    """
    def __init__(self, training_params_fname=None, train=True):
        """
        Initialize training parameters from a JSON configuration file.
        
        Args:
            training_params_fname: Path to the JSON configuration file. If None, defaults to
                'params.json' in the script's directory. Defaults to None.
            train: If True, creates a results directory with timestamped subdirectories
                and copies the params file. Defaults to True.
        """
        if training_params_fname is None:
            training_params_fname = os.path.join(
                os.path.dirname(__file__), "params.json"
            )
        
        config = None
        with open(training_params_fname) as f:
            config = json.load(f)
        
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

        if self.training_params.load_model is not None:
            self.training_params.load_model = os.path.join("..", "interesting_models", self.training_params.load_model)

        if train:
            repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            self.rslts_dir = os.path.join(repo_path, "rslts", "LfH_rslts", time.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, os.path.join(self.rslts_dir, "params.json"))

        super(TrainingParams, self).__init__(self.__dict__)

    def _clean_dict(self, dict_to_clean):
        """
        Recursively clean dictionary values by converting empty strings to None.
        
        Args:
            dict_to_clean: Dictionary to clean.
            
        Returns:
            AttrDict: A cleaned dictionary with empty strings converted to None
                and nested dictionaries recursively cleaned.
        """
        for k, v in dict_to_clean.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = self._clean_dict(v)
            dict_to_clean[k] = v
        return AttrDict(dict_to_clean)


def quat_to_psi(ori):
    q1, q2, q3, q0 = ori[...,0], ori[...,1], ori[...,2], ori[...,3]
    PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
    return PSI

def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = np.take(xy, 0, -1), np.take(xy, 1, -1)
    xx = x * np.cos(radians) - y * np.sin(radians)
    yy = x * np.sin(radians) + y * np.cos(radians)
    return np.stack([xx, yy], axis=-1)

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def setup_timesteps_traj(knot_start=-3, knot_end=18):
    ref_pt_ts = np.arange(knot_start+2,  knot_end -2)* 0.15
    reference_pt_idx = np.round(ref_pt_ts * 50).astype(np.int32)

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


### Plotting functions

# Main plotting functions
def plot_opt(writer, reference_pts, recon_control_points, loc, size, epoch, is_bug=False, num_sample=3):
    batch_size, num_obstacle, Dy = loc.size()
    if Dy not in [2, 3]:
        pass

    if batch_size <= num_sample:
        idx = 0
        num_sample = min(batch_size, num_sample)
    else:
        idx = np.random.randint(batch_size - num_sample)

    reference_pts = reference_pts[idx:idx + num_sample]
    recon_control_points = recon_control_points[:, idx:idx + num_sample]
    loc = loc[idx:idx + num_sample]
    size = size[idx:idx + num_sample]

    loc = to_numpy(loc)
    size = to_numpy(size)
    recon_control_points = to_numpy(recon_control_points)
    reference_pts = to_numpy(reference_pts)

    fig = plt.figure(figsize=(5 * num_sample, 5))
    colors = sns.color_palette('husl', n_colors=num_obstacle + 1)
    for i in range(num_sample):
        if Dy == 2:
            ax = fig.add_subplot(1, num_sample, i + 1)
            ax.plot(reference_pts[i, :, 0], reference_pts[i, :, 1], label="reference")
            ax.scatter(reference_pts[i, :, 0], reference_pts[i, :, 1])

            obses = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1]) for loc_, size_ in zip(loc[i], size[i])]
            for j, obs in enumerate(obses):
                ax.add_artist(obs)
                obs.set_alpha(0.5)
                obs.set_facecolor(colors[j])

            ode_num_timestamps = recon_control_points.shape[0]
            for j in range(ode_num_timestamps):
                ax.plot(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1], label="opt_{}".format(j))
                ax.scatter(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1])

            x_min = np.minimum(np.min(reference_pts[i, :, 0]), np.min(recon_control_points[:, i, :, 0]))
            x_max = np.maximum(np.max(reference_pts[i, :, 0]), np.max(recon_control_points[:, i, :, 0]))
            y_min = np.minimum(np.min(reference_pts[i, :, 1]), np.min(recon_control_points[:, i, :, 1]))
            y_max = np.maximum(np.max(reference_pts[i, :, 1]), np.max(recon_control_points[:, i, :, 1]))
            x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
            x_range, y_range = x_max - x_min, y_max - y_min
            x_min, x_max = x_mid - 1.5 * x_range, x_mid + 1.5 * x_range
            y_min, y_max = y_mid - 1.5 * y_range, y_mid + 1.5 * y_range

            ax.axis('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max])
            ax.legend()
        else:
            ax = fig.add_subplot(1, num_sample, i + 1, projection="3d")
            ax.plot(reference_pts[i, :, 0], reference_pts[i, :, 1], reference_pts[i, :, 2], label="reference")
            ax.scatter(reference_pts[i, :, 0], reference_pts[i, :, 1], reference_pts[i, :, 2])

            for j, (loc_, size_) in enumerate(zip(loc[i], size[i])):
                _draw_ellipsoid(loc_, size_, ax, colors[j])

            ode_num_timestamps = recon_control_points.shape[0]
            for j in range(ode_num_timestamps):
                ax.plot(recon_control_points[j, i, :, 0],
                        recon_control_points[j, i, :, 1],
                        recon_control_points[j, i, :, 2],
                        label="opt_{}".format(j))
                ax.scatter(recon_control_points[j, i, :, 0],
                           recon_control_points[j, i, :, 1],
                           recon_control_points[j, i, :, 2])

            x_min = np.minimum(np.min(reference_pts[i, :, 0]), np.min(recon_control_points[:, i, :, 0]))
            x_max = np.maximum(np.max(reference_pts[i, :, 0]), np.max(recon_control_points[:, i, :, 0]))
            y_min = np.minimum(np.min(reference_pts[i, :, 1]), np.min(recon_control_points[:, i, :, 1]))
            y_max = np.maximum(np.max(reference_pts[i, :, 1]), np.max(recon_control_points[:, i, :, 1]))
            z_min = np.minimum(np.min(reference_pts[i, :, 2]), np.min(recon_control_points[:, i, :, 2]))
            z_max = np.maximum(np.max(reference_pts[i, :, 2]), np.max(recon_control_points[:, i, :, 2]))
            x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
            x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
            x_min, x_max = x_mid - 1.0 * x_range, x_mid + 1.0 * x_range
            y_min, y_max = y_mid - 1.0 * y_range, y_mid + 1.0 * y_range
            z_min, z_max = z_mid - 1.0 * z_range, z_mid + 1.0 * z_range

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max], zlim=[z_min, z_max])
            ax.legend()
            _set_axes_equal(ax)

    fig.tight_layout()
    writer.add_figure("sample" if not is_bug else "opt_error_logging", fig, epoch)
    plt.close("all")

def plot_time_dist(writer, model, time_index, epoch, num_sample=3, seed=3):
    # not working as expected
    pass
    # for i in range(len(model.encoder.time_heads)):
    #     layer = model.encoder.time_heads[i]
    #     writer.add_histogram(f'time_heads/[{i}]/weight', layer.weight, epoch)
    #     writer.add_histogram(f'time_heads/[{i}]/bias', layer.bias, epoch)
        
    # batch_size, num_obstacles, num_ctrl_pts = time_index.size()
    # if batch_size <= num_sample:
    #     idx = 0
    #     num_sample = min(batch_size, num_sample)
    # else:
    #     np.random.seed(seed)
    #     idx = np.random.randint(batch_size - num_sample)
    
    # time_idx = to_numpy(time_index[idx:idx+num_sample])
    # for i in range(num_sample):
    #     for j in range(num_obstacles):
    #         writer.add_histogram(f'obs_time_index/sample_{i}/obs_[{j}]', time_idx[i, j], epoch)

def plot_obs_dist(writer, params, full_traj, loc_mu, loc_log_var, size_mu, size_log_var, epoch, num_sample=3, seed=3):
    batch_size, num_obstacle, Dy = loc_mu.size()
    if Dy not in [2, 3]:
        pass
    if batch_size <= num_sample:
        idx = 0
        num_sample = min(batch_size, num_sample)
    else:
        np.random.seed(seed)
        idx = np.random.randint(batch_size - num_sample)

    full_traj = to_numpy(full_traj[idx:idx + num_sample, :Dy])
    loc_prior_mu = np.mean(full_traj, axis=-1)
    loc_prior_var = np.maximum(np.var(full_traj, axis=-1), params.model_params.min_obs_loc_prior_var)
    loc_prior_var *= params.model_params.obs_loc_prior_var_coef
    loc_prior_std = np.sqrt(loc_prior_var)
    loc_mu = to_numpy(loc_mu[idx:idx + num_sample])
    loc_log_var = to_numpy(loc_log_var[idx:idx + num_sample])
    loc_std = np.exp(0.5 * loc_log_var)
    size_mu = to_numpy(size_mu[idx:idx + num_sample])
    size_log_var = to_numpy(size_log_var[idx:idx + num_sample])
    size_std = np.exp(0.5 * size_log_var)

    fig = plt.figure(figsize=(5 * num_sample, 5))

    def softplus(a):
        return np.log(1 + np.exp(a))

    colors = sns.color_palette('husl', n_colors=num_obstacle + 1)
    for i in range(num_sample):
        if Dy == 2:
            ax = fig.add_subplot(1, num_sample, i + 1)

            obs_loc_prior = Ellipse(xy=loc_prior_mu[i], width=2 * loc_prior_std[i, 0], height=2 * loc_prior_std[i, 1],
                                    facecolor='none')
            edge_c = colors[-1]
            ax.add_artist(obs_loc_prior)
            obs_loc_prior.set_edgecolor(edge_c)

            obs_loc = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1], facecolor='none')
                       for loc_, size_ in zip(loc_mu[i], loc_std[i])]
            obs_size_mu = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1], facecolor='none')
                           for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i]))]
            obs_size_s = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1], facecolor='none')
                          for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] - size_std[i]))]
            obs_size_l = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1], facecolor='none')
                          for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] + size_std[i]))]
            for j, (loc_, size_mu_, size_s, size_l) in enumerate(zip(obs_loc, obs_size_mu, obs_size_s, obs_size_l)):
                edge_c = colors[j]
                ax.add_artist(loc_)
                ax.add_artist(size_mu_)
                ax.add_artist(size_s)
                ax.add_artist(size_l)
                loc_.set_edgecolor(edge_c)
                size_mu_.set_edgecolor(edge_c)
                size_s.set_edgecolor(edge_c)
                size_l.set_edgecolor(edge_c)
                loc_.set_linestyle('--')

            x_min = np.min([loc_prior_mu[i, 0] - loc_prior_std[i, 0],
                            np.min(loc_mu[i, :, 0] - loc_std[i, :, 0]),
                            np.min(loc_mu[i, :, 0] - softplus(size_mu[i, :, 0] + size_std[i, :, 0]))])
            x_max = np.max([loc_prior_mu[i, 0] + loc_prior_std[i, 0],
                            np.max(loc_mu[i, :, 0] + loc_std[i, :, 0]),
                            np.max(loc_mu[i, :, 0] + softplus(size_mu[i, :, 0] + size_std[i, :, 0]))])
            y_min = np.min([loc_prior_mu[i, 1] - loc_prior_std[i, 1],
                            np.min(loc_mu[i, :, 1] - loc_std[i, :, 1]),
                            np.min(loc_mu[i, :, 1] - softplus(size_mu[i, :, 1] + size_std[i, :, 1]))])
            y_max = np.max([loc_prior_mu[i, 1] + loc_prior_std[i, 1],
                            np.max(loc_mu[i, :, 1] + loc_std[i, :, 1]),
                            np.max(loc_mu[i, :, 1] + softplus(size_mu[i, :, 1] + size_std[i, :, 1]))])
            x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
            x_range, y_range = x_max - x_min, y_max - y_min
            x_min, x_max = x_mid - x_range * 0.75, x_mid + x_range * 0.75
            y_min, y_max = y_mid - y_range * 0.75, y_mid + y_range * 0.75

            ax.axis('equal')
            ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max])
        else:
            ax = fig.add_subplot(1, num_sample, i + 1, projection="3d")
            _draw_ellipsoid(loc_prior_mu[i], loc_prior_std[i], ax, colors[-1], alpha=0.6)

            for j, (loc_, size_) in enumerate(zip(loc_mu[i], loc_std[i])):
                _draw_ellipsoid(loc_, size_, ax, colors[j])
            for j, (loc_, size_) in enumerate(zip(loc_mu[i], softplus(size_mu[i]))):
                _draw_ellipsoid(loc_, size_, ax, colors[j])

            x_min = np.min([loc_prior_mu[i, 0] - loc_prior_std[i, 0],
                            np.min(loc_mu[i, :, 0] - loc_std[i, :, 0]),
                            np.min(loc_mu[i, :, 0] - softplus(size_mu[i, :, 0]))])
            x_max = np.max([loc_prior_mu[i, 0] + loc_prior_std[i, 0],
                            np.max(loc_mu[i, :, 0] + loc_std[i, :, 0]),
                            np.max(loc_mu[i, :, 0] + softplus(size_mu[i, :, 0]))])
            y_min = np.min([loc_prior_mu[i, 1] - loc_prior_std[i, 1],
                            np.min(loc_mu[i, :, 1] - loc_std[i, :, 1]),
                            np.min(loc_mu[i, :, 1] - softplus(size_mu[i, :, 1]))])
            y_max = np.max([loc_prior_mu[i, 1] + loc_prior_std[i, 1],
                            np.max(loc_mu[i, :, 1] + loc_std[i, :, 1]),
                            np.max(loc_mu[i, :, 1] + softplus(size_mu[i, :, 1]))])
            z_min = np.min([loc_prior_mu[i, 2] - loc_prior_std[i, 2],
                            np.min(loc_mu[i, :, 2] - loc_std[i, :, 2]),
                            np.min(loc_mu[i, :, 2] - softplus(size_mu[i, :, 2]))])
            z_max = np.max([loc_prior_mu[i, 2] + loc_prior_std[i, 2],
                            np.max(loc_mu[i, :, 2] + loc_std[i, :, 2]),
                            np.max(loc_mu[i, :, 2] + softplus(size_mu[i, :, 2]))])
            x_mid, y_mid, z_mid = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
            x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
            x_min, x_max = x_mid - 0.75 * x_range, x_mid + 0.75 * x_range
            y_min, y_max = y_mid - 0.75 * y_range, y_mid + 0.75 * y_range
            z_min, z_max = z_mid - 0.75 * z_range, z_mid + 0.75 * z_range

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.set(xlim=[x_min, x_max], ylim=[y_min, y_max], zlim=[z_min, z_max])
            _set_axes_equal(ax)

    fig.tight_layout()
    writer.add_figure("distribution", fig, epoch)
    plt.close("all")

# Helper functions for plotting
def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def _set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _draw_ellipsoid(loc, size, ax, color, alpha=0.2):
    x, y, z = loc
    rx, ry, rz = size

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    sx = x + rx * np.outer(np.cos(u), np.sin(v))
    sy = y + ry * np.outer(np.sin(u), np.sin(v))
    sz = z + rz * np.outer(np.ones_like(u), np.cos(v))

    # Plot:
    ax.plot_surface(sx, sy, sz, rstride=4, cstride=4, color=color, alpha=alpha)

   