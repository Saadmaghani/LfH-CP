"""
Data loading module for trajectory hallucination model.

This module provides dataset classes for loading and preprocessing trajectory data,
including coordinate frame transformations and data augmentation.
"""

import os
import torch
import pickle
from typing import Dict, Any, Optional, Callable, Union

import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings("ignore")


class HallucinationDataset(Dataset):
    """
    PyTorch Dataset for trajectory data with B-spline control point sampling.
    
    Loads trajectory data (position, orientation, velocity) from pickle files,
    transforms to robot-centric frames, and extracts B-spline control points
    and trajectory segments for model training.
    """
    def __init__(self, params: Any, eval: bool = False, transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, random_seed: int = 3) -> None:
        """
        Initialize the HallucinationDataset.
        
        Expected directory structure:
            hallucination/
                data/
                    2D/
                        1m.p
                    3D/
                        1m.p
        
        Args:
            params: Configuration object containing:
                - Dy: State dimensionality (2 or 3)
                - odom_freq: Odometry frequency in Hz
                - data_fnames: List of pickle file names to load
                - model_params: Model parameters including B-spline configuration
            eval: If True, enables evaluation mode and loads additional data (e.g., commands).
                Defaults to False.
            transform: Optional transform to apply to each sample. Defaults to None.
        """
        """
        Assumed data orginazation
        hallucination/
            data/
                # different dimension
                2D/
                    # different data files
                    1m.p
        """
        super(HallucinationDataset, self).__init__()

        self.rng = np.random.default_rng(random_seed)
        
        self.params = params
        self.eval = eval
        self.transform = transform

        self.Dy = params.Dy
        self.odom_freq = params.odom_freq

        repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_dir = os.path.join(repo_path, "data", "2D" if self.Dy == 2 else "3D")

        model_params = params.model_params
        reference_pt_timestamp = np.arange(model_params.knot_start + 2, model_params.knot_end - 2)
        reference_pt_timestamp = reference_pt_timestamp * model_params.knot_dt
        # for reference pts
        self.reference_pt_idx = np.round(reference_pt_timestamp * self.odom_freq).astype(np.int32)
        # for full_traj
        full_traj_len = self.reference_pt_idx[-1] - self.reference_pt_idx[0] + 1  # from 0 to reference_pt_idx[-1]
        full_traj_len = full_traj_len
        # for traj (bspline order is 3)
        self.traj_start = self.reference_pt_idx[1] - self.reference_pt_idx[0]
        self.traj_end = self.reference_pt_idx[-2] - self.reference_pt_idx[0] + 1   # +1: for exclusion

        # needed in initializing hallucination model
        model_params.full_traj_len = int(full_traj_len)
        model_params.traj_len = int(self.traj_end - self.traj_start)

        self.trajs = []
        self.traj_mapper = []
        self.odom_mapper = []
        traj_idx = 0
        for fname in params.data_fnames:
            data = None
            with open(os.path.join(data_dir, fname), "rb") as f:
                data = pickle.load(f)

            data["pos"], data["ori"], data["vel"], data["ang_vel"], data["cmd_vel"], data["cmd_ang_vel"] = np.array(data["pos"]), np.array(data["ori"]), np.array(data["vel"]), np.array(data["ang_vel"]),  np.array(data["cmd_vel"]), np.array(data["cmd_ang_vel"]) 
            
            pos, ori, vel, ang_vel = data["pos"], data["ori"], data["vel"], data["ang_vel"]

            self.trajs.append(data)

            assert len(pos) == len(vel) == len(ori)
            self.traj_mapper.extend([traj_idx] * (len(pos) - full_traj_len + 1))
            self.odom_mapper.extend(np.arange(len(pos) - full_traj_len + 1) - self.reference_pt_idx[0])
            traj_idx += 1

        self.traj_mapper = np.array(self.traj_mapper)
        self.odom_mapper = np.array(self.odom_mapper)

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of trajectory samples available.
        """
        return len(self.traj_mapper)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - reference_pts: Control points for B-spline basis
                - full_traj: Concatenated position and velocity trajectory
                - traj: Stacked position and velocity labels
                - odom_frame_traj: Raw trajectory in odometry frame
                - odom_frame_ori_traj: Raw orientations in odometry frame
                - cmd: (eval mode only) Command velocities
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        traj_idx = self.traj_mapper[idx]
        odom_idx = self.odom_mapper[idx]

        traj = self.trajs[traj_idx]
        pos, ori, vel, ang_vel = traj["pos"], traj["ori"], traj["vel"], traj["ang_vel"]

        pos_base, ori_base = pos[odom_idx], ori[odom_idx]
        pos_full_traj = pos[odom_idx + self.reference_pt_idx[0]:odom_idx + self.reference_pt_idx[-1] + 1]
        ori_full_traj = ori[odom_idx + self.reference_pt_idx[0]:odom_idx + self.reference_pt_idx[-1] + 1]
        vel_full_traj = vel[odom_idx + self.reference_pt_idx[0]:odom_idx + self.reference_pt_idx[-1] + 1]
        # ang_vel_full_traj = ang_vel[odom_idx + self.reference_pt_idx[0]:odom_idx + self.reference_pt_idx[-1] + 1]

        r = R.from_quat(ori_base).inv()
        pos_transformed = r.apply(pos_full_traj - pos_base)
        vel_transformed = r.apply(vel_full_traj)

        if self.Dy == 2:
            pos_transformed = pos_transformed[:, :2]
            vel_transformed = vel_transformed[:, :2]

            if not self.eval and self.rng.random() > 0.5:
                # flipping augmentation
                pos_transformed[:, 1] = -pos_transformed[:, 1]
                vel_transformed[:, 1] = -vel_transformed[:, 1]

        pos_label = pos_transformed[self.traj_start:self.traj_end]
        vel_label = vel_transformed[self.traj_start:self.traj_end]

        raw_ori = ori_full_traj[self.traj_start: self.traj_end]
        raw_traj_pos = pos_full_traj[self.traj_start: self.traj_end]
        raw_traj_vel = vel_full_traj[self.traj_start: self.traj_end]

        data = {"reference_pts": pos_transformed[self.reference_pt_idx - self.reference_pt_idx[0]],
                "full_traj": np.concatenate([pos_transformed, vel_transformed], axis=-1),
                "traj": np.stack([pos_label, vel_label], axis=0),
                "odom_frame_traj": np.stack([raw_traj_pos, raw_traj_vel], axis=0),
                "odom_frame_ori_traj": raw_ori
                }
        if self.eval:
            if self.Dy == 2:
                cmd_vel, cmd_ang_vel = traj.get("cmd_vel"), traj.get("cmd_ang_vel")
                cmd_vel = cmd_vel[odom_idx + self.reference_pt_idx[0] + self.traj_start: self.traj_end + odom_idx + self.reference_pt_idx[0]]
                cmd_ang_vel = cmd_ang_vel[odom_idx + self.reference_pt_idx[0] + self.traj_start: self.traj_end + odom_idx + self.reference_pt_idx[0]]
                cmd = np.array([cmd_vel, cmd_ang_vel])
                data.update({"cmd": cmd})
                # data.update({"goal": })?

                # raise NotImplementedError("Goal needs to be tested!")
                # WARNING - DONT TAKE THIS AS CORRECT GOAL - IT NEEDS TO BE TESTED. 
                # traj_end = pos_label[0, [-1]]           # in traj0 frame
                # goal = traj_end - pos_label[0]          # unrotated, in robot centric frame at every step in traj
                # ori_ = raw_ori - raw_ori[[0]]           # rotation in traj0 frame
                # goal_rotation = R.from_quat(ori_).inv()
                # goal_rotated = goal_rotation.apply(goal) # goal now in correct frame at every step in traj
                # data.update({"goal": goal_rotated})


            # else:
            #     goals = traj.get("goal")
            #     goal = goals[odom_idx + self.reference_pt_idx[0] + self.traj_start]
            #     data.update({"goal": goal})
            #     lin_vel = vel_label[0]
            #     ang_vel = ang_vel[odom_idx + self.reference_pt_idx[0] + self.traj_start]
            #     ang_vel = r.apply(ang_vel)
            #     data.update({"lin_vel": lin_vel, "ang_vel": ang_vel, "ori_base": ori_base})

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    """
    Convert numpy arrays in sample to PyTorch tensors.
    
    Handles special cases like transposing trajectory data to match
    PyTorch's channel-first convention for convolutional networks.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert all numpy arrays in data to PyTorch tensors.
        
        Special handling for 'full_traj' which is transposed to channel-first
        format for convolutional layers.
        
        Args:
            data: Dictionary containing numpy array data.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with all arrays converted to tensors.
        """
        new_data = {}
        for key, val in data.items():
            if key == "full_traj":
                val = val.transpose((1, 0))  # as torch conv is channel first
            val = val.astype(np.float32)
            new_data[key] = torch.from_numpy(val)
        return new_data
