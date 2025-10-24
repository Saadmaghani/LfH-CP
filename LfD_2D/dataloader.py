import os
import torch
# import pickle
import numpy as np
import json
import h5py

from torch.utils.data import Dataset

from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore")


class Demo_2D_Dataset(Dataset, ABC):
    def __init__(self, params, split="train", transform=None, max_cache_size=1, shuffle_seed=29, test_over_dataset="osa"):
        """
        Assumed data orginazation - only works with h5py files
        scratch/<user>/hallucination_new/
            LfH_demo/
                # different sequence length
                5_seq/
                    # different demos
                    1+obs_lflh/
                        LfH_demo.h5
                    ...
                ...
        """
        super(Demo_2D_Dataset, self).__init__()

        assert split in ["train", "val", "test"], "split must be one of ['train', 'val', 'test']"
        if split != "train":
            assert test_over_dataset in ["osa", "static", "dynamic", "only_close", "dynamic_jerk", "dynamic_accel"], "Invalid test_over_dataset option"
        assert params.load_from_scratch, "now we're only loading from scratch directory"

        self.params = params
        self.split = split
        self.transform = transform
        self.train_prop = params.model_params.train_prop
        self.val_prop = params.model_params.val_prop  # New validation proportion
        self.autoregressive = params.model_params.autoregressive
        # repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
       
        self.metadatas = [{"cum_size": 0}]
        self.headers = ["laser", "cmd", "goal"]  # Laser = [L, 720], cmd = [N, 2], goal = [2]- (x, y) 
                                                # LfD model should take in as input: laser, goal and output: cmd

        for demo_dir in params.demo_dirs:
            demo_dir_path = self._get_demo_path(demo_dir) 
            render_params = os.path.join(demo_dir_path, "render_params.json")
            if os.path.exists(render_params):
                with open(render_params) as f:
                    render_params = json.load(f)

                if split != "train":
                    if test_over_dataset != render_params["dataset_name"]:
                        continue
            else:
                print("render_params.json file does not exist. Assuming this is a valid dataset...")
                
            if not os.path.exists(os.path.join(demo_dir_path, "LfH_demo.h5")):
                raise FileNotFoundError("h5 file not found - "+str(os.path.join(demo_dir_path, "LfH_demo.h5")))
        
            with h5py.File(os.path.join(demo_dir_path, "LfH_demo.h5"), "r") as file:
                dataset_metadata = {
                    "path_to_h5_file": os.path.join(demo_dir_path, "LfH_demo.h5"),
                    "h5_file_pointer": None,
                    "index_offset": 0,
                    "dataset_type": "h5_file",
                    "cum_size": self.metadatas[-1]["cum_size"]
                }

                dataset_length = self._get_dataset_length(file["laser"], dataset_metadata) # len(file["laser"])
                train_length = int(np.round(self.train_prop * dataset_length))
                val_length = int(np.round(self.val_prop * dataset_length))
                test_length = dataset_length - train_length - val_length
                
                dataset_metadata["shuffled_idxs"] = np.arange(dataset_length)
                if shuffle_seed > 0: # shuffling if shuffle_seed > 0, else no shuffling
                    np.random.seed(shuffle_seed)
                    np.random.shuffle(dataset_metadata["shuffled_idxs"])

                if self.split == "train":
                    dataset_metadata["cum_size"] += train_length
                    dataset_metadata["index_offset"] = 0
                elif self.split == "val":
                    dataset_metadata["cum_size"] += val_length
                    dataset_metadata["index_offset"] = train_length
                else:  # test
                    dataset_metadata["cum_size"] += test_length
                    dataset_metadata["index_offset"] = train_length + val_length
                
            self.metadatas.append(dataset_metadata)

    @abstractmethod
    def _get_demo_path(self, demo_dir):
        pass

    @abstractmethod
    def _get_dataset_length(self, h5_laser_dataset, dataset_metadata):
        pass
    
    @abstractmethod
    def _get_item_from_h5_file(self, metadata, dataset_idx):
        pass

    def __len__(self):
        return self.metadatas[-1]["cum_size"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        for metadata_i_minus_1, metadata in enumerate(self.metadatas[1:]):
            if idx < metadata["cum_size"]:
                # idx -> shuffled_indices idx -> dataset idx
                shuffled_idx = idx - self.metadatas[metadata_i_minus_1]["cum_size"] 
                shuffled_idx += metadata["index_offset"] 
                dataset_idx = metadata["shuffled_idxs"][shuffled_idx]           

                if metadata["dataset_type"] != "h5_file":
                    raise Exception("Invalid dataset_type: " + metadata["dataset_type"])

                if metadata["h5_file_pointer"] is None:
                    metadata["h5_file_pointer"] = h5py.File(metadata["path_to_h5_file"], "r", swmr=True)

                data = self._get_item_from_h5_file(metadata, dataset_idx) #{k: metadata["h5_file_pointer"][k][dataset_idx] for k in metadata["h5_file_pointer"] if k in self.headers}

                if self.transform:
                    data = self.transform(data)

                return data
        
        raise IndexError("No data for requested index.")
    
    def __del__(self):
        self.close_h5_file_pointers()
        
    def close_h5_file_pointers(self):
        for metadata in self.metadatas[1:]:
            if metadata["h5_file_pointer"] is not None:
                metadata["h5_file_pointer"].close()
                metadata["h5_file_pointer"] = None

class Single_Render_Dataset(Demo_2D_Dataset):
    def __init__(self, params, split="train", transform=None, max_cache_size=1, shuffle_seed=29, test_over_dataset="osa"):
        # trajectory length in time (s) = sequence_length * 1/laser_scan_freq  
        laser_scan_freq = params.dataset_params.laser_scan_freq
        cmd_vel_freq = params.dataset_params.cmd_vel_freq
        odom_freq = params.dataset_params.odom_freq
        self.len_of_one_seq = len_of_one_seq = params.dataset_params.sequence_length
        
        # load h5py file one by one, assert that the cmd.shape[1] == laser.shape[1]
        # even if autoregressive is not on. since the data will be filtered by the dataloader now

        if cmd_vel_freq <= odom_freq:
            di = laser_scan_freq/cmd_vel_freq
        else:
            print(f"desired cmd_vel freq too high. max. freq is odom_freq of {odom_freq}")
            di = laser_scan_freq/odom_freq

        self.total_required_len_in_timesteps = len_of_one_seq
        self.future_cmd_pred = future_cmd_pred = 0
        if params.dataset_params.num_pred_cmd > 1:
            # The horizon in timesteps is the number of predictions times the step interval
            self.future_cmd_pred = future_cmd_pred = int((params.dataset_params.num_pred_cmd -1) * di)
            # Calculate the total length needed for one full sample
            self.total_required_len_in_timesteps += future_cmd_pred

        if params.model_params.autoregressive:
            self.total_required_len_in_timesteps += 1

        # okay so now that we have L, we simply find the dataset size by N*L, where N=number of rows.
        super().__init__(params, split, transform, max_cache_size, shuffle_seed, test_over_dataset)
        
    def _get_demo_path(self, demo_dir):
        return os.path.join("/scratch", os.getenv("USER"), "hallucination_new", "LfH_demo", "single_render", demo_dir)
    
    def _get_dataset_length(self, h5_laser_dataset, dataset_metadata):
        number_of_rows, length_of_one_traj = h5_laser_dataset.shape[:2]
        # Calculate the number of valid sequences that can be extracted
        L_sequences = length_of_one_traj - self.total_required_len_in_timesteps +1 #+1 because both ends are included

        dataset_length = number_of_rows * L_sequences
        dataset_metadata["L_sequences"] = L_sequences
        return dataset_length

    def _get_item_from_h5_file(self, metadata, dataset_idx):
        L = metadata["L_sequences"]
        traj_idx = dataset_idx // L
        seq_idx = dataset_idx % L

        # make the dataitem
        # index values for no AR, no future_cmd_pred
        laser_start = seq_idx
        laser_end = seq_idx+self.len_of_one_seq
        cmd_start = seq_idx+self.len_of_one_seq
        cmd_end = seq_idx+self.len_of_one_seq + 1
        goal_idx = seq_idx+self.len_of_one_seq -1
        # goal_end = seq_idx+self.len_of_one_seq + 1

        if self.autoregressive:
            laser_start += 1
            laser_end += 1 
            cmd_start = seq_idx
            goal_idx += 1

        if self.future_cmd_pred > 0:
            cmd_end += self.future_cmd_pred
        
        goal = metadata["h5_file_pointer"]["goal"][traj_idx, goal_idx]
        laser = metadata["h5_file_pointer"]["laser"][traj_idx, laser_start : laser_end]
        cmd = metadata["h5_file_pointer"]["cmd"][traj_idx, cmd_start : cmd_end]

        return {"goal": goal, "laser":laser, "cmd": cmd}

class Regular_Render_Dataset(Demo_2D_Dataset):

    def _get_demo_path(self, demo_dir):
        seq_dir = str(self.params.model_params.sequence_length) +"_seq"
        return os.path.join("/scratch", os.getenv("USER"), "hallucination_new", "LfH_demo", seq_dir, demo_dir)
    
    def _get_dataset_length(self, h5_laser_dataset):
        return len(h5_laser_dataset)

    def _get_item_from_h5_file(self, metadata, dataset_idx):
        return {k: metadata["h5_file_pointer"][k][dataset_idx] for k in metadata["h5_file_pointer"] if k in self.headers}


"""   All below transformations take in laser scan of shape (T, 720) where T = |time_steps|,720 laser scans """
class Flip(object):
    """Right - Left Flip. If object is on right, flip left & vice versa."""

    def __init__(self,params):
        self.autoregressive = params.model_params.autoregressive

    def __call__(self, data):
        if np.random.rand() > 0.5:
            # laser = data["laser"]
            # # flip_wo_time = np.flip(laser[:, :-1], axis=1)
            # # time = laser[:,-1].reshape(-1,1)
            data["laser"] = np.flip(data["laser"], axis=1)
            if self.autoregressive:
                data["cmd"][:, 1] *= -1
            else:
                data["cmd"][1] *= -1

            
            data["goal"][1] *= -1
        return data


class Clip(object):
    """Clip laser scans to laser_max_range"""

    def __init__(self, laser_max_range=5):
        self.laser_max_range = laser_max_range

    def __call__(self, data):
        data["laser"] =  np.minimum(data["laser"], self.laser_max_range)
        return data


class Noise(object):
    """Adds Gausian noise to laser scans"""

    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale

    def __call__(self, data):
        data["laser"] += np.random.normal(0, self.noise_scale, size=data["laser"].shape)
        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,):
        pass

    def __call__(self, data):
        new_data = {}
        for key, val in data.items():
            val = val.astype(np.float32)
            new_data[key] = torch.from_numpy(val)
        return new_data
