import os
import time
import json
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataloader import Single_Render_Dataset, Regular_Render_Dataset, Flip, Clip, Noise, ToTensor
from models import MotionPlannerTransformer

torch.autograd.set_detect_anomaly(True)

DATASET_TYPES = ["osa", "static", "dynamic", "only_close", "dynamic_accel", "dynamic_jerk"]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TrainingParams:
    def __init__(self, training_params_fname="params.json"):
        self.repo_path = repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        params_path = os.path.join(repo_path, "LfD_2D", training_params_fname)
        config = json.load(open(params_path))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = AttrDict(self._clean_dict(v))
            _dict[k] = v
        return _dict

    def dump_dict(self, train=True):
        if train:
            self.rslts_dir = os.path.join(self.repo_path, "rslts", "LfD_2D_rslts", time.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(self.rslts_dir)
            json.dump(self.__dict__, open(os.path.join(self.rslts_dir, "params.json"), "w"), indent=4)


class LfD_2D_dynamic_model(nn.Module):
    def __init__(self, params):
        super(LfD_2D_dynamic_model, self).__init__()
        self.params = params
        self.model_type =  params.training_params.model_type
        self.autoregressive = True if hasattr(params.model_params, "autoregressive") and params.model_params.autoregressive else False
        self.num_pred_cmd = 1 
        if self.autoregressive:
            print("Using autoregressive model")

        if self.model_type == "trans":
            if hasattr(params.transformer_params, "num_pred_cmd"):
                self.num_pred_cmd = params.transformer_params.num_pred_cmd 
            self.model_params = model_params = params.transformer_params
            self.transformer = MotionPlannerTransformer(lidar_dim= model_params.lidar_dim,
                 goal_dim = model_params.goal_dim,
                 cmd_dim = 2,
                 action_dim = model_params.action_dim,
                 history_length = model_params.history_length,
                 embed_dim = model_params.embed_dim,
                 nhead = model_params.nhead,
                 num_encoder_layers = model_params.num_encoder_layers,
                 dim_feedforward = model_params.dim_feedforward,
                 dropout = model_params.dropout,
                 cnn_channels = model_params.cnn_channels,
                 cnn_kernel_size = model_params.cnn_kernel_size,
                 num_pred_cmd = self.num_pred_cmd
                 )
            print("Using transformer")
        else:
            raise Exception("LfD model: model_type must be one of [trans]. but was "+self.model_type)
   
    def forward(self, laser, goal, arcmd=None):
        # to make it match zizhao's code, laser + goal will be inputted into rnn model
        # laser: (N, L, 720)
        # cmd: (N, L, 2) if cmds_as_inputs==True else (N, 2)
        # goal: (N, 2)
        if self.model_type == "trans":
            # The training loop reshapes the goal tensor to (N, 1, 2).
            # The MotionPlannerTransformer expects a goal of shape (N, goal_dim), e.g., (N, 2).
            # We squeeze the dimension of size 1 to match the required input shape.
            if goal.dim() == 3 and goal.shape[1] == 1:
                goal = goal.squeeze(1)
            return self.transformer(laser, goal, arcmd)
        


def train(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params.device = device
    training_params = params.training_params
    noise_scale = params.model_params.noise_scale
    laser_max_range = params.model_params.laser_max_range

    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))

    train_transform = transforms.Compose([Flip(params), Clip(laser_max_range), Noise(noise_scale), ToTensor()])
    test_val_transform = transforms.Compose([Clip(laser_max_range), ToTensor()])


    train_dataset = Single_Render_Dataset(params, split="train", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=training_params.batch_size, persistent_workers=True, pin_memory=True, shuffle=True, num_workers=training_params.num_workers) #training_params.num_workers, pin_memory = True,persistent_workers = True
   
    val_test_datasets = {"val":[], "test":[]}
    val_test_dataloaders = {"val":[], "test":[]}
    for dataset_type in DATASET_TYPES:
        for split in ["val", "test"]:
            ds = Single_Render_Dataset(params, split=split, transform=test_val_transform, test_over_dataset=dataset_type)
            if len(ds) > 0:
                val_test_datasets[split].append((dataset_type, ds))
                dl = DataLoader(ds, batch_size=training_params.batch_size, persistent_workers=True, pin_memory=True, shuffle=False, num_workers=training_params.num_workers)
                val_test_dataloaders[split].append((dataset_type, dl))
       
    model = LfD_2D_dynamic_model(params).to(device)
    if training_params.load_model is not None and os.path.exists(training_params.load_model):
        model.load_state_dict(torch.load(training_params.load_model, map_location=torch.device("cpu")))
        model.to(device)
        print("loaded model: ", training_params.load_model)
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    # model saving
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    
    # training loop
    for epoch in range(training_params.epochs):
        metrics = {"train_loss": [], "train_used_cmd_linear":[], "train_used_cmd_angular":[]}
        # train_losses = []
        model.train(mode=True)
        for i_batch, sample_batched in enumerate(train_dataloader):
            for key, val in sample_batched.items():
                sample_batched[key] = val.to(device)

            laser = sample_batched["laser"]
            goal = sample_batched["goal"]
            cmd = sample_batched["cmd"]

            goal = goal.reshape(-1, 1, 2)

            optimizer.zero_grad()
            
            if model.autoregressive:
                N = model.num_pred_cmd
                if N > 1:
                    T = cmd.shape[1]
                    T_in = T - N
                    
                    laser_in = laser[:, :T_in, :]
                    arcmd_in = cmd[:, :T_in, :]
                    
                    logits = model(laser_in, goal, arcmd=arcmd_in)
                    target_blocks = [cmd[:, t+1:t+1+N, :] for t in range(T_in)]
                    cmd_tgt = torch.stack(target_blocks, dim=1)
                    # loss = F.mse_loss(logits, cmd_tgt)
                    loss = torch.mean(torch.sum((logits - cmd_tgt)**2, dim=-1))
                    used_linear_loss = torch.mean((logits[:, -1, 0, 0]-cmd_tgt[:, -1, 0, 0])**2)
                    used_angular_loss = torch.mean((logits[:, -1, 0, 1]-cmd_tgt[:, -1, 0, 1])**2)
                else: 
                    cmd_in    = cmd[:, :-1]        # (B, T-1, 2)
                    cmd_tgt   = cmd[:,  1:]        # (B, T-1, 2)
                    logits    = model(laser, goal, arcmd=cmd_in)  # (B, T-1, 2)
                    # loss      = F.mse_loss(logits, cmd_tgt, reduction="mean")
                    loss = torch.mean(torch.sum((logits - cmd_tgt)**2, dim=-1))
                    used_linear_loss = torch.mean((logits[:, -1, 0]-cmd_tgt[:, -1, 0])**2)
                    used_angular_loss = torch.mean((logits[:, -1, 1]-cmd_tgt[:, -1, 1])**2)
            else:
                cmd_pred = model(laser, goal, None)
                loss = torch.mean(torch.sum((cmd - cmd_pred) ** 2, dim=-1))
                used_linear_loss = torch.mean((logits[:, 0]-cmd_tgt[:, 0])**2)
                used_angular_loss = torch.mean((logits[:, 1]-cmd_tgt[:, 1])**2)
            
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.model_params.gradient_clip_max_norm)
            optimizer.step()

            metrics["train_loss"].append(loss.item())
            metrics["train_used_cmd_angular"].append(used_angular_loss.item())
            metrics["train_used_cmd_linear"].append(used_linear_loss.item())
            # train_losses.append(loss.item())

            if i_batch + 1 == training_params.batch_per_epoch:
                break


        if (epoch + 1) % training_params.report_freq == 0:
            print("{}/{}. Loss: {:.5f}".format(epoch + 1, training_params.epochs, loss.item()))
            if writer is not None:
                for k in metrics.keys():
                    writer.add_scalar(f"LfD/{k}", np.mean(metrics[k]), epoch)
                # writer.add_scalar("LfD/train_loss", np.mean(metrics["train_loss"]), epoch)

        if (epoch+1) % training_params.test_freq == 0:
            model.train(mode=False)
            
            for split_type, val_test_dataloaders_list in val_test_dataloaders.items():
                for ds_type, loader in val_test_dataloaders_list:
                    # test_val_losses = []
                    metrics = {"loss": [], "used_cmd_linear":[], "used_cmd_angular":[]}
                    for i_batch, sample_batched in enumerate(loader):
                        for key, value in sample_batched.items():
                            sample_batched[key] = value.to(device)

                        laser = sample_batched["laser"]
                        goal = sample_batched["goal"]
                        cmd = sample_batched["cmd"]

                        goal = goal.reshape(-1, 1, 2)

                        with torch.no_grad():
                            if model.autoregressive:
                                N = model.num_pred_cmd
                                if N > 1:
                                    T = cmd.shape[1]
                                    T_in = T - N
                                    
                                    laser_in = laser[:, :T_in, :]
                                    arcmd_in = cmd[:, :T_in, :]
                                    
                                    logits = model(laser_in, goal, arcmd=arcmd_in)
                                    target_blocks = [cmd[:, t+1:t+1+N, :] for t in range(T_in)]
                                    cmd_tgt = torch.stack(target_blocks, dim=1)                       
                                    # loss = F.mse_loss(logits, cmd_tgt)
                                    loss = torch.mean(torch.sum((logits - cmd_tgt)**2, dim=-1))
                                    used_linear_loss = torch.mean((logits[:, -1, 0, 0]-cmd_tgt[:, -1, 0, 0])**2)
                                    used_angular_loss = torch.mean((logits[:, -1, 0, 1]-cmd_tgt[:, -1, 0, 1])**2)
                                else: 
                                    cmd_in    = cmd[:, :-1]        # (B, T-1, 2)
                                    cmd_tgt   = cmd[:,  1:]        # (B, T-1, 2)
                                    logits    = model(laser, goal, arcmd=cmd_in)  # (B, T-1, 2)
                                    # loss      = F.mse_loss(logits, cmd_tgt, reduction="mean")
                                    loss = torch.mean(torch.sum((logits - cmd_tgt)**2, dim=-1))
                                    used_linear_loss = torch.mean((logits[:, -1, 0]-cmd_tgt[:, -1, 0])**2)
                                    used_angular_loss = torch.mean((logits[:, -1, 1]-cmd_tgt[:, -1, 1])**2)
                            else:
                                cmd_pred = model(laser, goal, None)
                                loss = torch.mean(torch.sum((cmd - cmd_pred) ** 2, dim=-1))
                                used_linear_loss = torch.mean((logits[:, 0]-cmd_tgt[:, 0])**2)
                                used_angular_loss = torch.mean((logits[:, 1]-cmd_tgt[:, 1])**2)

                            # test_val_losses.append(loss.item())
                            metrics["loss"].append(loss.item())
                            metrics["used_cmd_angular"].append(used_angular_loss.item())
                            metrics["used_cmd_linear"].append(used_linear_loss.item())

                    if writer is not None:
                        for k in metrics.keys():
                            writer.add_scalar(f"LfD/{split_type}_loss/{ds_type}_{k}", np.mean(metrics[k]), epoch)
                        # writer.add_scalar(f"LfD/{split_type}_loss/{ds_type}", np.mean(test_val_losses), epoch)


        if (epoch + 1) % training_params.saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_{}".format(epoch + 1)),
                       pickle_protocol=2, _use_new_zipfile_serialization=False)


    # close file pointers
    train_dataset.close_h5_file_pointers()
    for _, dataset_lists in val_test_datasets.items():
        for _, dataset in dataset_lists:
            dataset.close_h5_file_pointers()


def parse_args(params):
    seq_lens = [3,5,10,20]
    cmd_preds = [1,3,5,10]
    demo_dirs = [None]

    #parse arg trans
    model_types = ["trans"]
   
    seq_choices = [i for i in range(len(seq_lens))]
    seq_choices.insert(0, -1)
    demo_choices = [i for i in range(len(demo_dirs))]
    demo_choices.insert(0, -1)
    model_choices = [i for i in range(len(model_types))]

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len_idx", default=-1, type=int,choices=seq_choices, help=f"The Sequence length is set to {seq_lens}. seq_len_idx gets the corresponding value. Default: 5 which gets the sequence length in Params.")
    parser.add_argument("--cmd_predictions_idx", default=0, type=int, help=f"The Command predictions length is set to {cmd_preds}. cmd_predictions_idx gets the corresponding value. Default: 0 which sets the value to 1.")

    parser.add_argument("--demo_dir_idx", default=-1, type=int, choices=demo_choices, help=f"Along with the current demo_directories set in Params, one of the following is appended to it - {demo_dirs}. Default: -1 which doesnt append any demo dir.")
    parser.add_argument("--model_type_idx", default=0, type=int, choices=model_choices, help=f"LfD model types: {model_types}. Default: 0 - lstm.")
    
    # parser.add_argument("--seq_length", default=5, type=int, help=f"The Sequence length is now dynamic. Default it is 5 (0.5s) but can be set to any integer >= 1. Warning will be given if its too high.")
    # parser.add_argument("--cmd_predictions_idx", default=0, type=int, help=f"The number of commands that the model will predict. Includes the current cmd, can be set to integer >= 1. Default is 1. Warning will be given if its too high.")
    
    args = parser.parse_args()
   
    # if cmd_preds[args.cmd_predictions_idx] > seq_lens[args.seq_len_idx]:
    #     # print(f"num_cmd_preds ({cmd_preds[args.cmd_predictions_idx]}) > sequence length ({seq_lens[args.seq_len_idx]}). Exiting program.")
    #     raise ValueError(f"num_cmd_preds ({cmd_preds[args.cmd_predictions_idx]}) > sequence length ({seq_lens[args.seq_len_idx]}). Exiting program.")

    if cmd_preds[args.cmd_predictions_idx] + seq_lens[args.seq_len_idx] > 47:
        # print(f"num_cmd_preds ({cmd_preds[args.cmd_predictions_idx]}) + sequence length ({seq_lens[args.seq_len_idx]}) > default length of trajectory (47)")
        raise ValueError(f"num_cmd_preds ({cmd_preds[args.cmd_predictions_idx]}) + sequence length ({seq_lens[args.seq_len_idx]}) > default length of trajectory (47)")


    # if args.seq_len_idx != -1:
    #     seq_len = seq_lens[args.seq_len_idx]
    #     params.model_params.sequence_length = seq_len
    #     params.predictor_params.sequence_length = seq_len
    #     # params.predictor_params.sequence_dt = float(seq_len[args.seq_len_idx])/10 fixed to 1/40 -> changed to 1/20
    #     if hasattr(params, 'transformer_params'):
    #         params.transformer_params.history_length = seq_len

    if args.demo_dir_idx != -1:
        if demo_dirs[args.demo_dir_idx] is not None:
            params.demo_dirs.extend(demo_dirs[args.demo_dir_idx])

    params.training_params.model_type = model_types[args.model_type_idx]

    if hasattr(params, "dataset_params") and params.dataset_params.dataset_type == "single_render":
        params.dataset_params.num_pred_cmd = cmd_preds[args.cmd_predictions_idx]
        params.dataset_params.sequence_length = seq_lens[args.seq_len_idx]
        if hasattr(params, "transformer_params") and params.training_params.model_type == "trans":
            params.transformer_params.num_pred_cmd = cmd_preds[args.cmd_predictions_idx]
            params.transformer_params.history_length = seq_lens[args.seq_len_idx]
        params.predictor_params.sequence_length = seq_lens[args.seq_len_idx]
        if hasattr(params.predictor_params, "num_cmd_pred"):
            params.predictor_params.num_cmd_pred = cmd_preds[args.cmd_predictions_idx]

    num_workers = os.environ.get("SLURM_CPUS_PER_TASK") if os.environ.get("SLURM_CPUS_PER_TASK") is not None else 1
    params.training_params.num_workers = int(num_workers)

    print("Seq Lengthn: ", params.dataset_params.sequence_length)
    print("Number Cmd Prediction: ", params.dataset_params.num_pred_cmd)
    


if __name__ == "__main__":
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    slurm_task_id = 0 if slurm_task_id is None else int(slurm_task_id)
    time.sleep(slurm_task_id*30) # for job arrays to have different timestamps

    params = TrainingParams()
    parse_args(params)
    params.dump_dict()

    start_time = time.perf_counter()
    train(params)
    end_time = time.perf_counter()
    print("num_cpus_per_task:", params.training_params.num_workers)
    print("---- %s seconds ----" % (end_time - start_time))
