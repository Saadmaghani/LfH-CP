"""
Training and sampling module for the Hallucination model.

This module provides functionality for training a hallucination model using PyTorch,
including annealing schedules for loss parameters and model checkpointing.
"""

import os
import time
import json
import shutil
from typing import Dict, Any, Tuple, Optional
import traceback

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import TrainingParams, plot_opt, plot_obs_dist, plot_time_dist

#=================================
# Helper methods
#=================================
def anneal_gumbel_tau(cur_ts2_epoch: int, params: TrainingParams, model: Hallucination) -> None:
    """
    Update the Gumbel-softmax temperature (tau) based on annealing schedule.
    
    Supports three decay types: exponential, step-wise, and fixed. The tau parameter
    controls the softness of the Gumbel-softmax distribution, gradually transitioning
    from soft to hard (one-hot) distributions.
    
    Args:
        cur_ts2_epoch: Current epoch within training stage 2.
        total_ts2_epochs: Total number of epochs in training stage 2.
        params: Training parameters object containing model configuration.
        model: The Hallucination model instance to update.
        
    Raises:
        ValueError: If gumbel_decay_type is not one of ['exponential', 'step', 'fixed'].
    """
    model_params = params.model_params
    min_gumbel_tau = model_params.gumbel_final_tau
    max_gumbel_tau = model_params.gumbel_starting_tau
    tau = max_gumbel_tau
    decay_type = model_params.gumbel_decay_type
    annealing_steps = model_params.gumbel_annealing_steps

    if decay_type == "exponential":
        decay_rate = np.clip(np.log(max_gumbel_tau/min_gumbel_tau)/(annealing_steps), a_min=0, a_max=None) # decay over annealing_steps and then the remaining is one-hot
        tau = min_gumbel_tau + (max_gumbel_tau - min_gumbel_tau)*np.exp(-decay_rate * cur_ts2_epoch)
    elif decay_type == "step":
        decay_step = model_params.gumbel_decay_step
        current_tau = model.encoder.gumbel_softmax.get_current_tau()
        if (cur_ts2_epoch +1) % decay_step == 0 and current_tau > min_gumbel_tau:
            tau = current_tau * 0.5 # half tau every decay_step steps
    elif decay_type == "fixed":
        tau = max_gumbel_tau # no change in tau
    else:
        raise ValueError("decay type can only be ['exponential', 'step', 'fixed']. Got "+decay_type+". See gumbel_decay_type in params.json.")
    model.encoder.gumbel_softmax.set_current_tau(tau)

def anneal_time_idx_entropy_lambda(cur_ts2_epoch: int, total_ts2_epochs: int, params: TrainingParams) -> None:
    """
    Update the time index entropy loss weight using linear annealing.
    
    Gradually increases the lambda coefficient for time index entropy loss from 0
    to its final value over the course of training stage 2.
    
    Args:
        cur_ts2_epoch: Current epoch within training stage 2.
        total_ts2_epochs: Total number of epochs in training stage 2.
        params: Training parameters object containing model configuration.
    """
    model_params = params.model_params
    annealing_coef = np.clip(cur_ts2_epoch / total_ts2_epochs, 0, 1)
    model_params.lambda_time_idx_entropy_adjusted = model_params.lambda_final_time_idx_entropy*annealing_coef

def anneal_repulsion_lambdas(cur_ts2_epoch: int, params: TrainingParams) -> None:
    model_params = params.model_params
    lambda_mutual_repulsion_final = model_params.lambda_mutual_repulsion
    lambda_reference_repulsion_final = model_params.lambda_reference_repulsion
    repulsion_annealing_steps = model_params.repulsion_annealing_steps

    repulsion_annealing_coef = np.clip((cur_ts2_epoch + 1.) / repulsion_annealing_steps, 0, 1)

    model_params.lambda_mutual_repulsion_adjusted = lambda_mutual_repulsion_final * repulsion_annealing_coef
    model_params.lambda_reference_repulsion_adjusted = lambda_reference_repulsion_final * repulsion_annealing_coef
   
def anneal_regularization_lambdas(epoch: int, params: TrainingParams) -> None:
    """
    Update all loss function weight coefficients (lambda values) based on epoch.
    
    Applies annealing schedules to regularization and repulsion loss weights
    to balance training objectives across epochs.
    
    Args:
        epoch: Current training epoch.
        params: Training parameters object containing model configuration and loss weights.
    """
    model_params = params.model_params
    lambda_loc_reg_final = model_params.lambda_loc_reg
    lambda_size_kl_final = model_params.lambda_size_kl
    reg_final_prop = model_params.reg_final_prop
    reg_annealing_steps = model_params.reg_annealing_steps
    
    reg_annealing_coef = np.clip(1. - epoch / reg_annealing_steps, 0, 1)
    
    model_params.lambda_loc_reg_adjusted = lambda_loc_reg_final * (reg_annealing_coef * (1 - reg_final_prop) + reg_final_prop)
    model_params.lambda_size_kl_adjusted = lambda_size_kl_final * (reg_annealing_coef * (1 - reg_final_prop) + reg_final_prop)
    
def set_training_stage(current_training_stage: int, params: TrainingParams, epoch: int, model: Hallucination) -> int:
    """
    Determine and set the current training stage based on the epoch.
    
    The model supports multiple training stages with different objectives and
    constraints. This function transitions between stages and updates the model
    when stage boundaries are reached.
    
    Args:
        current_training_stage: The current training stage (1, 2, or 3).
        params: Training parameters object containing stage transition epochs.
        epoch: Current training epoch.
        model: The Hallucination model instance to update.
        
    Returns:
        int: The updated training stage (1, 2, or 3).
    """
    model_params = params.model_params
    if epoch >= model_params.training_stage_2_at_epoch and epoch < model_params.training_stage_3_at_epoch and current_training_stage != 2:
        print("="*50)
        print(">"*15,"TRAINING STAGE 2 STARTED","<"*15)
        print("="*50)
        model.set_trainingstage(2)
        current_training_stage = 2

    if epoch >= model_params.training_stage_3_at_epoch and current_training_stage != 3:
        print("="*50)
        print(">"*15,"TRAINING STAGE 2 STARTED","<"*15)
        print("="*50)
        model.set_trainingstage(3)
        current_training_stage = 3
    return current_training_stage 

def setup_adjusted_lambdas(params: TrainingParams) -> None:
    params.model_params.lambda_loc_reg_adjusted = 0
    params.model_params.lambda_size_kl_adjusted = 0

    params.model_params.lambda_mutual_repulsion_adjusted = 0 
    params.model_params.lambda_reference_repulsion_adjusted = 0

    params.model_params.lambda_time_idx_entropy_adjusted = 0

#=================================
# Main methods
#=================================
def train(params: TrainingParams) -> None:
    """
    Train the Hallucination model using the provided parameters.
    
    Performs multi-epoch training with learning rate optimization, loss annealing,
    model checkpointing, and tensorboard logging.
    
    Args:
        params: Training parameters object containing all configuration settings,
            model parameters, and paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params.device = device
    training_params = params.training_params
    model_params = params.model_params
    num_workers = training_params.num_workers

    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    dataset = HallucinationDataset(params, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=training_params.batch_size, shuffle=True, num_workers=num_workers)


    starting_epoch = 0
    model = Hallucination(params, writer).to(device)
    if training_params.load_model is not None:
        if os.path.exists(training_params.load_model):
            model.load_state_dict(torch.load(training_params.load_model, map_location=torch.device("cpu")))
            model.to(device)
            starting_epoch = training_params.load_model_at_epoch
        else:
            raise FileNotFoundError(f"model file does not exist at: {training_params.load_model}.")

    optimizer = optim.AdamW(model.parameters(), lr=training_params.lr, weight_decay=training_params.weight_decay)

    # model saving
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    current_training_stage = 1
    
    for epoch in range(starting_epoch, training_params.epochs):
        loss_details = []
        model.train(training=True)

        current_training_stage = set_training_stage(current_training_stage, params, epoch, model)

        anneal_regularization_lambdas(epoch, params)

        if current_training_stage == 2:
            cur_ts2_epoch = epoch - model_params.training_stage_2_at_epoch
            total_ts2_epochs = model_params.training_stage_3_at_epoch -model_params.training_stage_2_at_epoch
            anneal_time_idx_entropy_lambda(cur_ts2_epoch, total_ts2_epochs, params)
            anneal_gumbel_tau(cur_ts2_epoch, params, model)
            anneal_repulsion_lambdas(cur_ts2_epoch, params)
    

        for i_batch, batch_data in enumerate(dataloader):
            for key, val in batch_data.items():
                batch_data[key] = val.to(device)

            full_traj = batch_data["full_traj"]
            reference_pts = batch_data["reference_pts"]
            traj = batch_data["traj"]

            optimizer.zero_grad()
            
            try:
                recon_traj, recon_control_points, location_tup, size_tup, _, time_index = model(full_traj, reference_pts, decode=True)
                loss, loss_detail = model.loss(full_traj, traj, recon_traj, reference_pts,
                                               location_tup[0], location_tup[1], location_tup[2], size_tup[0], size_tup[1], size_tup[2], time_index)
            except AssertionError as e:
                print("----------------------------")
                print(f"LFH_MAIN: error at {epoch+1}/{training_params.epochs}, {i_batch+1}/{training_params.batch_per_epoch}")
                traceback.print_exc()
                print("----------------------------")
                continue

            loss.backward()
            print(loss_detail)
            optimizer.step()

            loss_details.append(loss_detail)

            print(f"{epoch + 1}/{training_params.epochs}, {i_batch + 1}/{training_params.batch_per_epoch}")
            if len(loss_details) >= training_params.batch_per_epoch:
                break
        
        # Optional plotting but it greatly slows down training
        if training_params.plot_freq > 0 :
            if epoch % training_params.plot_freq == 0:
                plot_opt(writer, reference_pts, recon_control_points, location_tup[2], size_tup[2], epoch)
                plot_time_dist(writer, model, time_index, epoch)
                plot_obs_dist(writer, params, full_traj, location_tup[1], location_tup[2], size_tup[1], size_tup[2], epoch)
                        
        if writer is not None:
            # list of dict to dict of list
            loss_details = {k: [dic[k] for dic in loss_details] for k in loss_details[0]}
            for k, v in loss_details.items():
                writer.add_scalar("train/{}".format(k), np.mean(v), epoch)

        if (epoch + 1) % training_params.saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_{}".format(epoch + 1)),
                       pickle_protocol=2, _use_new_zipfile_serialization=False)

    if writer is not None:
        writer.close()

def sample(
    params: TrainingParams,
    model_path: str,
    should_decode: bool = True,
    n_times: int = 10,
    training_stage: int = 1,
    seed: int = 3
) -> Dict[str, Any]:
    """
    Sample from a trained Hallucination model and extract intermediate activations.
    
    Loads a trained model, registers forward hooks to capture intermediate layer
    outputs, and performs a forward pass on a single batch of data.
    
    Args:
        params: Training parameters object containing model and data configuration.
        model_path: Path to the trained model checkpoint file.
        should_decode: Whether to decode the model's latent representation. Defaults to True.
        n_times: Number of samples to draw in the batch. Defaults to 10.
        training_stage: Training stage to set for the model. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to 3.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - trajectories: 'traj', 'raw_traj', 'raw_ori', 'recon_traj', 'reference_pts'
            - location outputs: 'loc_mu', 'loc_log_var', 'loc'
            - velocity outputs: 'vel_mu', 'vel_log_var', 'vel'
            - size outputs: 'size_mu', 'size_log_var', 'size'
            - temporal outputs: 'time_idx'
            - model outputs: 'hook_output', 'loss_detail', 'model'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params.device = device
    num_workers = params.training_params.num_workers

    dataset = HallucinationDataset(params, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=n_times, shuffle=True, num_workers=num_workers)
    
    model = Hallucination(params, None).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.to(device)
    
    
    def register_hooks(
        module: torch.nn.Module,
        hook_fn,
        outputs_dict: Dict[str, Dict[str, torch.Tensor]],
        prefix: str = ""
    ) -> None:
        """
        Recursively registers forward hooks in all submodules of a given module.
        
        Args:
            module (nn.Module): The root module to traverse.
            hook_fn (function): The hook function to apply.
            outputs_dict (dict): Dictionary to store the outputs of each layer.
            prefix (str): Naming prefix for layers (used for better debugging/logging).
        """
        for name, submodule in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Register the hook for the current submodule
            if isinstance(submodule, torch.nn.Linear) or isinstance(submodule, torch.nn.Conv1d):
                submodule.register_forward_hook(lambda mod, inp, out, n=full_name: hook_fn(mod, inp, out, n, outputs_dict))
            
            # Recursively apply to submodules (for handling nested structures like Sequential)
            register_hooks(submodule, hook_fn, outputs_dict, full_name)
                

    model.set_trainingstage(training_stage)
    model.train(training=False)


    def hook_fn(
        module: torch.nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
        name: str,
        outputs_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        outputs_dict[name] = {}
        outputs_dict[name]["input"] = input[0].detach()  # Save the output and detach it from the computation graph
        outputs_dict[name]["output"] = output.detach()  # Save the output and detach it from the computation graph

    hook_outputs = {}
    # Register hooks
    register_hooks(model.encoder, hook_fn, hook_outputs)

    torch.manual_seed(seed)
    with torch.no_grad():
        for _, batch_data in enumerate(dataloader): # sample once
            for key, val in batch_data.items():
                batch_data[key] = val.to(device)

            full_traj = batch_data["full_traj"]
            reference_pts = batch_data["reference_pts"]
            traj = batch_data["traj"]
            raw_traj = batch_data["odom_frame_traj"]
            raw_ori = batch_data["odom_frame_ori_traj"]

            try:
                recon_traj, recon_control_points, location_tup, size_tup, velocity_tup, time_index  = model(full_traj, reference_pts, decode=should_decode)
                loss, loss_detail = model.loss(full_traj, traj, recon_traj, reference_pts,
                                                location_tup[0], location_tup[1], location_tup[2], size_tup[0], size_tup[1], size_tup[2], time_index)
            except AssertionError as e:
                print("----------------------------")
                print("LFH_MAIN sample: error")
                traceback.print_exc()
                print("----------------------------")
                continue
            
            break


    return {"traj":traj, "raw_traj":raw_traj, "raw_ori":raw_ori, "recon_traj":recon_traj, "reference_pts": reference_pts,
            "loc_mu":location_tup[0], "loc_log_var":location_tup[1], "loc":location_tup[2],
            "vel_mu":velocity_tup[0], "vel_log_var": velocity_tup[1], "vel": velocity_tup[2], 
            "size_mu":size_tup[0], "size_log_var": size_tup[1], "size": size_tup[2], 
            "time_idx":time_index, "hook_output":hook_outputs,
            "loss_detail":loss_detail, "model": model}

if __name__ == "__main__":
    params = TrainingParams()
    setup_adjusted_lambdas(params)
    train(params)
