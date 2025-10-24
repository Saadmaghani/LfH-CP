import numpy as np
from scipy.interpolate import BSpline
import logging
logger = logging.Logger('catch_all')

import torch
import torch.nn as nn
import torch.nn.functional as F

import diffcp
from solver import CVX_Decoder, EPS
from utils import plot_opt

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelSoftmax:
    def __init__(self, model_params, training_stage):
        self.current_tau = model_params.gumbel_starting_tau
        self.final_tau = model_params.gumbel_final_tau
        self.training_stage = training_stage

    def __call__(self, logits):
        if self.current_tau > self.final_tau and self.training_stage == 2:
            # print("GumbelSoftmax - current_tau:", self.current_tau)
            dist_sampled_tensor = F.gumbel_softmax(logits, tau=self.current_tau, hard=False)
            # time_index = time_index / time_index.max(dim=-1, keepdim=True)[0]
        else:
            # print("GumbelSoftmax - one-hot encoding")
            dist_sampled_tensor = F.gumbel_softmax(logits, tau=self.current_tau, hard=True) 
        return dist_sampled_tensor
    
    def set_current_tau(self, value):
        self.current_tau = value
    
    def set_trainingstage(self, value):
        self.training_stage = value

    def  get_current_tau(self):
        return self.current_tau

# Mixture of Softmaxes for time_heads
class MixtureOfGumbelSoftmaxes(nn.Module):
    def __init__(self, input_dim, num_classes, num_components, GumbelSoftmaxObject):
        """
        Mixture of Softmaxes Model
        
        Args:
            input_dim (int): Dimension of input features
            num_classes (int): Number of output classes (categories)
            num_components (int): Number of softmax components in the mixture
        """
        super(MixtureOfGumbelSoftmaxes, self).__init__()
        self.gumbel_softmax = GumbelSoftmaxObject
        self.num_components = num_components
        self.num_classes = num_classes
        
        # Shared input projection to component-specific representations
        self.fc_hidden = nn.Linear(input_dim, num_components * num_classes)
        
        # Mixture weights (gating network)
        self.fc_gate = nn.Linear(input_dim, num_components)

    def forward(self, x):
        """
        Forward pass for Mixture of Softmaxes.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor: Mixture probability distribution over classes (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # Compute raw logits for each mixture component (batch_size, num_components, num_classes)
        logits = self.fc_hidden(x).view(batch_size, self.num_components, self.num_classes) # (B, K, C)
        
        # Compute softmax over classes for each component
        softmax_outputs = self.gumbel_softmax(logits) # (B, K, C)
        # softmax_outputs = F.softmax(logits, dim=2)  # (batch_size, num_components, num_classes)
        
        # Compute gating weights (mixture coefficients)
        gate_logits = self.fc_gate(x)  # (batch_size, num_components)
        gate_weights = F.softmax(gate_logits, dim=1)  # (batch_size, num_components)

        # Compute final probability as the weighted sum of component softmaxes
        output = torch.sum(gate_weights.unsqueeze(2) * softmax_outputs, dim=1)  # (batch_size, num_classes)
        
        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.model_params = model_params = params.model_params
        self.auto_regressive = model_params.auto_regressive

        self.gumbel_softmax = GumbelSoftmax(model_params=model_params, training_stage=1)
        
        self.conv1d_layers = nn.ModuleList()
        output_len = model_params.full_traj_len
        prev_channel = 2 * params.Dy
        for channel, kernel_size, stride in \
                zip(model_params.encoder_channels, model_params.kernel_sizes, model_params.strides):
            self.conv1d_layers.append(nn.Conv1d(in_channels=prev_channel, out_channels=channel,
                                                kernel_size=kernel_size, stride=stride))
            prev_channel = channel
            output_len = int((output_len - kernel_size) / stride + 1)
        conv1d_hidden_dim = output_len * model_params.encoder_channels[-1]

        num_var_modeled = 2 if model_params.model_variance else 1
        num_var_modeled += model_params.num_motion_vars


        Dy = params.Dy
        if self.auto_regressive:
            self.fcs = nn.ModuleList()
            self.time_heads = nn.ModuleList()
            self.motion_heads = nn.ModuleList()

            for i in range(model_params.num_obs):
                input_size = conv1d_hidden_dim + (Dy * num_var_modeled + model_params.num_control_pts) * i

                if not hasattr(model_params, "auto_regressive_hidden_size"):
                    # old model 
                    self.fcs.append(nn.Linear(input_size, Dy * (2 if model_params.model_variance else 1) * 2))
                    self.time_heads.append(nn.Linear(input_size, model_params.num_control_pts))
                    self.motion_heads.append(nn.Linear(input_size, Dy * model_params.num_motion_vars * 2))
                elif hasattr(model_params, "time_heads_MoS_num_components") and model_params.time_heads_MoS_num_components > 1:

                    make_ar_hidden_layer = lambda input_size : nn.Sequential(nn.Linear(input_size, model_params.auto_regressive_hidden_size), nn.ReLU()) if model_params.auto_regressive_hidden_size > 0 else nn.Identity()
                    hidden_size = model_params.auto_regressive_hidden_size if model_params.auto_regressive_hidden_size > 0 else input_size

                    self.fcs.append(nn.Sequential(make_ar_hidden_layer(input_size), nn.Linear(hidden_size, Dy * (2 if model_params.model_variance else 1) * 2)))
                    self.time_heads.append(nn.Sequential(make_ar_hidden_layer(input_size), MixtureOfGumbelSoftmaxes(hidden_size, model_params.num_control_pts, model_params.time_heads_MoS_num_components, self.gumbel_softmax)))
                    self.motion_heads.append(nn.Sequential(make_ar_hidden_layer(input_size), nn.Linear(hidden_size, Dy * model_params.num_motion_vars * 2)))

                else:
                    make_ar_hidden_layer = lambda input_size : nn.Sequential(nn.Linear(input_size, model_params.auto_regressive_hidden_size), nn.ReLU()) if model_params.auto_regressive_hidden_size > 0 else nn.Identity()
                    hidden_size = model_params.auto_regressive_hidden_size if model_params.auto_regressive_hidden_size > 0 else input_size

                    self.fcs.append(nn.Sequential(make_ar_hidden_layer(input_size), nn.Linear(hidden_size, Dy * (2 if model_params.model_variance else 1) * 2)))
                    self.time_heads.append(nn.Sequential(make_ar_hidden_layer(input_size), nn.Linear(hidden_size, model_params.num_control_pts)))
                    self.motion_heads.append(nn.Sequential(make_ar_hidden_layer(input_size), nn.Linear(hidden_size, Dy * model_params.num_motion_vars * 2)))

        else:
            raise NotImplementedError()
            self.fc = nn.Linear(conv1d_hidden_dim, model_params.num_obs * (Dy * 4 + model_params.num_control_pts))

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        output = input
        batch_size = input.size(0)
        for conv1d_layer in self.conv1d_layers:
            output = F.leaky_relu(conv1d_layer(output))
        output = output.view(batch_size, -1)

        if self.auto_regressive:
            loc_mu, loc_log_var, loc, size_mu, size_log_var, size = [], [], [], [], [], []
            time_idxs = []
            vel_mu, vel_log_var, vel = [], [], []
            auto_regressive_input = []

            fc_out = time_out = motion_out = None

            for i in range(len(self.fcs)):
                fc = self.fcs[i]
                time_head = self.time_heads[i]
                motion_head = self.motion_heads[i]
                if not auto_regressive_input:
                    fc_out = fc(output)
                    if self.training_stage >= 2:
                        time_out = time_head(output)
                    if self.training_stage == 3:
                        motion_out = motion_head(output)
                else:
                    fc_out = fc(torch.cat([output] + auto_regressive_input, dim=-1))
                    if self.training_stage >= 2:
                        time_out = time_head(torch.cat([output] + auto_regressive_input, dim=-1))
                    if self.training_stage == 3:
                        motion_out = motion_head(torch.cat([output] + auto_regressive_input, dim=-1))

                loc_mu_i, loc_log_var_i, loc_i, size_mu_i, size_log_var_i, size_i = self.get_dist_and_sample(fc_out)
                time_i, time_i_scaled = self.get_time_indexes(time_out, batch_size)
                vel_mu_i, vel_log_var_i, vel_i = self.get_motion_dist_and_sample(motion_out, batch_size)

                loc_mu.append(loc_mu_i)
                loc_log_var.append(loc_log_var_i)
                loc.append(loc_i)
                size_mu.append(size_mu_i)
                size_log_var.append(size_log_var_i)
                size.append(size_i)

                time_idxs.append(time_i)

                vel_mu.append(vel_mu_i)
                vel_log_var.append(vel_log_var_i)
                vel.append(vel_i)

                auto_regressive_input.extend([loc_i.view(batch_size, -1)])
                if self.model_params.model_variance:
                    auto_regressive_input.extend([size_i.view(batch_size, -1)])
                auto_regressive_input.extend([time_i_scaled.view(batch_size, -1)])
                auto_regressive_input.extend([vel_i.view(batch_size, -1)])

            loc_mu = torch.cat(loc_mu, dim=1)
            loc_log_var = torch.cat(loc_log_var, dim=1)
            loc = torch.cat(loc, dim=1)
            size_mu = torch.cat(size_mu, dim=1)
            size_log_var = torch.cat(size_log_var, dim=1)
            size = torch.cat(size, dim=1)
            time = torch.cat(time_idxs, dim=1)

            vel_mu = torch.cat(vel_mu, dim=1)
            vel_log_var = torch.cat(vel_log_var, dim=1)
            vel = torch.cat(vel, dim=1)

            return (loc_mu, loc_log_var, loc), (size_mu, size_log_var, size), (vel_mu, vel_log_var, vel), time
        else:
            raise NotImplementedError()
            output = self.fc(output)
            return self.get_dist_and_sample(output)

    def get_dist_and_sample(self, output_from_fc):
        """
        :param output_from_fc: tensor, shape: (batch_size, N_obs * 4 * Dy)
        :return: loc_mu, loc_log_var, loc, size_mu, size_log_var, size: tensor, shape: (batch_size, N_obs, Dy)
        """
        batch_size = output_from_fc.size(0)
        vars = 4 if self.model_params.model_variance else 2
        output = output_from_fc.view(batch_size, -1, vars, self.params.Dy)

        loc_mu = output[:, :, 0]
        loc_log_var = output[:, :, 1]
        loc_log_var = torch.clamp(loc_log_var, min=np.log(EPS))
        loc = self.reparameterize(loc_mu, loc_log_var)

        if self.model_params.model_variance:
            size_mu = output[:, :, 2]
            size_log_var = output[:, :, 3]
            size_log_var = torch.clamp(size_log_var, min=np.log(EPS))
            size = torch.log(1 + torch.exp(self.reparameterize(size_mu, size_log_var)))
        else:
            size_mu = torch.zeros_like(loc_mu)
            size_log_var = torch.zeros_like(loc_log_var)
            size = torch.ones_like(loc)*0.5 # size change from 0.3 to 0.5

        return loc_mu, loc_log_var, loc, size_mu, size_log_var, size

    def get_time_indexes(self, output_from_time_head, batch_size=32):
        """
        :param output_from_time_head: tensor, shape: (batch_size, N_obs * num_control_pts)
        :return: time_indexes : tensor, shape: (batch_size, N_obs, num_control_pts)
        """  
        if output_from_time_head is None:
            ones = torch.ones(batch_size, 1, self.model_params.num_control_pts).to(self.params.device)
            return ones, ones 
        
        time_out = output_from_time_head.view(batch_size, -1, self.model_params.num_control_pts)

        if hasattr(self.model_params, "time_heads_MoS_num_components") and self.model_params.time_heads_MoS_num_components > 1:
            if self.training_stage == 3:
                time_index = self.gumbel_softmax((time_out + EPS).log())
            else:
                time_index = time_out
        else:
            time_index = self.gumbel_softmax(time_out)
        
        time_scaled = time_index / time_index.max(dim=-1, keepdim=True)[0]

        return time_index, time_scaled


        # if self.current_gumbel_tau > self.model_params.gumbel_final_tau and self.training_stage == 2:
        #     print("get_time_indexes - self.current_gumbel_tau:", self.current_gumbel_tau)
        #     time_index = F.gumbel_softmax(time_out, tau=self.current_gumbel_tau, hard=False)
        #     # time_index = time_index / time_index.max(dim=-1, keepdim=True)[0]
        # else:
        #     print("get_time_indexes - one-hot encoding time")
        #     time_index = F.gumbel_softmax(time_out, tau=self.current_gumbel_tau, hard=True) 
        
    
    def get_motion_dist_and_sample(self, output_from_vel_head, batch_size=32):
        """
        :param output_from_vel_head: tensor, shape: (batch_size, N_obs * 4 * Dy)
        :return: vel_mu, vel_log_var, vel: (batch_size, N_obs, Dy)
        """
        if output_from_vel_head is None:
            zero_out = torch.zeros(batch_size, 1, 2).to(self.params.device)
            return zero_out, zero_out, zero_out

        if self.model_params.num_motion_vars > 1:
            raise NotImplementedError("get_motion_dist_and_sample not implemented for accelaration")

        output = output_from_vel_head.view(batch_size, -1, 2, self.params.Dy)

        vel_mu = output[:, :, 0]
        vel_log_var = output[:, :, 1]
        vel_log_var = torch.clamp(vel_log_var, min=np.log(EPS))
        vel = self.reparameterize(vel_mu, vel_log_var)

        return vel_mu, vel_log_var, vel

    def set_trainingstage(self, ts):
        self.training_stage = ts
        self.gumbel_softmax.set_trainingstage(self.training_stage)
        if self.training_stage == 2:

            if "feature_encoder" in self.model_params.freeze_layers_at_TS2:
                for param in self.conv1d_layers.parameters():
                    param.requires_grad = False
            
            if "location_heads" in self.model_params.freeze_layers_at_TS2:
                for param in self.fcs.parameters():
                    param.requires_grad = False
                    
            # freeze everything? or not?
            # for now dont freeze anything.
            # pass
        elif self.training_stage == 3:
            # again freeze?
            pass


class Hallucination(nn.Module):
    def __init__(self, params, writer=None):
        super(Hallucination, self).__init__()
        self.params = params
        self.model_params = self.params.model_params
        params.model_params.num_control_pts = params.model_params.knot_end - params.model_params.knot_start - 4
        if params.optimization_params.decoder == "CVX":
            Decoder = CVX_Decoder
        else:
            raise NotImplementedError

        self.encoder = Encoder(params).to(params.device)
        self.decoder = Decoder(params).to(params.device)
        self.coef = self._init_bspline_coef()

        # for optimization debugging
        self.writer = writer
        self.num_decoder_error = 0

        self.set_trainingstage(1)

    def _init_bspline_coef(self):
        model_params = self.model_params

        # Find B-spline coef to convert control points to pos, vel, acc, jerk
        knots = np.arange(model_params.knot_start, model_params.knot_end) * model_params.knot_dt
        pos_bspline = BSpline(knots, np.eye(model_params.num_control_pts), 3, extrapolate=False)
        vel_bspline = pos_bspline.derivative(nu=1)
        # acc_bspline = pos_bspline.derivative(nu=2)
        label_t_min, label_t_max = knots[3], knots[-4]
        t = np.linspace(label_t_min, label_t_max, model_params.traj_len)
        # coef = np.array([pos_bspline(t), vel_bspline(t), acc_bspline(t)])             # (3, T, num_control_pts)
        coef = np.array([pos_bspline(t), vel_bspline(t)])                               # (2, T, num_control_pts)
        assert not np.isnan(coef).any()
        coef = torch.from_numpy(coef.astype(np.float32)[None, ...]).to(self.params.device)   # (1, 2, T, num_control_pts)
        return coef

    def set_trainingstage(self, ts):
        if ts not in [1,2,3]:
            raise ValueError("training stage can only be one of {1,2,3}")
        self.training_stage = ts
        self.encoder.set_trainingstage(ts)

    def forward(self, full_traj, reference_pts, decode=False):
        """
        :param full_traj: (batch_size, T_full, 2 * Dy) tensor, recorded trajectory, pos + vel
        :param reference_pts: (batch_size, num_control_pts, Dy) tensor, reference control_pts during the optimization
        :param decode: True when training
        :return: recon_traj: (batch_size, T, Dy) tensor
        """
        loc_tup, size_tup, vel_tup, obs_time_idx = self.encode(full_traj)
        # loc_mu, loc_log_var, loc, size_mu, size_log_var, size, obs_time_idx = self.encode(full_traj)
        if not decode:
            return loc_tup, size_tup, vel_tup, obs_time_idx
            #loc_mu, loc_log_var, loc, size_mu, size_log_var, size, obs_time_idx
        
        if self.training_stage == 3:
            loc_rolledout, size_rolledout = self.rollout_obstacles(loc_tup[2], size_tup[2], vel_tup[2], obs_time_idx)
            ones_timeidx = torch.ones_like(obs_time_idx).to(self.params.device)
            recon_traj, recon_control_points = self.decode(reference_pts, loc_rolledout, size_rolledout, ones_timeidx)
        elif self.training_stage == 2:
            scaled_time_idx = obs_time_idx / obs_time_idx.max()
            recon_traj, recon_control_points = self.decode(reference_pts, loc_tup[2], size_tup[2], scaled_time_idx)
        else:
            recon_traj, recon_control_points = self.decode(reference_pts, loc_tup[2], size_tup[2], obs_time_idx)
        
        return recon_traj, recon_control_points, loc_tup, size_tup, vel_tup, obs_time_idx

    def encode(self, full_traj):
        loc_tup, size_tup, vel_tup, time_idx = self.encoder(full_traj)
        # loc_mu, loc_log_var, loc, size_mu, size_log_var, size, time = self.encoder(full_traj)
        return loc_tup, size_tup, vel_tup, time_idx

    def decode(self, reference_pts, loc, size, obs_time_idx):
        model_params = self.model_params

        # initial traj before optimization, straight line from start to goal, (batch_size, num_control_pts, Dy)
        init_control_pts = reference_pts[:, None, 0] + \
                           torch.linspace(0, 1, model_params.num_control_pts)[None, :, None].to(self.params.device) * \
                           (reference_pts[:, None, -1] - reference_pts[:, None, 0])
        
        try:
            recon_control_points = self.decoder(init_control_pts, loc, size, reference_pts, obs_time_idx)
        except diffcp.cone_program.SolverError as e:
            logger.error(str(e))
            recon_control_points = init_control_pts[None, ...]
            if self.writer:
                plot_opt(self.writer, reference_pts, recon_control_points, loc, size,
                         self.num_decoder_error, is_bug=True)
                self.num_decoder_error += 1
        # (batch_size, 1, num_control_pts, 3)
        last_recon_control_points = recon_control_points[-1, :, None]
        recon_traj = torch.matmul(self.coef, last_recon_control_points)
        return recon_traj, recon_control_points

    def loss(self, full_traj, traj, recon_traj, reference_pts, loc_mu, loc_log_var, loc, size_mu, size_log_var, size, time_idx):
        """
        :param full_traj: (batch_size, Dy, T_) tensor
        :param traj: (batch_size, T, Dy) tensor
        :param recon_traj: (batch_size, T, Dy) tensor
        :param loc_mu: (batch_size, num_obs, Dy) tensor
        :param loc_log_var: (batch_size, num_obs, Dy) tensor
        :param loc: (batch_size, num_obs, Dy) tensor
        :param size_mu: (batch_size, num_obs, Dy) tensor
        :param size_log_var: (batch_size, num_obs, Dy) tensor
        :param size: (batch_size, num_obs, Dy) tensor
        :param time_idx: (batch_size, num_obs, num_ctrl_pts) tensor
        :return:
        """
        batch_size, _, Dy = reference_pts.size()
        device = self.params.device

        scaled_time_idx =  time_idx / time_idx.max(dim=-1, keepdim=True)[0]

        # reconstruction error
        recon_loss = torch.mean(torch.sum((traj - recon_traj) ** 2, dim=(1, 2)))

        # regularization loss
        # repulsion between obs
        loc_diff = loc[:, :, None] - loc[:, None]                                   # (batch_size, num_obs, num_obs, Dy)
        loc_diff_norm = torch.norm(loc_diff, dim=-1)
        # mask distance between the same obs to avoid numerical issues
        loc_diff[loc_diff_norm == 0] = size.detach().view(-1, Dy) * 3

        loc_diff_norm = torch.norm(loc_diff, dim=-1, keepdim=True)
        loc_diff_direction = loc_diff / loc_diff_norm                               # (batch_size, num_obs, num_obs, Dy)
        size_ = size[:, None, :]                                                    # (batch_size, 1, num_obs, Dy)
        tmp = torch.sqrt(torch.sum(loc_diff_direction ** 2 / size_ ** 2, dim=-1))   # (batch_size, num_obs, num_obs)
        radius_along_direction = 1 / tmp                                            # (batch_size, num_obs, num_obs)

        combined_radius_along_direction = radius_along_direction + torch.transpose(radius_along_direction, 1, 2)
        obs_overlap = combined_radius_along_direction - loc_diff_norm[..., 0]
        
        repulsive_loss = torch.clamp(obs_overlap, min=0) ** 2

        obs_to_obs_time_idx = torch.bmm(time_idx, torch.transpose(time_idx, 1, 2))          # not using scaled_time_idx as we want to keep obs_to_obs in [0,1]
                                                                                            # this is the expected avg of two obstacles to occur together
        repulsive_loss = repulsive_loss * obs_to_obs_time_idx


        repulsive_loss = torch.sum(repulsive_loss) / batch_size
        repulsive_loss *= self.params.model_params.lambda_mutual_repulsion_adjusted

        # repulsion between obs and reference_pts
        diff = reference_pts.view(batch_size, 1, -1, Dy) - loc.view(batch_size, -1, 1, Dy)  # B N C Dy 
        diff_norm = torch.linalg.norm(diff, dim=-1)                                 # (B, num_obs, num_control_pts)
        direction = diff / torch.clamp(diff_norm, min=EPS)[..., None]               # B N C Dy 

        # intersection = t. denote direction = (x, y, z) and obs_size = (a, b, c)
        # then (t * x)^2 / a^2 + (t * y)^2 / b^2 + (t * z)^2 / c^2 = 1
        # shape: (B, num_obs, num_control_pts)
        size = size.view(batch_size, -1, 1, Dy)
        intersection_inv = torch.sqrt(torch.sum(direction ** 2 / size ** 2, dim=-1))
        intersection_inv = torch.clamp(intersection_inv, min=EPS)
        intersection = 1 / intersection_inv

        clearance = self.params.optimization_params.clearance
        dist = diff_norm - intersection                              
        dist_error = clearance - dist                               # B N C
        dist_error = dist_error * scaled_time_idx                          # masking away the time_idx ones
        reference_repulsion_loss = torch.sum(torch.clamp(dist_error, min=0) ** 2) / batch_size
        reference_repulsion_loss *= self.params.model_params.lambda_reference_repulsion_adjusted

        # KL divergence from prior
        loc_var = torch.exp(loc_log_var)                                            # (batch_size, num_obs, Dy)

        loc_prior_mu = torch.mean(full_traj[:, :Dy], dim=-1)                        # (batch_size, Dy)
        loc_prior_var = torch.var(full_traj[:, :Dy], dim=-1)                        # (batch_size, Dy)
        loc_prior_var = torch.clamp(loc_prior_var, min=self.params.model_params.min_obs_loc_prior_var)
        loc_prior_var *= self.params.model_params.obs_loc_prior_var_coef
        loc_prior_mu = loc_prior_mu[:, None]                                        # (batch_size, 1, Dy)
        loc_prior_var = loc_prior_var[:, None]                                      # (batch_size, 1, Dy)

        # kl divergence between two diagonal Gaussian
        # loc_kl_loss = 0.5 * (torch.sum(loc_var / loc_prior_var
        #                                + (loc_mu - loc_prior_mu) ** 2 / loc_prior_var
        #                                + torch.log(loc_prior_var) - torch.log(loc_var),
        #                                dim=-1)
        #                      - self.params.Dy)
        # loc_kl_loss = torch.mean(torch.sum(loc_kl_loss, dim=1))

        # Zizhao's location reg loss: -log normal
        # loc_logp = -0.5 * torch.sum((loc - loc_prior_mu) ** 2 / loc_prior_var, dim=-1) \
        #            -0.5 * (self.params.Dy * np.log(2 * np.pi) + torch.sum(torch.log(loc_prior_var), dim=-1))
        # loc_logp = torch.mean(torch.sum(loc_logp, dim=-1))
        # loc_reg_loss = -loc_logp * self.params.model_params.lambda_loc_reg
        # critical reg loss: weighted MSE between ref_pts and obs_pts at time_idx
        loc_reg_loss =  (dist - clearance)*scaled_time_idx
        loc_reg_loss = torch.sum(torch.clamp(loc_reg_loss, min=0) ** 2) / batch_size
        loc_reg_loss = loc_reg_loss * self.params.model_params.lambda_loc_reg_adjusted

        size_var = torch.exp(size_log_var)                                          # (batch_size, num_obs, Dy)

        size_prior_mu = self.params.model_params.obs_size_prior_mu
        size_prior_var = self.params.model_params.obs_size_prior_var
        size_prior_std = np.sqrt(size_prior_var)
        size_prior_mu_ = np.log(np.exp(size_prior_mu) - 1.0).astype(np.float32)
        size_prior_std_ = np.log(np.exp(size_prior_mu + size_prior_std) - 1.0).astype(np.float32)
        size_prior_var_ = size_prior_std_ ** 2
        size_prior_mu = size_prior_mu_ * torch.ones(self.params.Dy).to(device)
        size_prior_var = size_prior_var_ * torch.ones(self.params.Dy).to(device)
        size_prior_mu = size_prior_mu[None, None, :]                                # (1, 1, Dy)
        size_prior_var = size_prior_var[None, None, :]

        # kl divergence between two diagonal Gaussian
        size_kl_loss = 0.5 * (torch.sum(size_var / size_prior_var
                                        + (size_mu - size_prior_mu) ** 2 / size_prior_var
                                        + torch.log(size_prior_var) - torch.log(size_var),
                                        dim=-1)
                              - self.params.Dy)
        size_kl_loss = torch.mean(torch.sum(size_kl_loss, dim=1))
        size_kl_loss *= self.params.model_params.lambda_size_kl_adjusted

        # minimizing entropy of time idx to push it to be as sharp as possible
        time_idx_entropy = torch.mean(-(time_idx * torch.log(time_idx + EPS)).sum(dim=-1))
        time_idx_entropy *= self.params.model_params.lambda_time_idx_entropy_adjusted

        loss = recon_loss + loc_reg_loss + size_kl_loss + repulsive_loss + reference_repulsion_loss + time_idx_entropy
        loss_detail = {"loss": loss,
                       "recon_loss": recon_loss,
                       "loc_reg_loss": loc_reg_loss,
                       "size_kl_loss": size_kl_loss,
                       "repulsive_loss": repulsive_loss,
                       "reference_repulsion_loss": reference_repulsion_loss,
                       "time_idx_entropy": time_idx_entropy}
        loss_detail = dict([(k, v.item()) for k, v in loss_detail.items()])

        return loss, loss_detail

    def rollout_obstacles(self, loc, size, vel, time_index):
        batch_size, num_obs, num_ctrl_pts = time_index.size()
        ref_pt_ts = np.arange(self.model_params.knot_start + 2, self.model_params.knot_end - 2) * self.model_params.knot_dt
        ref_pt_ts_tensor = torch.from_numpy(ref_pt_ts).to(self.params.device)

        ref_pt_ts_tensor = ref_pt_ts_tensor.repeat(batch_size, num_obs,1)
        t0 = ref_pt_ts_tensor[time_index.bool()] # time_index has to be one-hot encoded
        timesteps = ref_pt_ts_tensor - t0.view(batch_size, num_obs,1)

        loc_rolledout = loc[:, :, None] + vel[:, :, None] * timesteps[..., None]
        loc_rolledout = loc_rolledout.float()
        size_rolledout = size.unsqueeze(2).repeat_interleave(len(ref_pt_ts), dim=2).float()

        return loc_rolledout, size_rolledout
    
    def test(self, reference_pts, recon_control_points, loc, size):
        model_params = self.model_params
        opt_func = self.decoder.opt_func
        # initial traj before optimization, straight line from start to goal, (batch_size, num_control_pts, Dy)
        init_control_pts = reference_pts[:, None, 0] + \
                           torch.linspace(0, 1, model_params.num_control_pts)[None, :, None].to(self.params.device) * \
                           (reference_pts[:, None, -1] - reference_pts[:, None, 0])
        opt_func.update(loc, size, reference_pts, init_control_pts)
        ode_num_timestamps, batch_size, num_control_pts, Dy = recon_control_points.size()

        losses = np.zeros((batch_size, ode_num_timestamps))
        for i in range(ode_num_timestamps):
            for j in range(batch_size):
                losses[j, i] = opt_func.loss(recon_control_points[i, j].view(1, num_control_pts * Dy)).item()

        loc = loc.cpu().detach().numpy()
        size = size.cpu().detach().numpy()
        recon_control_points = recon_control_points.cpu().detach().numpy()
        reference_pts = reference_pts.cpu().detach().numpy()

        return reference_pts, loc, size, recon_control_points, losses

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.decoder.train(training)
