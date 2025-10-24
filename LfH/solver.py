"""
Convex optimization-based trajectory decoder module.

This module provides a differentiable convex optimization layer for trajectory
reconstruction that enforces collision avoidance, velocity/acceleration constraints,
and smoothness objectives using CVXPyLayers.
"""

import torch
import torch.nn as nn

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from utils import TrainingParams

import traceback


EPS = 1e-6

class CVX_Decoder(nn.Module):
    """
    Convex optimization-based trajectory decoder.
    
    Reconstructs smooth, feasible, and collision-free trajectories through
    differentiable convex optimization. Uses ellipsoid collision constraints
    and enforces velocity and acceleration limits.
    """
    def __init__(self, params: TrainingParams) -> None:
        """
        Initialize the CVX decoder with optimization problem setup.
        
        Args:
            params: Configuration object containing:
                - device: Torch device (cpu/cuda)
                - Dy: State dimensionality (2 or 3)
                - model_params: Contains num_obs, num_control_pts, knot_dt
                - optimization_params: Contains max_vel, max_acc, lambda_smoothness,
                    lambda_collision, lambda_feasibility, clearance
        """
        super(CVX_Decoder, self).__init__()
        self.params = params
        self.device = params.device
        self._init_cvx_problem()

    def _init_cvx_problem(self):
        """
        Initialize the convex optimization problem using CVXPy.
        
        Sets up:
        - Decision variables: control points
        - Parameters: obstacle geometry, reference points
        - Objective: weighted sum of smoothness, collision, and feasibility losses
        - Constraints: Start/end point matching with reference trajectory
        
        The problem is Disciplined Parameterized Convex (DPP) to enable
        differentiation through the solver via CVXPyLayers.
        """
        Dy = self.params.Dy
        num_obs = self.params.model_params.num_obs
        num_control_pts = self.params.model_params.num_control_pts

        control_pts = cp.Variable((num_control_pts, Dy))
        direction = [cp.Parameter((num_control_pts, Dy)) for _ in range(num_obs)]
        intersection_projection = [cp.Parameter(num_control_pts) for _ in range(num_obs)]
        clearance = [cp.Parameter(num_control_pts) for _ in range(num_obs)]
        reference_pts = cp.Parameter((num_control_pts, Dy))
        constraints = [control_pts[0] == reference_pts[0], control_pts[-1] == reference_pts[-1]]

        vel = (control_pts[1:] - control_pts[:-1])  # (num_control_pts - 1, Dy)
        acc = (vel[1:] - vel[:-1])                  # (num_control_pts - 2, Dy)
        jerk = (acc[1:] - acc[:-1])                 # (num_control_pts - 3, Dy)
        smoothness_loss = cp.sum(acc ** 2) + cp.sum(jerk ** 2)

        collision_loss = 0
        for dir, intersec_proj, clr in zip(direction, intersection_projection, clearance):
            # dist = (control_pts - intersection_pts).dot(direction to reference on traj), shape: (num_control_pts,)
            dist = cp.sum(cp.multiply(control_pts, dir), axis=1) - intersec_proj
            dist_error = clr - dist
            dist_loss = cp.maximum(0, dist_error) ** 2
            collision_loss += cp.sum(dist_loss)

        knot_dt = self.params.model_params.knot_dt
        max_vel = self.params.optimization_params.max_vel
        max_acc = self.params.optimization_params.max_acc

        vel /= knot_dt
        acc /= knot_dt ** 2
    
        vel = cp.abs(vel)
        vel_feasibility_loss = cp.sum((cp.maximum(0, vel - max_vel)) ** 2)

        acc = cp.abs(acc)
        acc_feasibility_loss = cp.sum((cp.maximum(0, acc - max_acc)) ** 2)

        # extra "/ knot_dt ** 2": from ego_planner, to make vel and acc have similar magnitudes
        feasibility_loss = vel_feasibility_loss / knot_dt ** 2 + acc_feasibility_loss

        lambda_smooth = self.params.optimization_params.lambda_smoothness
        lambda_coll = self.params.optimization_params.lambda_collision
        lambda_feas = self.params.optimization_params.lambda_feasibility
        loss = lambda_smooth * smoothness_loss + lambda_coll * collision_loss + lambda_feas * feasibility_loss

        objective = cp.Minimize(loss)
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        parameters = direction + intersection_projection + clearance + [reference_pts]
        self.cvxpylayer = CvxpyLayer(problem, parameters=parameters, variables=[control_pts])

    def forward(
        self,
        init_control_pts: torch.Tensor,
        obs_loc: torch.Tensor,
        obs_size: torch.Tensor,
        reference_pts: torch.Tensor,
        obs_time_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct collision-free trajectory using convex optimization.
        
        Solves the optimization problem for each sample in the batch to find
        control points that balance smoothness, feasibility, and collision avoidance.
        
        Args:
            init_control_pts: Initial control points estimate.
                Shape: (batch_size, num_control_pts, Dy)
            obs_loc: Obstacle center locations.
                Shape: (batch_size, num_obs, Dy) or (batch_size, num_obs, num_control_pts, Dy)
            obs_size: Obstacle semi-axis sizes (for ellipsoid representation).
                Shape: (batch_size, num_obs, Dy) or (batch_size, num_obs, num_control_pts, Dy)
            reference_pts: Reference trajectory (nominal path without collisions).
                Shape: (batch_size, num_control_pts, Dy)
            obs_time_idx: Time validity indicator for obstacles (0 if absent, 1 if present).
                Shape: (batch_size, num_obs, num_control_pts) or (batch_size, num_obs)
                Used to disable obstacles at timesteps where they don't exist.
        
        Returns:
            torch.Tensor: Optimized control points.
                Shape: (batch_size, num_control_pts, Dy)
        
        Raises:
            AssertionError: If optimization problem is not DPP or solver fails.
        """
        if len(obs_loc.shape) != 4:
            obs_loc = obs_loc[:, :, None]
            obs_size = obs_size[:, :, None]

        # direction: normalized, from obs_loc to reference_pts
        batch_size, num_control_pts, Dy = list(reference_pts.size())
        diff = reference_pts.view(batch_size, 1, -1, Dy) - obs_loc
        diff_norm = torch.clamp(torch.linalg.norm(diff, dim=-1, keepdims=True), min=EPS)
        direction = diff / diff_norm

        # intersection = obs_loc + t * direction. denote direction = (x, y, z) and obs_size = (a, b, c)
        # then (t * x)^2 / a^2 + (t * y)^2 / b^2 + (t * z)^2 / c^2 = 1
        # shape: (B, num_obs, num_control_pts)
        # obs_size = obs_size.view(batch_size, -1, 1, Dy)
        t_inv = torch.sqrt(torch.sum(direction ** 2 / obs_size ** 2, dim=-1))
        t_inv = torch.clamp(t_inv, min=EPS)
        t = 1 / t_inv
        # intersection_projection = <intersection, direction>
        intersection_projection = torch.sum(obs_loc * direction, dim=-1) + t
        intersection_projection = intersection_projection * obs_time_idx


        dist = init_control_pts[:, None] - obs_loc                              # (B, num_obs, num_control_pts, Dy)
        dist_len = torch.norm(dist, dim=-1)                                     # (B, num_obs, num_control_pts)
        dist_len_clamped = torch.clamp(dist_len[..., None], min=EPS)            # (B, num_obs, num_control_pts, 1)
        dist_direction = dist / dist_len_clamped                                # (B, num_obs, num_control_pts, Dy)
        
        radius_along_dir_inv = torch.sqrt(torch.sum(dist_direction ** 2 / obs_size ** 2, dim=-1))
        radius_along_dir_inv = torch.clamp(radius_along_dir_inv, min=EPS)       # (B, num_obs, num_control_pts)
        radius_along_dir = 1 / radius_along_dir_inv

        in_collision = dist_len <= (radius_along_dir + self.params.optimization_params.clearance) * obs_time_idx # scaling it with whether it is present or not


        direction = direction * in_collision[..., None]
        intersection_projection = intersection_projection * in_collision
        clearance = self.params.optimization_params.clearance
        clearance = in_collision * clearance                                    # (B, num_obs, num_control_pts)


        direction = torch.unbind(direction, dim=1)
        intersection_projection = torch.unbind(intersection_projection, dim=1)
        clearance = torch.unbind(clearance, dim=1)
        parameters = direction + intersection_projection + clearance + (reference_pts,)
        try:
            recon_control_points, = self.cvxpylayer(*parameters, solver_args={"max_iters":10000})
        except AssertionError as e:
            print("----------------------------")
            print(f"LFH_MAIN:solver assertion error, traceback printed at stderr")
            traceback.print_exc()
            print("direction:")
            print(direction)
            print("intersection_projection:")
            print(intersection_projection)
            print("clearance:")
            print(clearance)
            print("----------------------------")
            return e
        return recon_control_points[None, ...]
