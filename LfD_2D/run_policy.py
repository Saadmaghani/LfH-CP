#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist

import dynamic_reconfigure.client

import os
import scipy
import numpy as np
import time
import torch
from collections import deque

from LfD_main import TrainingParams, LfD_2D_dynamic_model

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

_DEBUG = False

model_name = "1fixedVel_50U_20L_10C"

update_dt = 0.05 # 0.05 - trained model. prev 0.04
local_goal_dist = 2.25 #2.25 #1.5 #rendered local goal: 1 m
local_path_dir_dist = 0.5
laser_max_range = 5.0 # rendered laser max range: 5.0

a_min = -135
a_max = 135
lidar_dim = 720
dt = 0.01
step = 20
lidar_tolerance = 0.00

a_min = a_min * np.pi / 180
a_max = a_max * np.pi / 180
da = (a_max - a_min) / (lidar_dim - 1)

# for rolling out laser scans
past_scans = 1
lidar_freq = 50 # 40 in jackal. 50 in simulation

model_seq_dt = 0.025 # 0.025s (40Hz) between each lidar scans in the model
check_model_frequency = True

class Predictor:
    def __init__(self, policy, params):
        self.policy = policy
        self.policy.eval()
        self.params = params

        self.global_path = []
        self.local_goal = None
        self.local_goal_hist = []
        self.local_path_dir = None
        self.raw_scan = None
        self.clipped_scan = None
        self.scans_buffersize = params.predictor_params.sequence_length

        self.last_lidar_ts = None
        self.stacked_scans = None 
        # self.dt = params.predictor_params.sequence_dt
        self.raw_scans = None

        self.turn_flag = 0
        self.v = 0
        self.w = 0

        # define the boundary of the vehicle
        self.width = 0.165 * 2
        self.length = 0.21 * 2
        self.y_top = 0.165
        self.y_bottom = -0.165
        self.x_right = 0.21
        self.x_left = -0.21
        n = 10
        self.boundary = np.zeros((4 * n, 2))
        xx = np.linspace(self.x_left, self.x_right, n)
        yy = np.linspace(self.y_bottom, self.y_top, n)
        # order is U, B, L, R
        self.boundary[:n][:, 0] = xx
        self.boundary[n:2 * n][:, 0] = xx
        self.boundary[2 * n:3 * n][:, 0] = self.x_left
        self.boundary[3 * n:][:, 0] = self.x_right
        self.boundary[:n][:, 1] = self.y_top
        self.boundary[n:2 * n][:, 1] = self.y_bottom
        self.boundary[2 * n:3 * n][:, 1] = yy
        self.boundary[3 * n:][:, 1] = yy

        #simple moving avg for noisy scans
        self.window = 5
        self.thresh = 0.2
        self.window_scans = []

        # autoregressive cmds
        if self.params.predictor_params.autoregressive:
            self.stacked_cmds = np.zeros((self.params.predictor_params.sequence_length, 2)).astype(np.float32)

        self.save_laser_scans = False
        if self.save_laser_scans:
            self.all_scans = None

        # if on, give warning if the model is taking longer than model_seq_dt
        self.check_model_frequency = params.check_model_frequency
        if self.check_model_frequency:
            self.inference_times = deque(maxlen=100)


    def update_status(self, msg):
        # print("DYNA-LFLH CP5")
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.pose = msg.pose

        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))

    @staticmethod
    def transform_lg(gp, X, Y, PSI):
        R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)

        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = np.matmul(R_i2r, pi.T)
        return np.asarray(pr[:2, :]).T

    # removing savgol filters as it delays the start of the processing of the local goal.
    # this leads to the Dyna-lflh models start much later than dwa leading to incompatible results 
    def update_global_path(self, msg):
        if hasattr(self, "X"):            
            gp = []
            for pose in msg.poses:
                gp.append([pose.pose.position.x, pose.pose.position.y])
            gp = np.array(gp)
            x = gp[:, 0]
            # try:
            #     xhat = scipy.signal.savgol_filter(x, 19, 3)
            # except:
            xhat = x
            y = gp[:, 1]
            # try:
            #     yhat = scipy.signal.savgol_filter(y, 19, 3)
            # except:
            yhat = y
            gphat = np.column_stack((xhat, yhat))
            gphat.tolist() 

            self.global_path = self.transform_lg(gphat, self.X, self.Y, self.PSI)
            self.local_goal = self.get_local_goal(self.global_path)
            self.local_path_dir = self.get_local_path_dir(self.global_path)
 
    def get_local_goal(self, gp):
        # same as zizhao's
        local_goal = np.zeros(2)
        odom = np.zeros(2)
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - odom) > 0.05:
                odom = gp[0]
            for wp in gp:
                dist = np.linalg.norm(wp - odom)
                if dist > self.params.local_goal_dist:
                    break
            local_goal = wp - odom
            local_goal /= np.linalg.norm(local_goal)

        return local_goal.astype(np.float32)

    def get_local_path_dir(self, gp):
        lp_dir = 0
        prev_wp = np.zeros(2)
        start = np.zeros(2)
        cum_dist = 0
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - prev_wp) > 0.05:
                prev_wp = gp[0]
                start = gp[0]
            for wp in gp:
                wp_start_dist = np.linalg.norm(wp - start)
                if wp_start_dist > 0.1:
                    lp_dir += (wp - start) / wp_start_dist
                cum_dist += np.linalg.norm(wp - prev_wp)
                prev_wp = wp
                if cum_dist > local_path_dir_dist:
                    break
            if isinstance(lp_dir, np.ndarray):
                lp_dir = np.arctan2(lp_dir[1], lp_dir[0])

        return lp_dir

    def bch_safety_check(self, v, w, size, std):

        V = v + np.random.randn(size) * std * v
        W = w + np.random.randn(size) * std * w
        X = 0
        Y = 0
        PSI = 0
        if past_scans != 1:
            raise Exception("MPC only works with past_scans =1")
        # dynamic obstacles:
        if self.raw_scans is None or len(self.raw_scans) < 2:
            return 1

        
        scan_ds = self.raw_scans[-1] - self.raw_scans[-2]
        cur_scan = self.raw_scans[-1].copy()
        scan_dt = 1/lidar_freq

        for i in range(step):
            X_DOT = V * np.cos(PSI)
            Y_DOT = V * np.sin(PSI)
            PSI_DOT = W
            X = X + X_DOT * dt
            Y = Y + Y_DOT * dt
            PSI = PSI + PSI_DOT * dt
            if cur_scan is not None:
                cur_scan = cur_scan + scan_ds*scan_dt

        # V [B, 1]
        # W [B, 1]
        # PSI [B, 1]
        # X [B, 1]
        # Y [B, 1]
        R_r2i = np.concatenate([np.cos(PSI), -np.sin(PSI), np.sin(PSI), np.cos(PSI)]).reshape(-1, 2, 2)  # [B, 2, 2]

        # R * x = (x^T R^T) => [1,3] x [3, 3]^T => [1, 3]
        N = self.boundary.shape[0]
        rotation = np.matmul(R_r2i.reshape(-1, 1, 2, 2), self.boundary.reshape(1, -1, 2, 1)).reshape(-1, N, 2)  # [B, N, 2]
        translation = np.concatenate([X, Y], -1).reshape(-1, 1, 2)  # [B, 1, 2]
        boundary = rotation + translation  # [B, N, 2]

        XP, YP = boundary[:, :, 0], boundary[:, :, 1]
        beam_idx = ((np.arctan2(YP, XP) - a_min) // da).astype(np.int32)
        valid = ((beam_idx >= 0) & (beam_idx < lidar_dim)).astype(np.int32)
        beam_idx = beam_idx * valid
        RHO = np.sqrt(np.square(boundary)).sum(2)  # [B, N] the distance

        # calculating the predicted laser scans

        RHO_beam = cur_scan[beam_idx]  # [B, N]

        crash = np.sign(((RHO_beam < RHO + lidar_tolerance) * valid).sum(-1))  # [B]
        safety_percentage = 1 - crash.sum() / float(crash.shape[0])
        return safety_percentage

    def update_laser(self, msg):
        self.raw_scan = np.array(msg.ranges)
        if self.raw_scans is None:
            self.raw_scans = self.raw_scan.copy()[None]
        self.raw_scans = np.concatenate((self.raw_scans, self.raw_scan[None]), 0)
        if len(self.raw_scans) > past_scans +1:
            self.raw_scans = np.delete(self.raw_scans, 0, axis=0)
        self.clipped_scan = np.minimum(self.raw_scan, self.params.laser_max_range).astype(np.float32)
        now = rospy.Time.now()
        if self.last_lidar_ts is not None and (now-self.last_lidar_ts).to_sec() < self.params.predictor_params.sequence_dt:
            return
        self.last_lidar_ts = now
        skip_scan = False
        if len(self.window_scans) < self.window:
            self.window_scans.append(self.clipped_scan)
        else:
            if (self.clipped_scan == self.params.laser_max_range).all(): # it may be a noisy reading
                # compare it to the moving avg and see if the max diff is > thresh
                simple_avg_scan = np.array(self.window_scans).mean(axis=0)
                max_diff = np.abs(simple_avg_scan - self.clipped_scan).max()
                if max_diff > self.thresh:
                    skip_scan = True
                    self.clipped_scan = self.stacked_scans[-1]
            if not skip_scan:
                self.window_scans.pop(0)
                self.window_scans.append(self.clipped_scan)

        if not skip_scan:
            if self.stacked_scans is None:
                self.stacked_scans = np.repeat(self.clipped_scan[None], self.scans_buffersize, axis=0)
            else:
                changed_scan = np.delete(self.stacked_scans, 0, axis=0)
                self.stacked_scans = np.concatenate((changed_scan, self.clipped_scan[None]), 0)

        if self.save_laser_scans:
            if self.all_scans is None:
                self.all_scans = self.stacked_scans[None]
            else:
                self.all_scans = np.concatenate((self.all_scans, self.stacked_scans[None]), axis=0)
            print(self.all_scans.shape)
            with open("scan_file.npy", "wb") as f:
                np.save(f, self.all_scans)

    def update_cmd_vel(self,):
        if self.check_model_frequency:
            start_time = rospy.get_time()
        if self.stacked_scans is None or self.local_goal is None:
            return
        
        # print("DYNA-LFLH: CP3")
        # if needs hard turn
        # try:
        #     turning_threshold = np.pi / 3  # this will depend on max_v
        #     stop_turning_threshold = np.pi / 18
        #     direction_angle = self.local_path_dir
        #     # print(direction_angle)
        #     if self.turn_flag == 0:
        #         if direction_angle > turning_threshold:
        #             print("Hard Turn Left")
        #             self.turn_flag = 1
        #         elif direction_angle < -turning_threshold:
        #             print("Hard Turn Right")
        #             self.turn_flag = -1
        #         else:
        #             # print("Normal Operation")
        #             self.turn_flag = 0
        #     else:
        #         if abs(direction_angle) < stop_turning_threshold:
        #             print("Resume Normal Operation")
        #             self.turn_flag = 0
        # except:
        #     self.turn_flag = 0

        # if self.turn_flag == 1:
        #     self.v = 0
        #     self.w = 2 * direction_angle
        # elif self.turn_flag == -1:
        #     self.v = 0
        #     self.w = 2 * direction_angle
        # else:
        scans = torch.from_numpy(self.stacked_scans[None]).to(self.params.device)
        local_goal = torch.from_numpy(self.local_goal[None]).to(self.params.device)
        ar_cmds = torch.from_numpy(self.stacked_cmds[None]).to(self.params.device)

        cmd = self.policy(scans, local_goal, arcmd=ar_cmds)
        cmd = cmd[0].detach().cpu().numpy()                    # remove batch size

        if self.params.predictor_params.autoregressive:
            cmd = cmd[-1]
        if hasattr(self.policy, "num_pred_cmd") and self.policy.num_pred_cmd > 1:
            cmd = cmd[0]

        if self.params.predictor_params.autoregressive:         # update stacked_cmds
            changed_cmd = np.delete(self.stacked_cmds, 0, axis=0)
            self.stacked_cmds = np.concatenate((changed_cmd, cmd[None]), 0)

        self.v, self.w = cmd


        if self.check_model_frequency:
            self.inference_times.append(rospy.get_time() - start_time)
            avg_infer_time = np.array(self.inference_times).mean()
            if avg_infer_time > self.params.predictor_params.sequence_dt:
                print(f"WARNING - LfD model inference time ({avg_infer_time:.4f}) > sequence_dt ({self.params.predictor_params.sequence_dt})")

        # recovery behavior:
        #   stopping
        #   backing up 
        #   other?
        # how it checks:
        #   rollout of current action    -> first
        #   based on obstacle dynamics

        # safety check
        ctr = 0  # how many recover count
        while self.bch_safety_check(self.v, self.w, 1, 0) == 0:
            # new recovery: just turn in place
            if ctr < 1:
                print("Recovery: Stop")
                self.v = 0
                self.w = 0
                # if self.w > 0 and self.w < 0.1:
                #     self.w = 0.1
                # elif self.w < 0 and self.w >-0.1:
                #     self.w = -0.1
                ctr += 1
            elif ctr >= 1:
                print("Recovery: Back up")
                self.v = -0.2
                self.w = 0
                break
        print("[INFO] goal: ({:.1f}, {:.1f}) current v: {:4.2f}, w: {:5.2f}".format(local_goal[0,0], local_goal[0,1], self.v, self.w))

def get_motion_planner_name_checkpoint(repo, index):
    if(index < 0):
        return None, None
    
    highestmodel = ''
    folder_path = os.path.join(repo, "LfD_2D","rslts")
    listofmodels = [d for d in os.listdir(folder_path) if "evaluated" not in d]
    listofmodels.sort()
    modelname = listofmodels[index]
    highestmodel = get_latest_cp(folder_path, modelname)
    return modelname, highestmodel

def get_latest_cp(folder_path, model_name):
    model_path = os.path.join(folder_path, model_name, 'trained_models')
    listofdonemodels = [x for x in os.listdir(model_path) if "model_" in x]
    listofdonemodels.sort()
    highest = -1
    for i in listofdonemodels:
        number_part = i.split('_')[1]
        model_number = int(number_part)
        if model_number > highest:
            highest = model_number
            highestmodel = i
    return highestmodel

if __name__ == '__main__':
    print("LOADED DYNA_LFLH - RUN_POLICY")

    # import debugpy

    # # Enable the debugpy server to listen on a specific port
    # debugpy.listen(("localhost", 5678))
    # print("Waiting for debugger attach...")

    # # Pause execution until the debugger attaches
    # debugpy.wait_for_client()
    # print("Debugger attached, resuming execution...")

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not _DEBUG:
        mp_index = int(rospy.get_param('/mp_index'))
        if mp_index is None or mp_index == -1:
            folder_path = os.path.join(repo_path, "LfD_2D","rslts")
            model_cp = get_latest_cp(folder_path, model_name)
        else:
            model_name, model_cp = get_motion_planner_name_checkpoint(repo_path, mp_index)
            # if model_name is None:
            #     if MODEL_NAME is None or MODEL_NAME=="":
            #         raise Exception("MODEL_NAME cant be None or empty if mp_index is -1")
            #     model_name = MODEL_NAME 
            #     model_cp = MODEL_CP
            
        print("DYNA_LFLH - RUN POLICY. RUNNING MODEL", model_name, "@", model_cp)
        
        folder_path = os.path.join("LfD_2D", "rslts", model_name) 
        params_path = os.path.join(repo_path, folder_path, "params.json")
        model_path = os.path.join(repo_path, folder_path, "trained_models", model_cp)

        params = TrainingParams(params_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params.device = device
        params.local_goal_dist = local_goal_dist
        params.laser_max_range = laser_max_range
        params.check_model_frequency = check_model_frequency

        model = LfD_2D_dynamic_model(params).to(device)
        assert os.path.exists(model_path)
        if "_5dist" in model_path:
            # this was a DDP model that saved the state_dict as module.state_dict
            from collections import OrderedDict
            cp = torch.load(model_path, map_location=device)
            new_cp = OrderedDict([(".".join(k.split(".")[1:]), v) for k,v in cp.items()])
            model.load_state_dict(new_cp)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("DYNA_LFLH - DEBUG MODE, LOADING LFD FROM PARAMS")
        params = TrainingParams()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params.device = device
        params.local_goal_dist = local_goal_dist
        params.laser_max_range = laser_max_range
        params.check_model_frequency = check_model_frequency
        model = LfD_2D_dynamic_model(params).to(device)

    if params.predictor_params.sequence_dt != model_seq_dt:
        print(f"params.predictor_params.sequence_dt ({params.predictor_params.sequence_dt}) != model_seq_dt ({model_seq_dt}). using former value")

    if params.predictor_params.sequence_length != params.transformer_params.history_length:
        print(f"params.predictor_params.sequence_length ({params.predictor_params.sequence_length}) != params.transformer_params.history_length ({params.transformer_params.history_length}). using latter (transformer) value")
        params.predictor_params.sequence_length = params.transformer_params.history_length


    predictor = Predictor(model, params)

    rospy.init_node('context_classifier', anonymous=True)
    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, predictor.update_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan",
                              Path, predictor.update_global_path, queue_size=1)
    sub_scan = rospy.Subscriber("/front/scan", LaserScan, predictor.update_laser, queue_size=1)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    goal_publisher = rospy.Publisher('/custom/robot_goal', Marker, queue_size=1)

    client = dynamic_reconfigure.client.Client('move_base/TrajectoryPlannerROS')
    client2 = dynamic_reconfigure.client.Client('move_base/local_costmap/inflater_layer')

    prev_cmd_time = None
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            if prev_cmd_time is None or (now - prev_cmd_time).to_sec() >= update_dt:
                predictor.update_cmd_vel()
                vel_msg = Twist()
                vel_msg.linear.x = predictor.v
                vel_msg.angular.z = predictor.w
                velocity_publisher.publish(vel_msg)
                
                if hasattr(predictor, "X") and predictor.local_goal is not None:
                    goal = Marker()
                    goal.header.frame_id = 'odom'
                    goal.header.stamp = rospy.Time.now()
                    goal.type = 0

                    goal.color.a = goal.color.g = 1
                    goal.color.r = goal.color.b = 0
                    goal.scale.x = 0.25
                    goal.scale.y = 0.5
                    goal.scale.z = 0

                    starting_point = Point()
                    starting_point.x = predictor.X
                    starting_point.y = predictor.Y
                    
                    end_point = Point()
                    end_point.x = -predictor.local_goal[0] +starting_point.x
                    end_point.y = -predictor.local_goal[1] +starting_point.y
                    
                    goal.points.append(starting_point)
                    goal.points.append(end_point)
                    goal_publisher.publish(goal)
                
                prev_cmd_time = now
        except rospy.exceptions.ROSInterruptException:
            break