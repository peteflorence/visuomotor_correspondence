# system
import numpy as np
import random

# spartan
import spartan.utils.utils as spartan_utils
from spartan.utils import transformations
from spartan.utils import constants

pos = np.array([0.61, 0.15, 0.0])
# pos = np.array([0.55, 0.15, 0.0])
quat = np.array([1, 0, 0, 0])
T_W_P_goal = transformations.quaternion_matrix(quat)
T_W_P_goal[:3, 3] = pos

# point on which you want to make contact with plate, expressed in frame
# WSG_50_base_link which also happens to be the command frame
# push_point_gripper_frame = np.array([0.0259909128079909, 0, 0.12675232381954094])
push_point_gripper_frame = np.array([0.0259909128079909, 0, 0.14])

# push point to command frame
T_C_PP = np.eye(4)
T_C_PP[:3, 3] = push_point_gripper_frame
T_PP_C = np.linalg.inv(T_C_PP)

# push point to
T_P_PP = np.eye(4)
T_P_PP[:3, 3] = np.array([0.0, -0.10, 0.01795])

# rotation part
T_P_PP[:3, 0] = np.array([0,1,0])
T_P_PP[:3, 1] = np.array([1,0,0])
T_P_PP[:3, 2] = np.array([0,0,-1])

T_P_C = np.matmul(T_P_PP, T_PP_C) # command frame to plate frame


def compute_distance_to_goal(T_W_P):
    return np.linalg.norm(T_W_P[:3, 3] - T_W_P_goal[:3, 3])

def should_terminate(T_W_P, demonstration=False):
    eps = 0.01
    if demonstration:
        eps = 0.0025

    dist_to_goal = compute_distance_to_goal(T_W_P)
    terminate = dist_to_goal < eps
    return dist_to_goal, terminate


def sample_initial_pose(fixed_initial_pose=False):
    q0 = np.array([0.61, -0.05, 0.1, 0.0, 0.0, 0.0])
    q0[0] = random.uniform(0.55, 0.65)
    q0[1] = random.uniform(-0.2, -0.12)

    if fixed_initial_pose:
        q0[0] = 0.55
        q0[1] = -0.12

    return q0




