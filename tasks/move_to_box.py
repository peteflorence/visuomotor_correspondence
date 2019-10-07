"""
Stores relevant information for the move_to_box task
"""
import numpy as np
import math
from spartan.utils import transformations

def get_end_effector_to_box(above=False): # type -> nump.ndarray 4x4 homogeneous transform

    pos = np.array([0.00903093, 0.11725698, 0.13])
    if above:
        pos = np.array([0.00903093, 0.0, 0.18]) # for move_to_box_then_flip
    quat = np.array([-0.38365426, -0.58456535, -0.40083757,  0.59196453])

    # end-effector to sugarbox
    T_B_E = transformations.quaternion_matrix(quat)
    T_B_E[0:3,3] = pos

    return T_B_E


def compute_ee_target_pose(T_W_B, # type nump.ndarray 4x4 homogeneous transform
                           above=False): # type -> nump.ndarray 4x4 homogeneous transform

    T_B_E = get_end_effector_to_box(above)
    T_W_E = np.matmul(T_W_B, T_B_E)
    return T_W_E


def compute_reward(T_W_B, T_W_E): # type float
    """
    Computes reward for distance from goal
    """

    """
    NOTE:
    - the T_W_B used in demonstrations was relative to
      the "starting pose" (before the box fell onto the table)
      so we adjust for that here.
    """
    T_W_B_before_drop = T_W_B * 1.0
    T_W_B_before_drop[2,3] = 0.05

    # target position
    T_W_E_target = compute_ee_target_pose(T_W_B_before_drop)

    T_E_E_target = np.matmul(np.linalg.inv(T_W_E), T_W_E_target)

    d = compute_transform_cost(T_E_E_target)
    d['reward'] = -d["cost"]
    return d


def compute_transform_cost(T): # type float
    pos = np.linalg.norm(T[0:3, 3])
    angle, direction, point = transformations.rotation_from_matrix(T)

    cost = 1/0.01 * np.linalg.norm(pos) + 1/np.deg2rad(5) * abs(angle)

    return {"cost": cost,
            "pos": np.linalg.norm(pos),
            "angle": abs(angle)}