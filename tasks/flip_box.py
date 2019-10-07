# system
import numpy as np

# spartan
import spartan.utils.utils as spartan_utils
from spartan.utils import transformations
from spartan.utils import constants

# R_W_B_target = np.eye(3)
# R_W_B_target[:, 0] = np.array([0, 0, -1])
# R_W_B_target[:, 1] = np.array([-1, 0, 0])
# R_W_B_target[:, 2] = np.array([0, 1, 0])

pos = [0.6389922858973416, 0.05023972760185024, 0.04384769813084115]
quat = [0.4954175578622114, -0.5045408242767304, 0.49541755786221153, 0.5045408242767303]
T_W_B_target = transformations.quaternion_matrix(quat)
T_W_B_target[:3, 3] = pos
R_W_B_target = T_W_B_target[:3, :3]

ANGLE_THRESHOLD = np.deg2rad(2.0)


def vectors_within_angle_threshold(v1, v2, threshold=ANGLE_THRESHOLD):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_angle > np.cos(ANGLE_THRESHOLD)


def is_vertical(T_W_B):
    x_axis = np.dot(T_W_B[:3, :3], np.array([1, 0, 0]))
    x_axis_target = np.array([0, 0, -1])
    return vectors_within_angle_threshold(x_axis, x_axis_target)

def angle_error_to_target(T_W_B):
    R_Btarget_B = np.matmul(np.linalg.inv(R_W_B_target), T_W_B[:3, :3])
    angle_axis = spartan_utils.angle_axis_from_rotation_matrix(R_Btarget_B)
    # print("angle_axis", angle_axis)
    angle_error = np.linalg.norm(angle_axis)
    return angle_error


def is_sucessful(T_W_B):
    return is_vertical(T_W_B) and (T_W_B[2, 3] > 0)

def compute_reward(T_W_B):
    success = is_sucessful(T_W_B)

    reward = 0.0
    if success:
        reward = 1.0

    angle_error = angle_error_to_target(T_W_B)
    angle_error_degrees = np.rad2deg(angle_error)

    return {"success": success,
            "reward": reward,
            "angle_error": angle_error,
            "angle_error_degrees": angle_error_degrees}
