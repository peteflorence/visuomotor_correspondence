from scipy.spatial.transform import Rotation
import numpy as np

import spartan.utils.transformations as transformations
import robot_msgs.msg
import spartan.utils.utils as spartan_utils

def make_cartesian_gains_msg(kp_rot, kp_trans):
    msg = robot_msgs.msg.CartesianGain()

    msg.rotation.x = kp_rot
    msg.rotation.y = kp_rot
    msg.rotation.z = kp_rot

    msg.translation.x = kp_trans
    msg.translation.y = kp_trans
    msg.translation.z = kp_trans

    return msg

def tf_matrix_from_pose(pose):
    trans, quat = pose
    mat = transformations.quaternion_matrix(quat)
    mat[:3, 3] = trans
    return mat


def interpolate_frames(T_W_start, T_W_end, max_translation, max_rotation):
    delta_pos = T_W_end[:3, 3] - T_W_start[:3, 3]

    if np.linalg.norm(delta_pos) > max_translation:
        delta_pos = delta_pos * max_translation / np.linalg.norm(delta_pos)

    T_start_W = np.linalg.inv(T_W_start)
    T_start_end = np.matmul(T_start_W, T_W_end)
    angle_axis = spartan_utils.angle_axis_from_rotation_matrix(T_start_end[:3, :3])
    if np.linalg.norm(angle_axis) > max_rotation:
        angle_axis *= max_rotation/np.linalg.norm(angle_axis)


    dR = Rotation.from_rotvec(angle_axis).as_dcm()

    T = np.eye(4)
    T[:3, 3] = T_W_start[:3, 3] + delta_pos
    T[:3, :3] = np.matmul(T_W_start[:3, :3], dR)
    return T


