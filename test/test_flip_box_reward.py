import numpy as np
from imitation_agent.tasks import flip_box

import spartan.utils.utils as spartan_utils
from spartan.utils import transformations


def test_vertical_check():
    pos = np.array([ 0.67531002, -0.05338346,  0.0438477 ])
    quat = np.array([ 0.49558626, -0.50437512,  0.49558626,  0.50437512])
    T_W_B = transformations.quaternion_matrix(quat)
    T_W_B[:3, 3] = pos

    angle_error_to_target = flip_box.angle_error_to_target(T_W_B)
    print("angle error (degrees)", np.rad2deg(angle_error_to_target))
    print("is_vertical:", flip_box.is_vertical(T_W_B))
    print("is_sucessful:", flip_box.is_sucessful(T_W_B))

if __name__ == "__main__":
    test_vertical_check()