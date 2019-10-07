import numpy as np
import os
from spartan.utils import utils as spartan_utils
from spartan.utils import transformations
from imitation_agent.utils.visibility_utils import check_sugar_box_in_frame


def test_check_sugar_box_in_frame():
    # load camera info
    camera_info_file = os.path.join(os.getcwd(), '../config/camera_info_d415_left.yaml')
    camera_info_dict = spartan_utils.getDictFromYamlFilename(camera_info_file)
    K = camera_info_dict["camera_matrix"]["data"]
    K = np.asarray(K).reshape(3, 3)
    T_W_camera = spartan_utils.homogenous_transform_from_dict(camera_info_dict["extrinsics"])


    # out of frame
    pos = np.array([0.643599489628, -0.189629506326, 0.0200476981308])
    quat = np.array([0.679287637544, 3.47585450334e-18, 1.82053004607e-18, 0.733872131559])
    T_W_B = transformations.quaternion_matrix(quat)
    T_W_B[:3, 3] = pos

    if check_sugar_box_in_frame(T_W_camera=T_W_camera, T_W_B=T_W_B, K=K, verbose=True):
        raise ValueError("was supposed to be out of frame, but reported as in frame")


    # in frame
    print("")
    pos = np.array([0.640768703699, -0.177853584618, 0.0200476981308])
    quat = np.array([0.530492595959, -2.68732184169e-18, -1.82585972093e-18, 0.847689569142])

    T_W_B = transformations.quaternion_matrix(quat)
    T_W_B[:3, 3] = pos

    if not check_sugar_box_in_frame(T_W_camera=T_W_camera, T_W_B=T_W_B, K=K, verbose=True):
        raise ValueError("was supposed to be out in frame, but reported as out frame")

    print "test passed"
if __name__ == "__main__":
    test_check_sugar_box_in_frame()