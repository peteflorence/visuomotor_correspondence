from __future__ import print_function

import numpy as np

import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.add_dense_correspondence_to_python_path()
from dense_correspondence.correspondence_tools import correspondence_finder


from imitation_agent.objects import sugar_box
def check_point_in_frame(pt, K=None, urange=None, vrange=None, T_W_pt=None, T_W_camera=None):
    if T_W_pt is not None:
        pt = np.matmul(T_W_pt, np.append(pt, 1))[0:3]

    if urange is None:
        urange = [0, 640]

    if vrange is None:
        vrange = [0, 480]


    uv = correspondence_finder.pinhole_projection_world_to_image(pt, K, camera_to_world=T_W_camera)
    # print("uv", uv)

    return (uv[0] >= urange[0]) and (uv[0] <= urange[1]) and (uv[1] >= vrange[0]) and (uv[1] <= vrange[1])



def check_sugar_box_in_frame(K, T_W_camera, T_W_B, urange=None, vrange=None, verbose=False):
    pts = sugar_box.points
    # print("pts.shape", pts.shape)

    num_pts = pts.shape[0]
    for i in range(0, num_pts):
        pt = pts[i, :]
        pt_in_frame = check_point_in_frame(pt, K=K, urange=urange, vrange=vrange, T_W_pt=T_W_B, T_W_camera=T_W_camera)

        if not pt_in_frame:
            return False

    return True

