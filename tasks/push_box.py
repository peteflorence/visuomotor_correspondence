# system
import numpy as np

# spartan
import spartan.utils.utils as spartan_utils
from spartan.utils import transformations
from spartan.utils import constants

pos = np.array([ 0.64446168, -0.37246525,  0.15697373])
quat = np.array([ 0.68974905,  0.13396046,  0.69895202, -0.13329258])
T_W_E_default = transformations.quaternion_matrix(quat)
T_W_E_default[:3, 3] = pos
T_W_C_default = np.matmul(T_W_E_default, constants.T_E_cmd)

Y_TARGET_POS = 0.23432 - 0.1

def compute_reward(T_W_B):
	# box to world transform
	y_pos = T_W_B[1,3]
	y_error = np.linalg.norm(Y_TARGET_POS - y_pos)


	# target rpy
	rpy_target = np.array([0,0,np.pi/2.0])
	R_W_B_target = transformations.euler_matrix(0, 0, np.pi/2.0)[:3, :3]

	R_B_B_target = np.matmul(np.linalg.inv(T_W_B[:3, :3]), R_W_B_target)
	angle_axis_err = spartan_utils.angle_axis_from_rotation_matrix(R_B_B_target)
	angle_err = np.linalg.norm(angle_axis_err)


	pos_scale = 1.0/0.1
	angle_scale = 1.0/np.deg2rad(2)
	reward = -1.0 * pos_scale * np.clip(y_error - 0.02, 0, 100.0) + -1.0*angle_scale*angle_err


	return {"angle_error": angle_err,
			"y_error": y_error,
			"reward": reward}


def should_terminate(T_W_B, demonstration=False):
	y_pos = T_W_B[1,3]
	y_target_pos = Y_TARGET_POS
	if not demonstration:
		y_target_pos -= 0.01 # go 2 more cm during demonstrations

	return (y_pos > y_target_pos)


