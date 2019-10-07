from __future__ import print_function

import torch
import numpy as np

# spartan
import spartan.utils.transformations as transformations
import spartan.utils.utils as spartan_utils
from spartan.utils import constants

from imitation_agent.utils import utils as imitation_agent_utils

VERBOSE = False

def compute_action(raw_actions, # vector
                   config, # dict, the global config
                   T_W_C=None, # homogeneous transform 4x4, current position of frame T_W_C
                   T_W_C_default=None, # default T_W_cmd to use to fill in missing indices
                   gripper_width_default=None):

	if gripper_width_default is None:
		gripper_width_default = imitation_agent_utils.get_gripper_width_default_from_config(config)

	assert T_W_C_default is not None

	T_W_cmd = np.eye(4)

	pos_default = T_W_C_default[:3, 3]

	# different modes, rpy relative, delta mode etc.
	position_mode = None # options are ["GLOBAL", "DELTA", "DELTA_WORLD_FRAME"]
	orientation_mode = None # options are ["RPY", "ANGLE_AXIS_DELTA", "QUAT"]

	action_config = config["action"]["config"]


	counter = 0
	if action_config["rpy"]["roll"] or action_config["rpy"]["pitch"] or action_config["rpy"]["yaw"]:
		orientation_mode = "RPY"
		counter += 1
	elif action_config["angle_axis_delta"]["x"] or action_config["angle_axis_delta"]["y"] or action_config["angle_axis_delta"]["z"]:
		orientation_mode = "ANGLE_AXIS_DELTA"
		counter += 1
	elif action_config["quaternion"]:
		orientation_mode = "QUAT"
		counter += 1
	elif action_config["angle_axis_relative_to_nominal"]["x"] or action_config["angle_axis_relative_to_nominal"]["y"] or action_config["angle_axis_relative_to_nominal"]["z"]:
		orientation_mode = "ANGLE_AXIS_RELATIVE_TO_NOMINAL"
	else:
		orientation_mode = "DEFAULT"

	if counter > 1:
		raise ValueError("multiple orientation modes were specified, please select only one")


	counter = 0
	if action_config["translation"]["x"] or action_config["translation"]["y"] or action_config["translation"]["z"]:
		position_mode = "GLOBAL"
		counter += 1
	elif action_config["translation_delta"]["x"] or action_config["translation_delta"]["y"] or action_config["translation_delta"]["z"]:
		position_mode = "DELTA"
		counter += 1
	elif action_config["translation_delta_world_frame"]["x"] or action_config["translation_delta_world_frame"]["y"] or action_config["translation_delta_world_frame"]["z"]:
		position_mode = "DELTA_WORLD_FRAME"
		counter += 1
	else:
		raise ValueError("no position mode specfified")

	if counter > 1:
		raise ValueError("multiple position modes were specified, please select only one")


	# figure out position
	if VERBOSE:
		print("position mode is:", position_mode)
		print("orientation mode is:", orientation_mode)
		print("raw_actions", raw_actions)

	counter = 0
	pos = np.zeros(3)
	if position_mode == "GLOBAL":
		pos = np.zeros(3)
		if action_config["translation"]["x"]:
			pos[0] = raw_actions[counter]
			counter += 1
		else:
			pos[0] = pos_default[0]

		if action_config["translation"]["y"]:
			pos[1] = raw_actions[counter]
			counter += 1
		else:
			pos[1] = pos_default[1]

		if action_config["translation"]["z"]:
			pos[2] = raw_actions[counter]
			counter += 1
		else:
			pos[2] = pos_default[2]

	elif position_mode == "DELTA":
		delta_pos = np.zeros(3)
		if action_config["translation_delta"]["x"]:
			delta_pos[0] = raw_actions[counter]
			counter += 1

		if action_config["translation_delta"]["y"]:
			delta_pos[1] = raw_actions[counter]
			counter += 1

		if action_config["translation_delta"]["z"]:
			delta_pos[2] = raw_actions[counter]
			counter += 1

		# figure out global position
		# note delta_pos is in frame C
		pos = T_W_C[:3, 3] + np.matmul(T_W_C, delta_pos)

	elif position_mode == "DELTA_WORLD_FRAME":
		delta_pos = np.zeros(3)
		if action_config["translation_delta_world_frame"]["x"]:
			delta_pos[0] = raw_actions[counter]
			counter += 1

		if action_config["translation_delta_world_frame"]["y"]:
			delta_pos[1] = raw_actions[counter]
			counter += 1

		if action_config["translation_delta_world_frame"]["z"]:
			delta_pos[2] = raw_actions[counter]
			counter += 1

		# figure out global position
		# note delta_pos is in frame C
		pos = T_W_C[:3, 3] + delta_pos

		if VERBOSE:
			print("delta_pos", delta_pos)

	# compute orientation
	R_W_cmd = np.eye(3)
	if orientation_mode == "RPY":
		raise NotImplementedError
	elif orientation_mode == "QUAT":
		raise NotImplementedError
	elif orientation_mode == "ANGLE_AXIS_DELTA":
		angle_axis = np.zeros(3) # expressed in gripper_frame

		if action_config["angle_axis_delta"]["x"]:
			angle_axis[0] = raw_actions[counter]
			counter += 1

		if action_config["angle_axis_delta"]["y"]:
			angle_axis[1] = raw_actions[counter]
			counter += 1

		if action_config["angle_axis_delta"]["z"]:
			angle_axis[2] = raw_actions[counter]
			counter += 1

		R_C_cmd = spartan_utils.rotation_matrix_from_angle_axis(angle_axis)
		R_W_cmd = np.matmul(T_W_C[:3, :3], R_C_cmd)

	elif orientation_mode == "ANGLE_AXIS_RELATIVE_TO_NOMINAL":
		angle_axis_relative_to_nominal = np.zeros(3)

		if action_config["angle_axis_relative_to_nominal"]["x"]:
			angle_axis_relative_to_nominal[0] = raw_actions[counter]
			counter += 1

		if action_config["angle_axis_relative_to_nominal"]["y"]:
			angle_axis_relative_to_nominal[1] = raw_actions[counter]
			counter += 1

		if action_config["angle_axis_relative_to_nominal"]["z"]:
			angle_axis_relative_to_nominal[2] = raw_actions[counter]
			counter += 1

		R_Cnominal_cmd = spartan_utils.rotation_matrix_from_angle_axis(angle_axis_relative_to_nominal)
		R_W_cmd = np.matmul(T_W_C_default[:3, :3], R_Cnominal_cmd)

	elif orientation_mode == "DEFAULT":
		R_W_cmd = T_W_C_default[:3, :3]

	gripper_width = gripper_width_default
	if action_config["gripper"]["width"]:
		gripper_width = raw_actions[counter]
		counter += 1

	assert(len(raw_actions) == counter, "actions not parsed correctly")


	T_W_cmd = np.eye(4)
	T_W_cmd[:3, 3] = pos
	T_W_cmd[:3, :3] = R_W_cmd

	action = dict()
	action["T_W_cmd"] = T_W_cmd
	action["gripper_width"] = gripper_width
	action["T_W_E"] = np.matmul(T_W_cmd, constants.T_cmd_E)
	return action