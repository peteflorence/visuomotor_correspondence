from __future__ import print_function

import os
import copy

# spartan
import spartan.utils.utils as spartan_utils

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.set_cuda_visible_devices([0])

# imitation_agent
from imitation_agent.training import train_lstm
from imitation_agent.training import train_utils

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()

# logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_se2")
# logs_dir_path = os.path.join(data_dir, "pdc/imitation/push_box")
# logs_dir_path = "/home/manuelli/data_ssd/imitation/logs/push_box" # set individually
logs_dir_path = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716"
config_yaml = os.path.join(imitation_src_dir, "experiments", "05", "05_lstm_sequence.yaml")


logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_then_flip_0716_box_in_frame.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


# These will primarily determine how long this evaluation runs
NUM_REPEAT_TRIALS = 1
# NUM_DEPLOYS_EACH = 100
NUM_DEPLOYS_EACH = 20 # testing

GLOBAL_TRAINING_STEPS = 200000
# GLOBAL_TRAINING_STEPS = 150 # testing

# DEBUG
# NUM_REPEAT_TRIALS = 1
# NUM_DEPLOYS_EACH = 0
# GLOBAL_TRAINING_STEPS = 150 # testing

# SAVE_TO = "pdc/imitation/trained_models/experiment_05/lstm_standard"
SAVE_TO = "pdc/imitation/trained_models/experiment_05_debug/lstm_standard"
DEPLOY_LOGGING_DIR = os.path.join(data_dir, "pdc/imitation/deploy_evaluation/experiment_05_debug")


# DO_GT_STATIC_POSE = False
# DO_GT_POSE = False
# DO_GT_3D_POINTS = False
# DO_GT_3D_POINTS_PROJECTED = False
# DO_DD = False
# DO_END_TO_END = True


TRAIN_GT_STATIC_POSE = False
TRAIN_GT_POSE = False
TRAIN_GT_3D_POINTS = False
TRAIN_GT_3D_POINTS_PROJECTED = False
TRAIN_DD = False
TRAIN_END_TO_END = False

DEPLOY_GT_STATIC_POSE = False
DEPLOY_GT_POSE = False
DEPLOY_GT_3D_POINTS = False
DEPLOY_GT_3D_POINTS_PROJECTED = False
DEPLOY_DD = True
DEPLOY_END_TO_END = False




# NUM_LOGS = [100, 50]
# NUM_LOGS = [25]
NUM_LOGS = [100]


TRAINING_POSES_FILE = os.path.join(spartan_source_dir, "modules/imitation_agent/config/dataset/move_to_box_then_flip_0716_in_frame_100_logs.npy")

# DEPLOY_COMMAND_ROOT = "cd ../../evaluation && python move_to_box_evaluator.py --flip --lstm --training_poses_file %s --logging_dir" %(TRAINING_POSES_FILE, DEPLOY_LOGGING_DIR)


def get_deploy_cmd(seed, deploy_idx, network_chkpt):
	# seed = 2 # HACK FOR DEBUGGING
	cmd = "cd ../../evaluation && python move_to_box_evaluator.py --flip --lstm --training_poses_file %s --logging_dir %s --seed %d --deploy-idx %d --net %s" % (
	TRAINING_POSES_FILE, DEPLOY_LOGGING_DIR, seed, deploy_idx, network_chkpt)

	return cmd

# for trial_seed in range(NUM_REPEAT_TRIALS):

for num_logs in NUM_LOGS:
	trial_seed = 1 # for use below

	# reload the config
	config = spartan_utils.getDictFromYamlFilename(config_yaml)
	config["global_training_steps"] = GLOBAL_TRAINING_STEPS
	config["num_downsampled_logs"] = num_logs
	config["num_workers"] = 20

	logs_config_downsampled = train_utils.deterministic_downsample(logs_config, num_logs)

	# """
	# Baseline 0: use GT static pose
	# """
	#
	# config["use_vision"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["x"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["y"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["z"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["x"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["y"] = False
	# config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["y"] = False
	#
	# config["observation"]["config"]["angle_axis_relative_to_nominal"]['x'] = True
	# config["observation"]["config"]["angle_axis_relative_to_nominal"]['y'] = True
	# config["observation"]["config"]["angle_axis_relative_to_nominal"]['z'] = True
	#
	# config["observation"]["config"]["ee_points"].append([0.0, 0.1, 0.0])
	# config["observation"]["config"]["ee_points"].append([0.0, 0.0, 0.1])
	# config["observation"]["config"]["project_gt_object_points_into_camera"] = False
	#
	# config["observation"]["config"]["gripper"]["width"] = True
	#
	# # action
	# config["action"]["config"]["angle_axis_relative_to_nominal"]['x'] = True
	# config["action"]["config"]["angle_axis_relative_to_nominal"]['y'] = True
	# config["action"]["config"]["angle_axis_relative_to_nominal"]['z'] = True
	#
	# config["action"]["config"]["angle_axis_delta"]['x'] = False
	# config["action"]["config"]["angle_axis_delta"]['y'] = False
	# config["action"]["config"]["angle_axis_delta"]['z'] = False
	#
	#
	# # these below are static
	# config["use_gt_object_pose"] = True
	# config["project_pose_into_camera"] = False
	#
	# config["model"]["vision_net"] = "none"
	# config["model"]["config"]["num_ref_descriptors"] = 1  # but not used
	# config["precompute_features"] = False  # but not used
	# config["freeze_vision"] = False  # but not used
	#
	# if DO_GT_STATIC_POSE:
	# 	print("Train with GT static pose")
	# 	# -----> train
	# 	save_dir = os.path.join(data_dir, SAVE_TO, "00-gt-pose-num-logs-%d" % (num_logs))
	# 	train_utils.make_deterministic(trial_seed)
	# 	train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)
	# 	longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
	# 	network_chkpt = os.path.join(save_dir, longest_chkpt)
	# 	# # -----> eval
	# 	for i in range(NUM_DEPLOYS_EACH):
	# 		cmd = "cd %s/evaluation && python move_to_box_evaluator.py --flip --lstm --seed %d --deploy-idx %d --net %s" % (
	# 			imitation_src_dir, i, i, network_chkpt)
	# 		os.system(cmd)


	"""
	Baseline 1: use GT dynamic pose
	"""

	config["use_vision"] = False
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["x"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["y"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["z"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["x"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["y"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["z"] = True

	config["observation"]["config"]["ee_points"].append([0.0, 0.1, 0.0])
	config["observation"]["config"]["ee_points"].append([0.0, 0.0, 0.1])
	config["observation"]["config"]["project_gt_object_points_into_camera"] = False

	# gripper width True
	config["observation"]["config"]["gripper"]["width"] = True

	# these below are static
	config["use_gt_object_pose"] = False
	config["project_pose_into_camera"] = False

	config["model"]["vision_net"] = "none"
	config["model"]["config"]["num_ref_descriptors"] = 1 # but not used
	config["precompute_features"] = False # but not used
	config["freeze_vision"] = False # but not used



	save_dir = os.path.join(data_dir, SAVE_TO, "01-gt-pose-num-logs-%d" % (num_logs))
	if TRAIN_GT_POSE:
		print("Train with GT pose")
		# -----> train
		train_utils.make_deterministic(trial_seed)
		train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)



	# # -----> eval
	if DEPLOY_GT_POSE:
		longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
		network_chkpt = os.path.join(save_dir, longest_chkpt)
		for i in range(NUM_DEPLOYS_EACH):
			cmd = get_deploy_cmd(i, i, network_chkpt)
			os.system(cmd)


	"""
	Baseline 2: use GT 3D points
	"""

	del config["observation"]["config"]["use_dynamic_gt_object_pose"]


	config["observation"]["config"]["gt_object_points"] = []
	config["observation"]["config"]["gt_object_points"].append([0.0, 0.0, 0.0])
	config["observation"]["config"]["gt_object_points"].append([0.0, 0.1, 0.0])
	config["observation"]["config"]["gt_object_points"].append([0.0, 0.0, 0.1])
	config["observation"]["config"]["project_gt_object_points_into_camera"] = False



	# -----> train
	save_dir = os.path.join(data_dir, SAVE_TO, "02-gt-3D-points-%d" %(num_logs))

	if TRAIN_GT_3D_POINTS:
		print("Train with GT 3D points")
		train_utils.make_deterministic(trial_seed)
		train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)



	# # -----> eval
	if DEPLOY_GT_3D_POINTS:
		longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
		network_chkpt = os.path.join(save_dir, longest_chkpt)
		for i in range(NUM_DEPLOYS_EACH):
			cmd = get_deploy_cmd(i, i, network_chkpt)
			os.system(cmd)

	"""
	Baseline 3: use GT projected points
	"""
	config["observation"]["config"]["project_gt_object_points_into_camera"] = True



	# -----> train
	save_dir = os.path.join(data_dir, SAVE_TO, "03-gt-3D-points-projected-num-logs-%d" %(num_logs))

	if TRAIN_GT_3D_POINTS_PROJECTED:
		print("Train with GT pose projected")
		train_utils.make_deterministic(trial_seed)
		train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)




	if DEPLOY_GT_3D_POINTS_PROJECTED:
		longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
		network_chkpt = os.path.join(save_dir, longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			cmd = get_deploy_cmd(i, i, network_chkpt)
			os.system(cmd)

	"""
	Baseline 4: use Dense Descriptors 
	"""
	# """
	# Comparison: use dense descriptors
	# """
	del config["observation"]["config"]["gt_object_points"]
	config["use_vision"] = True
	config["model"]["vision_net"] = "DonSpatialSoftmax"
	config["model"]["descriptor_net"] = os.path.join(data_dir, "pdc/trained_models/imitation/flip_box_0716_in_frame_10/010000.pth")
	# config["model"]["reference_vec"] = os.path.join(data_dir, "pdc/imitation/features/sugar_spatial_flip_0807_only_last_10/000/reference_descriptors.pth")
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = False
	config["precompute_descriptor_images"] = True
	config["freeze_vision"] = True
	config["surf_descriptors"] = True



	#-----> train
	save_dir = os.path.join(data_dir, SAVE_TO, "05-dd-num-logs-%d" %(num_logs))

	if TRAIN_DD:
		print("Train with dense correspondence features")
		train_utils.make_deterministic(trial_seed)
		train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)


	if DEPLOY_DD:
		longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
		network_chkpt = os.path.join(save_dir, longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			cmd = get_deploy_cmd(i, i, network_chkpt)
			os.system(cmd)


	"""
	Baseline 5: use End-to-End
	"""
	config["use_vision"] = True
	config["model"]["vision_net"] = "EndToEnd"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["model"]["u_range_start"] = 0
	config["model"]["u_range_end"] = 480
	config["precompute_features"] = False
	config["precompute_descriptor_images"] = False
	config["freeze_vision"] = False



	# -----> train
	save_dir = os.path.join(data_dir, SAVE_TO, "04-endtoend-num-logs-%d" %(num_logs))

	if TRAIN_END_TO_END:
		print("Train End-to-End")
		train_utils.make_deterministic(trial_seed)
		train_lstm.train(save_dir, config, logs_config_downsampled, logs_dir_path)



	if DEPLOY_END_TO_END:
		longest_chkpt = sorted(os.listdir(save_dir))[-2]  # "tensorboard is last"
		network_chkpt = os.path.join(save_dir, longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			cmd = get_deploy_cmd(i, i, network_chkpt)
			os.system(cmd)