from __future__ import print_function

import os

# spartan
import spartan.utils.utils as spartan_utils

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.set_cuda_visible_devices([0])

# imitation_agent
from imitation_agent.training import train_mlp_position
from imitation_agent.training import train_utils

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()

# logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_se2")
logs_dir_path = "/home/manuelli/data_ssd/imitation/logs/push_box" # set individually

config_yaml = os.path.join(imitation_src_dir, "experiments", "03", "03_mlp_stateless_position.yaml")
config = spartan_utils.getDictFromYamlFilename(config_yaml)

logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/push_box.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


# These will primarily determine how long this evaluation runs
NUM_REPEAT_TRIALS = 1
NUM_DEPLOYS_EACH = 100
# config["global_training_steps"] = 75000

NUM_DEPLOYS_EACH = 0
config["global_training_steps"] = 150 # just for testing purposes
# config["num_downsampled_logs"] = 200
config["num_downsampled_logs"] = 1

SAVE_TO = "pdc/imitation/trained_models/experiment_03/mlp_position"

logs_config_downsampled = train_utils.deterministic_downsample(logs_config, config["num_downsampled_logs"])


DO_GT_POSE = True
DO_GT_3D_POINTS = False
DO_GT_3D_POINTS_PROJECTED = True
DO_END_TO_END = False
DO_DD = False

for trial_seed in range(NUM_REPEAT_TRIALS):

#for num_logs in [50]:

	"""
	Baseline 1: use GT dynamic pose
	"""

	config["use_vision"] = False
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["x"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["y"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["z"] = True

	config["observation"]["config"]["ee_points"].append([0.0, 0.1, 0.0])
	config["observation"]["config"]["ee_points"].append([0.0, 0.0, 0.1])
	config["observation"]["config"]["project_gt_object_points_into_camera"] = False
	
	# these below are static
	config["use_gt_object_pose"] = False
	config["project_pose_into_camera"] = False

	config["model"]["vision_net"] = "none"
	config["model"]["config"]["num_ref_descriptors"] = 1 # but not used
	config["precompute_features"] = False # but not used
	config["freeze_vision"] = False # but not used

	if DO_GT_POSE:
		print("Train with GT pose")
		# -----> train
		save_dir = os.path.join(data_dir, SAVE_TO, "01-gt-pose")
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# # -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			cmd = "cd %s/evaluation && python push_box_evaluator.py --mlp --seed %d --deploy_idx %d --net %s" % (
			imitation_src_dir, i, i, network_chkpt)
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
	

	if DO_GT_3D_POINTS:
		print("Train with GT 3D points")
		# -----> train
		save_dir = os.path.join(data_dir, SAVE_TO, "02-gt-3D-points")
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# # -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			os.system("cd ../../evaluation && python move_to_box_evaluator.py --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt)

	"""
	Baseline 3: use GT projected points
	"""
	config["observation"]["config"]["project_gt_object_points_into_camera"] = True

	if DO_GT_3D_POINTS_PROJECTED:
		print("Train with GT pose pro`jected")
		# -----> train
		save_dir = os.path.join(data_dir, SAVE_TO, "03-gt-3D-points-projected")
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			cmd = "cd %s/evaluation && python push_box_evaluator.py --lstm --seed %d --deploy_idx %d --net %s" %(imitation_src_dir, i, i, network_chkpt)
			os.system(cmd)

	"""
	Baseline 5: use End-to-End
	"""
	del config["observation"]["config"]["gt_object_points"]
	config["use_vision"] = True
	config["model"]["vision_net"] = "EndToEnd"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = False
	config["freeze_vision"] = False

	if DO_END_TO_END:
		print("Train End-to-End")
		# -----> train
		save_dir = os.path.join(data_dir, SAVE_TO, "05-endtoend")
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			os.system("cd ../../evaluation && python move_to_box_evaluator.py --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt)

	
	# """
	# Comparison: use dense descriptors
	# """
	
	
	config["model"]["vision_net"] = "DonSpatialSoftmax"
	config["model"]["descriptor_net"] = os.path.join(data_dir, "pdc/trained_models/imitation/push_box_spatial_3d_10_cam1onlyScaled/015000.pth")
	# config["model"]["reference_vec"] = os.path.join(data_dir, "pdc/imitation/features/push_box_spatial_3d_10_cam1onlyScaled/000/reference_descriptors.pth")
	config["model"]["config"]["num_ref_descriptors"] = 16 
	config["precompute_features"] = True 
	config["freeze_vision"] = True

	if DO_DD:
		print("Train with dense correspondence features")
		#-----> train
		save_dir = os.path.join(data_dir, SAVE_TO, "05-dd")
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			os.system("cd ../../evaluation && python move_to_box_evaluator.py --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt)