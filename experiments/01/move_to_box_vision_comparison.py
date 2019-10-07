from __future__ import print_function

import os

# spartan
import spartan.utils.utils as spartan_utils

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.set_cuda_visible_devices([1])

# imitation_agent
from imitation_agent.training import train_mlp_position
from imitation_agent.training import train_utils

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_0710")

logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_0710_box_in_frame.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

# These will primarily determine how long this evaluation runs
NUM_REPEAT_TRIALS  = 1     # set to more than 1 to run repeat trials
NUM_DEPLOYS_EACH   = 0     # set to 100 to actually run deploys
NUM_TRAINING_STEPS = 75000
TRAIN              = True  # set to False to not train

# For deploys
LOGGING_DIR_ROOT = "/home/peteflo/spartan/deploy_evaluations/experiment_01"

# --- baseline comparisons

DO_BLIND                  = False
DO_GT_POSE                = False
DO_GT_3D_POINTS_PROJECTED = False  
DO_DSAE                   = False
DO_DSAE_E2E               = True   
DO_E2E                    = False
 
# ---- DD comparisons

DO_DD_VANILLA             = False 
DO_DD_SURF                = False
DO_DD_soft_3d             = False 
DO_DD_hard_3d             = False

num_logs = 200
for trial_seed in range(NUM_REPEAT_TRIALS):
	
	config_yaml = os.path.join(imitation_src_dir, "experiments", "01", "01_mlp_stateless_position.yaml")
	config = spartan_utils.getDictFromYamlFilename(config_yaml)
	config["global_training_steps"] = NUM_TRAINING_STEPS
	config["num_downsampled_logs"] = num_logs


	SAVE_TO = "pdc/imitation/trained_models/mlp_position/experiment_01-sep3-logs-"+str(num_logs)
	
	config["num_downsampled_logs"] = num_logs
	logs_config_downsampled = train_utils.deterministic_downsample(logs_config, config["num_downsampled_logs"])

	# """
	# Baseline 0: blind
	# """
	config["use_vision"] = False
	config["use_gt_object_pose"] = False
	config["project_pose_into_camera"] = False
	config["model"]["vision_net"] = "none"
	config["model"]["config"]["num_ref_descriptors"] = 1 # but not used
	config["precompute_features"] = False # but not used
	config["freeze_vision"] = False # but not used
	if DO_BLIND:
		print("Train blind")
		# -----> train

		name = "00-blind-"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Baseline 1: use GT pose
	"""
	
	config["use_gt_object_pose"] = True

	if DO_GT_POSE:	
		print("Train with GT pose")
		# -----> train
		name = "01-gt-pose-"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# # -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Baseline 3: use GT projected point
	"""
	config["project_pose_into_camera"] = True

	if DO_GT_3D_POINTS_PROJECTED:
		print("Train with GT pose projected")
		# -----> train
		name = "03-gt-3D-points-projected-"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	# """
	# Baseline 4: use DSAE
	# """
	config["use_vision"] = True
	config["use_gt_object_pose"] = False
	config["project_pose_into_camera"] = False
	config["model"]["vision_net"] = "SpatialAutoencoder"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = True 
	config["freeze_vision"] = True 

	if DO_DSAE:
		print("Train with DSAE vision")
		# -----> train

		name = "04-dsae_full_width_dropout-slow-"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	config["precompute_features"] = False
	config["freeze_vision"] = False

	if DO_DSAE_E2E:
		print("Train with DSAE vision, then E2E")
		# -----> train

		name = "04-dsae_full_widthE2E-"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Baseline 5: use End-to-End
	"""
	config["use_vision"] = True
	config["model"]["vision_net"] = "EndToEnd"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = False
	config["freeze_vision"] = False
	config["model"]["u_range_start"] = 0
	config["model"]["u_range_end"] = 640
	
	if DO_E2E:
		print("Train End-to-End")
		#-----> train
		name = "05-endtoend-full-image-seed"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)

	
	# """
	# Comparison: use dense descriptors
	# """
	
	
	config["model"]["vision_net"] = "DonSpatialSoftmax"
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_0710_in_frame_1e-6lamda_10/010000.pth"
	config["model"]["config"]["num_ref_descriptors"] = num_ref = 16
	config["precompute_features"] = True
	config["precompute_descriptor_images"] = False
	config["freeze_vision"] = True
	config["surf_descriptors"] = False

	if DO_DD_VANILLA:
		print("Train with dense correspondence features")
		# -----> train
		name = "06-dd-d_images-"+str(num_ref)+"-10D"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)

	"""
	Comparison: DD surf
	"""
	config["precompute_features"] = False
	config["precompute_descriptor_images"] = True
	config["surf_descriptors"] = True

	if DO_DD_SURF:
		print("Train with dense correspondence features")
		# -----> train
		name = "06-dd-d_images-SURF-"+str(num_ref)+"-10D"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Comparison: DD 3D soft
	"""

	config["use_depth"] = True
	config["use_soft_3D_unprojection"] = True
	config["precompute_features"] = True
	config["precompute_descriptor_images"] = False
	config["freeze_vision"] = True
	config["surf_descriptors"] = False
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_0710_in_frame_1e-6lamda_3d_10-soft/010000.pth"

	if DO_DD_soft_3d:
		print("Train with dense correspondence features into 3D")
		# -----> train

		name = "06-dd-3D-soft-cameraframe"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Comparison: DD 3D hard
	"""

	config["use_soft_3D_unprojection"] = False
	config["use_hard_3D_unprojection"] = True
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_0710_in_frame_1e-6lamda_3d_10-hard/010000.pth"

	if DO_DD_hard_3d:
		print("Train with dense correspondence features into 3D")
		# -----> train
		name = "06-dd-3D-hard-cameraframe"+str(trial_seed)
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py   --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)