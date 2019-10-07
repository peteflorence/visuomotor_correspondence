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
logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_se2")

logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_se2_box_in_frame.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

LOGGING_DIR_ROOT = "/home/peteflo/spartan/sandbox/experiment_02"


# These will primarily determine how long this evaluation runs
NUM_REPEAT_TRIALS = 1
NUM_DEPLOYS_EACH = 0 # set to 0 to not deploy. We are using 100.
NUM_TRAINING_STEPS = 75000
TRAIN = True         # set to False to not train


# --- baseline comparisons

DO_BLIND                  = False

DO_GT_POSE                = False
DO_GT_3D_POINTS           = False
DO_GT_3D_POINTS_PROJECTED = False
DO_DSAE                   = True
DO_E2E                    = False
 
# ---- DD comparisons

DO_DD_VANILLA             = False
DO_DD_SURF                = False
DO_DD_soft_3d             = False
DO_DD_hard_3d             = False

trial_seed = 0
#for num_logs in [200, 50, 30, 100]:
for num_logs in [200]:

	
	# # JUST TEMPORARY WHILE I HAVE A FEW ALREADY DONE

	# if num_logs == 200:
	# 	DO_GT_POSE = False
	# 	DO_DD_VANILLA = False
	# 	DO_DD_soft_3d = False
	# 	DO_DD_hard_3d = False
	# else:
	# 	DO_GT_POSE = False
	# 	DO_DD_VANILLA = False
	# 	DO_DD_soft_3d = False
	# 	DO_DD_hard_3d = False

	
	config_yaml = os.path.join(imitation_src_dir, "experiments", "02", "02_mlp_stateless_position.yaml")
	config = spartan_utils.getDictFromYamlFilename(config_yaml)
	config["global_training_steps"] = NUM_TRAINING_STEPS
	config["num_downsampled_logs"] = num_logs


	SAVE_TO = "pdc/imitation/trained_models/mlp_position/experiment_02-sep5-logs-"+str(num_logs)
	
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
		name = "00-blind"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+name)


	"""
	Baseline 1: use GT dynamic pose
	"""
	
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["x"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["translation"]["y"] = True
	config["observation"]["config"]["use_dynamic_gt_object_pose"]["angle_axis_relative_to_nominal"]["z"] = True
	config["observation"]["config"]["ee_points"].append([0.0, 0.1, 0.0])
	config["observation"]["config"]["ee_points"].append([0.0, 0.0, 0.1])
	config["observation"]["config"]["project_gt_object_points_into_camera"] = False

	if DO_GT_POSE:	
		print("Train with GT pose")
		# -----> train
		name = "01-gt-pose"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# # -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)



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
		name = "02-gt-3D-points"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)

	

	"""
	Baseline 3: use GT projected points
	"""
	config["observation"]["config"]["project_gt_object_points_into_camera"] = True

	if DO_GT_3D_POINTS_PROJECTED:
		print("Train with GT pose projected")
		# -----> train
		name = "03-gt-3D-points-projected"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	# """
	# Baseline 4: use DSAE
	# """
	del config["observation"]["config"]["gt_object_points"]
	config["use_vision"] = True
	config["model"]["vision_net"] = "SpatialAutoencoder"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = True 
	config["freeze_vision"] = True 

	if DO_DSAE:
		print("Train with DSAE vision")
		# -----> train
		name = "04-dsae-no-gt-no-mask"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Baseline 5: use End-to-End
	"""
	
	config["use_vision"] = True
	config["model"]["vision_net"] = "EndToEnd"
	config["model"]["config"]["num_ref_descriptors"] = 16
	config["precompute_features"] = False
	config["freeze_vision"] = False
	
	if DO_E2E:
		print("Train End-to-End")
		#-----> train
		name = "05-endtoend-fullwidth"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)

	
	# """
	# Comparison: use dense descriptors
	# """
	
	
	config["model"]["vision_net"] = "DonSpatialSoftmax"
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_se2_in_frame_1e-6lamda_10/010000.pth"
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
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)

	"""
	Comparison: DD surf
	"""
	config["precompute_features"] = False
	config["precompute_descriptor_images"] = True
	config["surf_descriptors"] = True

	if DO_DD_SURF:
		print("Train with dense correspondence features")
		# -----> train
		name = "06-dd-d_images-SURF-"+str(num_ref)+"-10D"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		if TRAIN:
			train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Comparison: DD 3D
	"""

	config["use_depth"] = True
	config["use_soft_3D_unprojection"] = True
	config["precompute_features"] = True
	config["precompute_descriptor_images"] = False
	config["freeze_vision"] = True
	config["surf_descriptors"] = False
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_se2_in_frame_1e-6lamda_3d_10-soft/010000.pth"

	if DO_DD_soft_3d:
		print("Train with dense correspondence features into 3D")
		# -----> train
		name = "06-dd-3D-soft-cameraframe"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)


	"""
	Comparison: DD 3D
	"""

	config["use_soft_3D_unprojection"] = False
	config["use_hard_3D_unprojection"] = True
	config["model"]["descriptor_net"] = "/home/peteflo/data/pdc/trained_models/imitation/move_to_box_se2_in_frame_1e-6lamda_3d_10-hard/010000.pth"

	if DO_DD_hard_3d:
		print("Train with dense correspondence features into 3D")
		# -----> train
		name = "06-dd-3D-hard-cameraframe"
		save_dir = os.path.join(data_dir, SAVE_TO, name)
		train_utils.make_deterministic(trial_seed)
		train_mlp_position.train(save_dir, config, logs_config_downsampled, logs_dir_path)
		longest_chkpt = sorted(os.listdir(save_dir))[-2] # "tensorboard is last"
		network_chkpt = os.path.join(save_dir,longest_chkpt)
		# -----> eval
		for i in range(NUM_DEPLOYS_EACH):
			logging_dir = os.path.join(LOGGING_DIR_ROOT, "logs-"+str(num_logs),name)
			os.system("cd ../../evaluation && python move_to_box_evaluator.py  --se2 --mlp --seed "+str(i)+ " --deploy-idx "+str(i)+" --net "+network_chkpt+" --logging_dir "+logging_dir)
