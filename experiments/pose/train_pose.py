from __future__ import print_function

import os

# spartan
import spartan.utils.utils as spartan_utils

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.set_cuda_visible_devices([0])

# imitation_agent
from imitation_agent.training import train_pose_estimation
from imitation_agent.training import train_utils

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/push_box")

config_yaml = os.path.join(imitation_src_dir, "experiments", "pose", "pose_stateless.yaml")
config = spartan_utils.getDictFromYamlFilename(config_yaml)

logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/push_box_small.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


# These will primarily determine how long this evaluation runs
config["global_training_steps"] = 10000
config["num_downsampled_logs"] = 200

SAVE_TO = "pdc/imitation/trained_models/mlp_position/pose"

logs_config_downsampled = train_utils.deterministic_downsample(logs_config, config["num_downsampled_logs"])

# these below are static
config["use_gt_object_pose"] = False
config["project_pose_into_camera"] = False

config["observation"]["config"]["gt_object_points"] = []
config["observation"]["config"]["gt_object_points"].append([0.0, 0.0, 0.0])
config["observation"]["config"]["gt_object_points"].append([0.0, 0.1, 0.0])
config["observation"]["config"]["gt_object_points"].append([0.0, 0.0, 0.1])
config["observation"]["config"]["project_gt_object_points_into_camera"] = True


config["use_vision"] = True
config["model"]["vision_net"] = "EndToEnd"
config["model"]["config"]["num_ref_descriptors"] = 16
config["precompute_features"] = False
config["freeze_vision"] = False

print("Train Pose Estimation")

# -----> train
save_dir = os.path.join(data_dir, SAVE_TO, spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
train_utils.make_deterministic(0)
train_pose_estimation.train(save_dir, config, logs_config_downsampled, logs_dir_path)