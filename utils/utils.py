import numpy as np
from spartan.utils import constants

def get_T_W_Cnominal_from_config(config):
    T_W_Cnominal = None

    # this is just to preserve backwards compatibility
    if "T_W_Cnominal" in config:
        T_W_Cnominal = np.array(config["T_W_Cnominal"])
    elif "T_W_Enominal" in config:
        T_W_E_init = np.array(config["T_W_Enominal"])
        T_W_Cnominal = np.matmul(T_W_E_init, constants.T_E_cmd)
    else:
        T_W_Cnominal = constants.T_W_cmd_init

    return T_W_Cnominal

def get_gripper_width_default_from_config(config):
    # backwards compatibility
    gripper_width_default = 0.1
    if "gripper_width_default" in config:
        gripper_width_default = config['gripper_width_default']

    return gripper_width_default

def get_deploy_image_topic_from_config(config):
    # backwards compatibility
    topic = "/camera_sim_d415_right/rgb/image_rect_color"
    if "deploy_image_topic" in config:
        topic = config["deploy_image_topic"]

    return topic

def get_image_index_to_sample_from_config(config):

    # backwards compatibility
    index = 0
    if "image_index_to_sample" in config:
        index = config["image_index_to_sample"]

    return index

def get_software_safety_params_from_config(config):
    params = {"translation_threshold": 0.02,
              "rotation_threshold_degrees": 1.5}

    if "software_safety" in config:
        params = config["software_safety"]

    return params