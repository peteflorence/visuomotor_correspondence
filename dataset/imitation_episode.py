import attr
import random
import yaml
from yaml import CLoader
import json
import numpy as np
import os

# torch
import torch

# spartan
import spartan.utils.utils as spartan_utils
import dense_correspondence_manipulation.utils.utils as pdc_utils
from spartan.utils import constants

# imitation_agent
from imitation_agent.dataset.directory_structure import SingleEpisodeDirectoryStructure


class ImitationEpisode(object):

    def __init__(self, config, # type ImitationEpisodeConfig
                 type="offline", # either online or offline
                 action_function=None,
                 observation_function=None,
                 processed_dir=None):

        self._config = config
        self._type = type

        if self._type == "offline":
            self.directory_structure = SingleEpisodeDirectoryStructure(config['path_to_processed_dir'])
            #self.state_dict = yaml.load(file(self.directory_structure.state_file), Loader=CLoader)
            self.state_dict = json.load(file(self.directory_structure.state_file))
            self.filter_valid_indices()
            self.num_elements = len(self.state_dict)

            # FOR DEBUG-ONLY USE ITEMS
            if os.path.isfile(self.directory_structure.sim_config_file):
                self.sim_config_dict = yaml.load(file(self.directory_structure.sim_config_file), Loader=CLoader)
            else:
                self.sim_config_dict = None

        # or if online
        else:
            self.directory_structure = SingleEpisodeDirectoryStructure(processed_dir)

            print "WARNING... ASSUMING CAMERAS HAVE NOT MOVED"

            self.sim_config_dict = dict()
            self.sim_config_dict["instances"] = []
            q0_dict = dict()
            self.sim_config_dict["instances"].append(q0_dict)

            # in training set for 2 boxes
            #self.sim_config_dict["instances"][0]["q0"] = [0.6632670558646401, -0.02846374967378179, 0.05]  # should basically be in front of robot
            self.sim_config_dict["instances"][0]["q0"] = [0.6369677479177303,  0.16540254417748684, 0.05] # "right" on table from person view

            # not in training set
            #self.sim_config_dict["instances"][0]["q0"] = [0.6632670558646401, -0.16, 0.05]                 # "left" on table from person view
            #self.sim_config_dict["instances"][0]["q0"] = [0.6632670558646401, -0.26, 0.05]                 # "far left" on table from person view
            #self.sim_config_dict["instances"][0]["q0"] = [0.6632670558646401, 0.26, 0.05]                 # "far right" on table from person view

        self.load_camera_info_dicts()

    def set_online_sim_config_dict(self, sim_config_dict):
        self.sim_config_dict = sim_config_dict

    def get_state_dict(self):
        return self.state_dict

    def set_state_dict(self, 
                       val, # type dict
                       ): # no return
        """
        Used by agents at runtime to update the state dict so they can then extract
        observations needed for running the policy
        """
        self.state_dict = val
        self.num_elements = len(self.state_dict)

    def filter_valid_indices(self): # no return
        """
        Remove some indices from log if necessary. For example where we
        have very low velocity
        """

        # for now don't filter anything
        valid_indices = range(0, len(self.state_dict))
        self._unfiltered_length = len(self.state_dict)

        # create new state dict with just those indices
        state_dict_filtered = dict()
        for counter, idx in enumerate(valid_indices):
            # Reason for str(idx)...
            # in json, these indices are strings, this effectively
            # makes the state_dict be int-indexed
            # at the end of this function, when it's written over.
            self.state_dict[str(idx)]["original_index"] = idx
            state_dict_filtered[counter] = self.state_dict[str(idx)]

        self.state_dict = state_dict_filtered

        if self._config["filtering"]["filter_no_movement"]:
            prev_data = self.state_dict[0]
            valid_indices = []
            valid_indices.append(0)
            for idx in range(1, len(self.state_dict)):
                data = self.state_dict[idx]

                T_W_E_prev = spartan_utils.homogenous_transform_from_dict(prev_data['observations']['ee_to_world'])
                T_W_C_prev = np.matmul(T_W_E_prev, constants.T_E_cmd)

                T_W_E = spartan_utils.homogenous_transform_from_dict(data['observations']['ee_to_world'])
                T_W_C = np.matmul(T_W_E, constants.T_E_cmd)

                dT = np.matmul(np.linalg.inv(T_W_C), T_W_C_prev)
                angle_axis = spartan_utils.angle_axis_from_rotation_matrix(dT[:3, :3])

                idx_is_valid = False

                if np.linalg.norm(dT[:3, 3]) > self._config["filtering"]['translation_threshold']:
                    idx_is_valid = True
                elif np.linalg.norm(angle_axis) > np.deg2rad(self._config["filtering"]["rotation_threshold_degrees"]):
                    idx_is_valid = True
                elif np.linalg.norm(data["observations"]["gripper_state"]["width"] - prev_data["observations"]["gripper_state"]["width"]) > self._config["filtering"]["gripper_width_threshold"]:
                    idx_is_valid = True

                if not idx_is_valid:
                    continue

                valid_indices.append(idx)
                prev_data = data



            print("original number of steps", len(self.state_dict))
            print("number steps after filtering", len(valid_indices))

            state_dict_tmp = dict()
            for counter, idx in enumerate(valid_indices):
                state_dict_tmp[counter] = self.state_dict[idx]


            self.state_dict = state_dict_tmp







        # do additional pre-processing
        for counter, idx in enumerate(range(len(self.state_dict))):
            action_idx = min(idx + self._config["action_bias"], len(self.state_dict) - 1)


            self.state_dict[idx]["actions_offset"] = self.state_dict[action_idx]["actions"]
            self.state_dict[idx]["control_offset"] = self.state_dict[action_idx]["control"]

            T_W_nxt_cmd = spartan_utils.homogenous_transform_from_dict(self.state_dict[action_idx]["actions"]["ee_setpoint"])
            T_W_E = spartan_utils.homogenous_transform_from_dict(self.state_dict[idx]['observations']['ee_to_world'])
            R_W_E = T_W_E[:3, :3]
            T_E_W = np.linalg.inv(T_W_E)
            T_E_nxt_cmd = np.matmul(T_E_W, T_W_nxt_cmd)

            # extract translation and angle axis
            # note these are expressed in the frame T_W_E
            translation = T_E_nxt_cmd[:3, 3]
            angle_axis = spartan_utils.angle_axis_from_rotation_matrix(T_E_nxt_cmd[:3, :3])

            # convert to world frame
            translation_W = np.dot(R_W_E, translation)
            angle_axis_W = np.dot(R_W_E, angle_axis)


            # note that this assumes no noise
            self.state_dict[idx]['actions_offset']['delta_to_EE_frame'] = dict()
            self.state_dict[idx]['actions_offset']['delta_to_EE_frame']['translation_world_frame'] = translation_W
            self.state_dict[idx]['actions_offset']['delta_to_EE_frame']['angle_axis_world_frame'] = angle_axis_W


            T_W_C = np.matmul(T_W_E, constants.T_E_cmd)
            self.state_dict[idx]['T_W_C'] = T_W_C
            self.state_dict[idx]['actions_offset']['T_W_C'] = np.matmul(T_W_nxt_cmd, constants.T_E_cmd)



    def get_random_idx(self):
        return np.random.randint(0, len(self))

    def __len__(self):
        return len(self.state_dict)

    def get_unfiltered_length(self):
        return self._unfiltered_length

    def get_entry(self, idx, # type int
                  ): # -> dict (inside the dict)
        """
        Get the dictionary entry for the given idx
        """
        state_dict = self.state_dict[idx]

        if self.sim_config_dict is not None:
            state_dict["debug_observations"] = dict()
            state_dict["debug_observations"]["object_starting_pose"] = dict()
            state_dict["debug_observations"]["object_starting_pose"]["x"] = self.sim_config_dict["instances"][0]["q0"][0]
            state_dict["debug_observations"]["object_starting_pose"]["y"] = self.sim_config_dict["instances"][0]["q0"][1]
            state_dict["debug_observations"]["object_starting_pose"]["z"] = self.sim_config_dict["instances"][0]["q0"][2]

        return state_dict


    def load_camera_info_dicts(self):
        self.camera_info_dicts = dict()
        for camera_num in range(2): # HACK
            camera_info_filename = self.directory_structure.get_camera_info_yaml(camera_num)
            self.camera_info_dicts[camera_num] = spartan_utils.getDictFromYamlFilename(camera_info_filename)

    def get_camera_info_dict(self, camera_num):
        return self.camera_info_dicts[camera_num]

    def get_camera_pose_data(self, camera_num):
        camera_info_dict = self.get_camera_info_dict(camera_num)
        return camera_info_dict["extrinsics"]

    def get_camera_pose_matrix(self, camera_num):
        """
        :return: 4 x 4 numpy array
        """
        pose_data = self.get_camera_pose_data(camera_num)
        return pdc_utils.homogenous_transform_from_dict(pose_data)

    def get_K_matrix(self, camera_num):
        camera_info_dict = self.get_camera_info_dict(camera_num)
        K = camera_info_dict["camera_matrix"]["data"]
        return np.asarray(K).reshape(3,3)


    @staticmethod
    def convert_to_flattened_torch_tensor(a):
        a = torch.from_numpy(a).flatten()
        return a.type(torch.FloatTensor)






