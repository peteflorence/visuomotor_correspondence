from __future__ import print_function

import os
import copy
import math
import yaml
import logging
import numpy as np
import random

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import spartan.utils.utils as spartan_utils

from imitation_agent.dataset.imitation_episode import ImitationEpisode
import imitation_agent.dataset.dataset_utils as dataset_utils
from imitation_agent.dataset.precompute_helper_dataset import PrecomputeHelperDataset
from imitation_agent.dataset.feature_saver import FeatureSaver

from imitation_agent.model.spatial_autoencoder import SpatialAutoencoderWrapper


class ImitationEpisodeDataset(data.Dataset):

    def __init__(self,
                 logs_dir_path, # type str
                 log_config, # type dict
                 config, # type dict
                 action_function=None, # type function
                 observation_function=None, # type function
                 ):

        self._config = config
        self._action_function = action_function
        self._observation_function = observation_function

        self.episodes = dict()
        self._num_entries = 0
        self._use_only_first_index = False
        self._force_first_state_same = False
        self._noise_enabled = True

        print("loading logs...")

        # construct the ImitationEpisodeConfig for each log in our dataset
        for log_name in log_config['logs']:
            path_to_processed_dir = os.path.join(logs_dir_path, log_name, "processed")
            # print("path_to_processed_dir", path_to_processed_dir)
            assert os.path.isdir(path_to_processed_dir), path_to_processed_dir

            episode_config = copy.copy(self._config)
            episode_config['path_to_processed_dir'] = path_to_processed_dir

            imitation_episode = self.construct_imitation_episode(episode_config, type="offline")

            self.episodes[log_name] = imitation_episode
            self._num_entries += len(imitation_episode)

        if self._config["use_vision"]:
            self.spartan_dataset = dataset_utils.build_spartan_dataset(logs_dir_path)


    def construct_imitation_episode(self, config, # type dict
                                    type=None, # type str ("online", "offline")
                                    ): # type -> ImitationEpisode

        return ImitationEpisode(config, type=type)

    def get_empty_imitation_episode(self): # type -> ImitationEpisode
        return self.construct_imitation_episode(self._config, type="online")

    def get_random_log_name(self):
        return random.choice(self.episodes.keys())

    def set_use_only_first_index(self):
        self._use_only_first_index = True

    def unset_use_only_first_index(self):
        self._use_only_first_index = False

    def set_force_state_same(self):
        self._force_first_state_same = True

    def disable_noise(self):
        self._noise_enabled = False

    def augment_state_with_noise(self, state, sigma=None):
        if sigma is None:
            sigma = self._config["sigma_noise_augmentation"]  # this is in meters

        noise = torch.randn_like(state)*sigma

        # block noise on some values
        if self._config["use_gt_object_pose"] and not self._config["project_pose_into_camera"]:
            #print("NO NOISE ON GT OBJECT POSE, state shape", state.shape)
            noise[-3] = 0.0
            noise[-2] = 0.0
            noise[-1] = 0.0
        if self._config["use_gt_object_pose"] and self._config["project_pose_into_camera"]:
            #print("NO NOISE ON GT OBJECT POSE, state shape", state.shape)
            noise[-2] = 0.0
            noise[-1] = 0.0

        return state + noise

    def sample_noise_transform(self):
        if self._noise_enabled:
            translation = np.random.normal(0, self._config["position_noise_augmentation_sigma"], 3)
            angle_axis = np.random.normal(0, np.deg2rad(self._config["rotation_noise_augmentation_sigma"]), 3)
            R = spartan_utils.rotation_matrix_from_angle_axis(angle_axis)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = translation
        else:
            T = np.eye(4)

        return T

    def __getitem__(self, index, # type int
                    idx=None,
                    episode=None): # type -> dict
        """
        The method through which the dataset is accessed for training.

        We still need to normalize these with dataset mean and std dev
        """

        
        if self._use_only_first_index:
            log_name = self.episodes.keys()[index]
            episode = self.episodes[log_name]
            idx = 0
        elif idx is None and episode is None:
            log_name = self.get_random_log_name()
            episode = self.episodes[log_name]
            idx = episode.get_random_idx()
        elif idx is not None and episode is not None:
            pass
        else:
            raise ValueError("Expected one of above options")


        # figure out T_W_C from
        T_W_C = episode.get_entry(idx)['T_W_C']
        # add noise to this frame
        T_noise = self.sample_noise_transform()

        # now do action and observation function
        obs = torch.tensor(self._observation_function(episode, idx, T_noise=T_noise))
        action = torch.tensor(self._action_function(episode, idx, T_noise=T_noise))


        if not self._config["use_vision"]:
            image_data = torch.Tensor([])
        else:
            camera_num = self._config["camera_num"]
            image_data = self.get_image_data(log_name, idx, camera_num)

        depth_data, camera_to_world, K = self.get_3D_data(episode, log_name, idx)

        return {'action': action,
                'observation': obs,
                'image': image_data,
                'depth': depth_data,
                'camera_to_world': camera_to_world,
                'K': K}

    
    def get_image_data(self, log_name, idx, camera_num):
        original_index = self.episodes[log_name].get_entry(idx)["original_index"]
        rgb = self.spartan_dataset.get_rgb_image_from_scene_name_and_idx_and_cam(log_name, original_index, camera_num)
        rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(rgb)
        image_data = rgb_tensor
        return image_data

    def get_depth_data(self, log_name, idx, camera_num):
        original_index = self.episodes[log_name].get_entry(idx)["original_index"]
        depth = self.spartan_dataset.get_depth_image_from_scene_name_and_idx_and_cam(log_name, original_index, camera_num)
        depth = np.asarray(depth)
        depth = torch.from_numpy(depth).float()/1000.0 # convert to meters
        return depth

    def get_3D_data(self, episode, log_name, idx):
        camera_num = self._config["camera_num"]
        if not self._config["use_depth"]:
            depth_data = torch.Tensor([])
            camera_to_world = torch.Tensor([])
            K = torch.Tensor([])
        else:
            depth_data = self.get_depth_data(log_name, idx, camera_num)
            camera_to_world = torch.from_numpy(episode.get_camera_pose_matrix(camera_num))
            K = torch.from_numpy(episode.get_K_matrix(camera_num))
        return depth_data, camera_to_world, K

    def precompute_all_features(self, vision_net, 
                                      train_or_test): # type str, either "train" or "test"
        """
        Use the vision net to precompute all features and store in RAM.
        This can make training much faster when vision is frozen.
        """

        print("Precomputing features...")
        import time; start = time.time()

        if not isinstance(vision_net, SpatialAutoencoderWrapper):
            feature_saver = FeatureSaver()
            loaded = feature_saver.load_if_already_have("features", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)
        else:
            loaded = False

        if not loaded:

            # make dicts to store
            for log_name in self.episodes.keys():
                episode = self.episodes[log_name]
                episode.precomputed_features = dict()

            precompute_helper_dataset = PrecomputeHelperDataset(self.spartan_dataset, self.episodes, self._config["camera_num"], self)
            precompute_helper_loader = DataLoader(precompute_helper_dataset, batch_size=12, shuffle=False, num_workers=8)

            for step_counter, data in enumerate(precompute_helper_loader):
                print("Batch", step_counter, "of", len(precompute_helper_loader), "for precomputing features")
                feature_batch = vision_net.forward(data["rgb_tensor"].cuda(), data).detach().cpu()

                counter = 0
                for log_name, idx in zip(data["log_name"], data["idx"]):
                    episode = self.episodes[log_name]

                    # idx corresponds to the "original_idx" in state_dict
                    episode.precomputed_features[int(idx)] = feature_batch[counter]
                    counter += 1


            if not isinstance(vision_net, SpatialAutoencoderWrapper):
                feature_saver.save("features", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)

        print("I was able to precompute all features")
        print("Took me", time.time() - start, "seconds")

        # print("I was able to precompute all features")
        # print("Took me", time.time() - start, "seconds")

        # overwrite function
        self.get_image_data = self.get_precomputed_features
        self.get_depth_data = self.nothing

    def precompute_all_descriptor_images(self, vision_net, 
                                               train_or_test): # type str, either "train" or "test"
        """
        Use the vision net to precompute all descriptor images and store in RAM.
        Unlike `precompute_all_features`,
        this still allows surfing, etc.
        """

        print("Precomputing descriptor images...")
        import time; start = time.time()

        feature_saver = FeatureSaver()
        loaded = feature_saver.load_if_already_have("d_images", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)

        if not loaded:

            # make dicts to store
            for log_name in self.episodes.keys():
                episode = self.episodes[log_name]
                episode.precomputed_descriptor_images = dict()

            precompute_helper_dataset = PrecomputeHelperDataset(self.spartan_dataset, self.episodes, self._config["camera_num"], self)
            precompute_helper_loader = DataLoader(precompute_helper_dataset, batch_size=12, shuffle=False, num_workers=8)

            for step_counter, data in enumerate(precompute_helper_loader):
                #step_start = time.time()
                print("Batch", step_counter, "of", len(precompute_helper_loader), "for precomputing descriptor images")
                d_image_batch = vision_net.descriptor_net.forward(data["rgb_tensor"].cuda(), upsample=False).detach().cpu() # N, 3, 60, 80
                
                counter = 0
                for log_name, idx in zip(data["log_name"], data["idx"]):
                    episode = self.episodes[log_name]
                    episode.precomputed_descriptor_images[int(idx)] = d_image_batch[counter]
                    counter += 1
                #print(time.time() - step_start, "seconds for batch")


            feature_saver.save("d_images", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)

        print("I was able to precompute all descriptor images")
        print("Took me", time.time() - start, "seconds")

        # overwrite function
        self.get_image_data = self.get_precomputed_descriptor_images
        if not self._config["use_depth"]:
            self.get_depth_data = self.nothing

    def nothing(self, log_name, idx, camera_num):
        return torch.Tensor([])


    def precompute_only_first_frame_features(self, vision_net):
        """
        Use the vision net to precompute all features and store in RAM.
        This can make training much faster when vision is frozen.
        """

        print("Precomputing feature for first frame only...")
        counter = 0

        for log_name in self.episodes.keys():
            counter += 1
            print("log_name:", log_name, "num", counter, "of", len(self.episodes.keys()))

            episode = self.episodes[log_name]
            episode.precomputed_features = dict()

            camera_num = self._config["camera_num"]

            rgb = self.spartan_dataset.get_rgb_image_from_scene_name_and_idx_and_cam(log_name, 0, camera_num)
            rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(rgb).unsqueeze(0).cuda()

            depth_data = self.get_depth_data(log_name, 0, camera_num)
            camera_to_world = torch.from_numpy(episode.get_camera_pose_matrix(camera_num))
            K = torch.from_numpy(episode.get_K_matrix(camera_num))
            input_data = dict()
            input_data['depth'] = depth_data.unsqueeze(0)
            input_data['camera_to_world'] = camera_to_world.unsqueeze(0)
            input_data['K'] = K.unsqueeze(0)

            features = vision_net.forward(rgb_tensor, input_data).squeeze(0).detach().cpu()

            for idx in range(len(episode)):

                episode.precomputed_features[idx] = features

        print("I was able to precompute features for first frame only")

        # overwrite function
        self.get_image_data = self.get_precomputed_features
        self.get_depth_data = self.nothing # we would have used depth to compute features, don't need anymore



    def get_precomputed_features(self, log_name, idx, camera_num):
        episode = self.episodes[log_name]
        original_index = episode.get_entry(idx)["original_index"]
        return episode.precomputed_features[original_index]

    def get_precomputed_descriptor_images(self, log_name, idx, camera_num):
        episode = self.episodes[log_name]
        original_index = episode.get_entry(idx)["original_index"]
        return episode.precomputed_descriptor_images[original_index]


    def __len__(self):
        if self._use_only_first_index:
            print("should be test mode only. ONLY USING FIRST INDEX")
            return len(self.episodes.keys())
        else:
            return self._num_entries

