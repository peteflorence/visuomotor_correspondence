from __future__ import print_function

import os
import copy
import math
import yaml
import logging
import numpy as np
import random
from numpy import linalg as LA

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from spartan.utils import utils as spartan_utils

from imitation_agent.dataset.imitation_episode import ImitationEpisode
import imitation_agent.dataset.dataset_utils as dataset_utils
from imitation_agent.dataset.precompute_helper_dataset import PrecomputeHelperDataset
from imitation_agent.dataset.feature_saver import FeatureSaver



class ImitationEpisodeSequenceDataset(data.Dataset):

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
        self._noise_enabled = True
        self._length = 2000

        self.episodes = dict()
        self._num_entries = 0

        import time; start = time.time()
        print("starting to load logs...")

        # construct the ImitationEpisodeConfig for each log in our dataset
        for log_name in log_config['logs']:
            path_to_processed_dir = os.path.join(logs_dir_path, log_name, "processed")
            # print("path_to_processed_dir", path_to_processed_dir)
            assert os.path.isdir(path_to_processed_dir)

            episode_config = copy.copy(self._config)
            episode_config['path_to_processed_dir'] = path_to_processed_dir

            imitation_episode = self.construct_imitation_episode(episode_config, type="offline")

            self.episodes[log_name] = imitation_episode
            self._num_entries += len(imitation_episode)

        print(time.time() - start, " is seconds to load all logs")

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

    def get_full_pose_path_sliced(self, 
                                  episode, # type ImitationEpisode
                                  slice_index, # type int 
                                  ): # type -> torch.Tensor of shape L,D
        pose_path = []
        indices = []
        T_noise_list = []

        T_noise = self.sample_noise_transform()
        T_noise_list.append(T_noise)
        observations = self._observation_function(episode, 0, T_noise=T_noise)

        pose_path.append(observations)
        first = np.asarray(observations) * 1.0
        indices.append(0)

        counter = 0
        skip_initial = True
        prev = first


        for idx in range(slice_index, len(episode) - 1, self._config["num_slices"]):
            T_noise = self.sample_noise_transform()
            T_noise_list.append(T_noise)
            observations = torch.Tensor(self._observation_function(episode, idx, T_noise=T_noise))
            cur = np.asarray(observations)

            pose_path.append(observations)
            indices.append(idx)
            prev = cur

            # if (np.max(np.abs(cur - first)) < 0.001) and skip_initial:
            #     counter += 1
            # else:
            #     skip_initial = False
            #
            # if not skip_initial:
            #     if (np.max(np.abs(cur - prev)) > 0.001):
            #         # pose_path.append(observations)
            #         # indices.append(idx)
            #
            # prev = cur

        # print counter, "skipped!"
        # print "total:", len(observation_action_data)/self._config["num_slices"]
        # print "took :", len(pose_path)

        # what shape/type is pose_path
        pose_path = torch.stack(pose_path)
        pose_path = self.augment_state_with_noise(pose_path)
        return pose_path, indices, T_noise_list

    def augment_state_with_noise(self, state, sigma=None):
        if sigma is None:
            sigma = self._config["sigma_noise_augmentation"]  # this is in meters

        if self._config["observation"]["config"]["gripper"]["width"]:
            state[:,-1] += torch.randn_like(state[:,-1])*sigma

        return state


    def get_random_slice_index(self):
        return random.randint(0, self._config["num_slices"] - 1)

    def get_full_desired_path_sliced(self, 
                                     episode, # type ImitationEpisode
                                     indices, # type list of ints
                                     T_noise_list, # list of 4 x 4 homogeneous transforms
                                     ): # type -> torch.Tensor of shape L,A
        desired_path = []

        for idx, T_noise in zip(indices, T_noise_list):
            actions = self._action_function(episode, idx, T_noise=T_noise)
            desired_path.append(actions)

        desired_path = torch.FloatTensor(desired_path)
        return desired_path

    def get_full_image_path_sliced(self, log_name, indices):

        camera_num = self._config["camera_num"]

        img_path = torch.zeros(len(indices), 3, 480, 640)

        counter = 0
        for idx in indices:
            original_index = self.episodes[log_name].get_entry(idx)["original_index"]
            rgb = self.spartan_dataset.get_rgb_image_from_scene_name_and_idx_and_cam(log_name, original_index, camera_num)
            rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(rgb).unsqueeze(0)
            img_path[counter, :, : ,:] = rgb_tensor
            counter += 1

        return img_path

    def replicate_some_indices(self, pose_path, desired_path, indices):
        pose_path = pose_path.numpy()
        desired_path = desired_path.numpy()

        assert len(pose_path) == len(desired_path) == len(indices)

        num_indices = len(pose_path)
        num_random_indices_to_replicate = int(num_indices / 10)
        rand_indices = np.sort(np.floor(np.random.rand(num_random_indices_to_replicate, 1) * num_indices).astype(int))

        increase = 0
        for i in rand_indices:
            i += increase
            pose_path = np.insert(pose_path, i, pose_path[i].squeeze(), axis=0)
            desired_path = np.insert(desired_path, i, desired_path[i].squeeze(), axis=0)
            indices.insert(i, indices[i[0]])
            increase += 1

        return torch.from_numpy(pose_path), torch.from_numpy(desired_path), indices

    def delete_some_indices(self, pose_path, desired_path, indices):
        pose_path = pose_path.numpy()
        desired_path = desired_path.numpy()

        assert len(pose_path) == len(desired_path) == len(indices)

        num_indices = len(pose_path)
        num_random_indices_to_replicate = int(num_indices / 10)
        rand_indices = np.unique(
            np.sort(np.floor(np.random.rand(num_random_indices_to_replicate, 1) * num_indices).astype(int)))

        decrease = 0
        for i in rand_indices:
            i -= decrease
            pose_path = np.delete(pose_path, i, axis=0)
            desired_path = np.delete(desired_path, i, axis=0)
            del indices[i]
            decrease += 1

        assert len(pose_path) == len(desired_path) == len(indices)

        return torch.from_numpy(pose_path), torch.from_numpy(desired_path), indices


    def shift_augmentation(self, pose_path, desired_path):
        assert self._config["observation"]["config"]["translation"]["x"] and self._config["observation"]["config"]["translation"]["y"]
        assert self._config["action"]["config"]["translation"]["x"] and self._config["action"]["config"]["translation"]["y"]
        assert self._config["use_gt_object_pose"] and not self._config["project_pose_into_camera"]
        
        x_shift = torch.randn(1) * self._config["shift_augmentation_sigma"]
        y_shift = torch.randn(1) * self._config["shift_augmentation_sigma"]

        pose_path[:,0] += x_shift
        pose_path[:,1] += y_shift
        pose_path[:,-3] += x_shift
        pose_path[:,-2] += y_shift

        desired_path[:,0] += x_shift
        desired_path[:,1] += y_shift

        return pose_path, desired_path

    def __getitem__(self, index):
        """
        The method through which the dataset is accessed for training.
        """

        log_name = self.get_random_log_name()
        episode = self.episodes[log_name]

        slice_index = self.get_random_slice_index()

        # return whole trajectories but sliced
        pose_path, indices, T_noise_list = self.get_full_pose_path_sliced(episode, slice_index)
        desired_path = self.get_full_desired_path_sliced(episode, indices, T_noise_list)

        if self._config["temporal_augmentation"]:
            pose_path, desired_path, indices = self.replicate_some_indices(pose_path, desired_path, indices)
            pose_path, desired_path, indices = self.delete_some_indices(pose_path, desired_path, indices)

        if self._config["shift_augmentation"]:
            pose_path, desired_path = self.shift_augmentation(pose_path, desired_path)

        if not self._config["use_vision"]:
            images_path = torch.Tensor([])
        else:
            images_path = self.get_full_image_path_sliced(log_name, indices)

        #depth_path, camera_to_world_path, K_path = self.get_3D_data_path_sliced(episode, log_name, idx)


        return {"observations": pose_path,
                "images": images_path,
                "actions": desired_path
                }

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

        feature_saver = FeatureSaver()
        loaded = feature_saver.load_if_already_have("features", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)

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
                    episode.precomputed_features[int(idx)] = feature_batch[counter]
                    counter += 1


            feature_saver.save("features", self._config, vision_net._reference_descriptor_vec, self.episodes, train_or_test)

        print("I was able to precompute all features")
        print("Took me", time.time() - start, "seconds")

        # overwrite function
        self.get_full_image_path_sliced = self.get_precomputed_features_sliced
        
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
        self.get_full_image_path_sliced = self.get_precomputed_descriptor_images
        if not self._config["use_depth"]:
            self.get_depth_data = self.nothing

    def nothing(self, log_name, idx, camera_num):
        return torch.Tensor([])


    def get_precomputed_features_sliced(self, log_name, indices):
        episode = self.episodes[log_name]

        feature_size = episode.precomputed_features[0].shape[0]

        feature_path = torch.zeros(len(indices), feature_size)

        counter = 0
        for idx in indices:
            original_index = episode.get_entry(idx)["original_index"]
            feature_path[counter, :] = episode.precomputed_features[original_index]
            counter += 1

        #feature_path = self.augment_state_with_noise(feature_path, sigma = 0.01)

        return feature_path
    
    def get_precomputed_descriptor_images(self, log_name, indices):
        episode = self.episodes[log_name]
        
        D, H, W = episode.precomputed_descriptor_images[0].shape

        # I'm expecting these to be...
        assert D > 2 and D < 64
        assert H == 60
        assert W == 80

        # N, 3, 60, 80
        d_image_path = torch.zeros(len(indices),D,H,W)

        counter = 0
        for idx in indices:
            original_index = episode.get_entry(idx)["original_index"]
            d_image_path[counter] = episode.precomputed_descriptor_images[original_index]
            counter += 1
            
        return d_image_path


    def set_length(self, length):
        self._length = length

    def __len__(self):
        return self._length

