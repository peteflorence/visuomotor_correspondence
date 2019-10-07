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

import spartan.utils.utils as spartan_utils

from imitation_agent.dataset.imitation_episode import ImitationEpisode
import imitation_agent.dataset.dataset_utils as dataset_utils


class PrecomputeHelperDataset(data.Dataset):

    def __init__(self, spartan_dataset, episodes, camera_num, imitation_episode_dataset):
        self.spartan_dataset = spartan_dataset
        self.episodes = episodes
        self.camera_num = camera_num
        self.imitation_episode_dataset = imitation_episode_dataset

        # add tuples of (log_name, index)
        self.list_of_log_name_index_tuples = []

        for log_name in self.episodes.keys():
            episode = self.episodes[log_name]
            for index in range(episode.get_unfiltered_length()):
                log_name_index_tuple = (log_name, index)
                self.list_of_log_name_index_tuples.append(log_name_index_tuple)

    def __len__(self):
        return len(self.list_of_log_name_index_tuples)

    def __getitem__(self, index, # type int
                            ): 

        log_name, idx = self.list_of_log_name_index_tuples[index]

        rgb = self.spartan_dataset.get_rgb_image_from_scene_name_and_idx_and_cam(log_name, idx, self.camera_num)
        rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(rgb)

        input_data = dict()
        input_data["rgb_tensor"] = rgb_tensor
        input_data["log_name"] = log_name
        input_data["idx"] = idx

        if self.imitation_episode_dataset._config["use_depth"]:
            depth_data = self.imitation_episode_dataset.get_depth_data(log_name, idx, self.camera_num)
            episode = self.imitation_episode_dataset.episodes[log_name]

            camera_to_world = torch.from_numpy(episode.get_camera_pose_matrix(self.camera_num))
            K = torch.from_numpy(episode.get_K_matrix(self.camera_num))
        
            input_data['depth'] = depth_data
            input_data['camera_to_world'] = camera_to_world
            input_data['K'] = K

        return input_data

