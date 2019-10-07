from __future__ import print_function

import os
import numpy as np
import time

# pytorch
import torch

# spartan
import spartan.utils.utils as spartan_utils

# imitation_agent
from imitation_agent.dataset.imitation_episode_sequence_dataset import ImitationEpisodeSequenceDataset
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

import matplotlib.pyplot as plt

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
# logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_then_flip_0716")
#
# logs_config_yaml = os.path.join(spartan_source_dir, "modules/imitation_agent/config/task/move_to_box_then_flip_0716.yaml")
# logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)
#
#
# config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence.yaml")
# config = spartan_utils.getDictFromYamlFilename(config_yaml)

logs_dir_path = os.path.join(spartan_utils.getSpartanSourceDir(), 'sandbox/push_box')
logs_config_yaml = os.path.join(spartan_source_dir,
                                "modules/imitation_agent/config/task/push_box_small.yaml")


logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


# config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence.yaml")
config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence_push_box.yaml")
config = spartan_utils.getDictFromYamlFilename(config_yaml)


# print("SETTING NO NOISE")
#config["sigma_noise_augmentation"] = 0.0



def construct_dataset():
    # obs_function = ObservationFunctionFactory.observation_from_config(config)
    obs_function = ObservationFunctionFactory.ee_position_history_observation(config)
    action_function = ActionFunctionFactory.action_from_config(config)
    dataset = ImitationEpisodeSequenceDataset(logs_dir_path, logs_config, config,
                                      action_function=action_function,
                                      observation_function=obs_function)
    return dataset

dataset = construct_dataset()


def plot_observations(observations, actions):
    print(observations.shape)
    print(actions.shape)
    x_hist = observations[:,0].numpy()
    y_hist = observations[:,1].numpy()
    z_hist = observations[:,2].numpy()

    x_actions = actions[:,0].numpy()
    y_actions = actions[:,1].numpy()
    z_actions = actions[:,2].numpy()


    t = np.arange(0,observations.shape[0],1)

    plt.plot(t, x_hist, color="red",   ls=":",  label="x")
    plt.plot(t, y_hist, color="blue",  ls=":",  label="y")
    plt.plot(t, z_hist, color="green", ls=":",  label="z")

    plt.plot(t,x_actions,     color="red",   ls="-",  label="x_d")
    plt.plot(t,y_actions,     color="blue",  ls="-",  label="y_d")
    plt.plot(t,z_actions,     color="green", ls="-",  label="z_d")
    
    quat_w = observations[:,3].numpy()
    quat_x = observations[:,4].numpy()
    quat_y = observations[:,5].numpy()
    quat_z = observations[:,6].numpy()

def plot_dynamic_pose(observations, actions):
    print(observations.shape)
    print(actions.shape)
    # x_hist = observations[:,-7].numpy()
    # y_hist = observations[:,-6].numpy()
    # z_hist = observations[:,-5].numpy()
    grip = observations[:,6].numpy()

    w = observations[:,-7].numpy()
    x = observations[:,-6].numpy()
    y = observations[:,-5].numpy()
    z = observations[:,-4].numpy()

    t = np.arange(0,observations.shape[0],1)

    # plt.plot(t, x_hist, color="red", label="x")
    # plt.plot(t, y_hist, color="green", label="y")
    # plt.plot(t, z_hist, color="blue", label="z")
    plt.plot(t, grip,   color="red",  label="grip")
    plt.plot(t, w,      color="orange", label="w")
    plt.plot(t, x,      color="purple", label="x")
    plt.plot(t, y,      color="black", label="y")
    plt.plot(t, z,      color="cyan", label="z")
    

def plot_actions(observations, actions):
    print(observations.shape)
    print(actions.shape)
    
    t = np.arange(0,observations.shape[0],1)
    
    quat_w = observations[:,3].numpy()
    quat_x = observations[:,4].numpy()
    quat_y = observations[:,5].numpy()
    quat_z = observations[:,6].numpy()

    plt.plot(t, quat_w, color="orange", ls=":",  label="q_w")
    plt.plot(t, quat_x, color="black",  ls=":",  label="q_x")
    plt.plot(t, quat_y, color="purple", ls=":",  label="q_y")
    plt.plot(t, quat_z, color="cyan",   ls=":",  label="q_z")

    roll_d =  actions[:,3].numpy() 
    pitch_d = actions[:,4].numpy() 
    yaw_d =   actions[:,5].numpy() 

    plt.plot(t, roll_d,  color="red",    ls="-",  label="roll_d")
    plt.plot(t, pitch_d, color="green",  ls="-",  label="pitch_d")
    plt.plot(t, yaw_d,   color="blue",   ls="-",  label="yaw_d")
    
def plot_dynamic_pose_angle_axis(observations, actions):
    x = observations[:, -3].numpy()
    y = observations[:, -2].numpy()
    z = observations[:, -1].numpy()

    t = np.arange(0, observations.shape[0], 1)
    plt.plot(t, x, color="red", label="x")
    plt.plot(t, y, color="green", label="y")
    plt.plot(t, z, color="blue", label="z")

def plot_dynamic_pose_pos(observations, actions):
    x = observations[:, -6].numpy()
    y = observations[:, -5].numpy()
    z = observations[:, -4].numpy()

    t = np.arange(0, observations.shape[0], 1)
    plt.plot(t, x, color="red", label="x")
    plt.plot(t, y, color="green", label="y")
    plt.plot(t, z, color="blue", label="z")



def plot_actions_test(observations, actions):
    x = actions[:, -3].numpy()
    y = actions[:, -2].numpy()
    z = actions[:, -1].numpy()

    t = np.arange(0, observations.shape[0], 1)
    plt.plot(t, x, color="red", label="x")
    plt.plot(t, y, color="green", label="y")
    plt.plot(t, z, color="blue", label="z")



num_overlay = 30 


# # plot observations
# for i in range(num_overlay):
#     data = dataset[i]
#     plot_dynamic_pose(data["observations"], data["actions"])
#
# plt.legend()
# plt.show()


# plot actions
# for i in range(num_overlay):
#     data = dataset[i]
#     plot_actions(data["observations"], data["actions"])
#
# plt.title("actions")
# plt.legend()
# plt.show()

# plot observations
for i in range(num_overlay):
    data = dataset[i]
    plot_dynamic_pose_pos(data["observations"], data["actions"])

plt.legend()
plt.title('dynamic pose pos')
plt.show()


# plot observations
for i in range(num_overlay):
    data = dataset[i]
    plot_dynamic_pose_angle_axis(data["observations"], data["actions"])

plt.legend()
plt.title('dynamic pose angle_axis')
plt.show()

# plot actions
for i in range(num_overlay):
    data = dataset[i]
    plot_actions_test(data["observations"], data["actions"])

plt.legend()
plt.title('actions')
plt.show()


# plot time
counter = 0
for log_name in dataset.episodes.keys():
    counter += 1
    print(log_name)
    episode = dataset.episodes[log_name]
    indices = episode.state_dict.keys()
    times = []
    for index in indices:
        times.append(episode.state_dict[index]["observations"]["timestamp"])
    times = np.asarray(times)
    times = times - times[0]
    plt.plot(np.asarray(indices), times)
    
    # ocassionally plot
    # if counter % 10 == 9:
    #     plt.xlabel('index')
    #     plt.ylabel('timestamp')
    #     plt.show()



plt.xlabel('index')
plt.ylabel('timestamp')
plt.show()
