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
from imitation_agent.dataset.imitation_episode_dataset import ImitationEpisodeDataset
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

import matplotlib.pyplot as plt

"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_careful")

logs_config_yaml = os.path.join(spartan_source_dir, "modules/imitation_agent/config/task/move_to_box_0710.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


config_yaml = os.path.join(imitation_src_dir, "config", "model", "mlp_stateless_position.yaml")
config = spartan_utils.getDictFromYamlFilename(config_yaml)

config["use_vision"] = False
config["model"]["config"]["num_inputs"] = 13
config["model"]["vision_net"] = "none"
config["use_gt_object_pose"] = True
config["project_pose_into_camera"] = False


# print("SETTING NO NOISE")
#config["sigma_noise_augmentation"] = 0.0



def construct_dataset():
    obs_function = ObservationFunctionFactory.observation_from_config(config)
    action_function = ActionFunctionFactory.action_from_config(config)
    dataset = ImitationEpisodeDataset(logs_dir_path, logs_config, config,
                                      action_function=action_function,
                                      observation_function=obs_function)
    return dataset

dataset = construct_dataset()
dataset.set_use_only_first_index()

for i in range(100):
    print(dataset[i])


def plot_x_object_correlation(observation):
    x_hist = observation[0].numpy()
    object_pose_x = observation[10].numpy()
    plt.scatter(x_hist, object_pose_x)
    return

def plot_y_object_correlation(observation):
    y_hist = observation[1].numpy()
    object_pose_y = observation[11].numpy()
    plt.scatter(y_hist, object_pose_y)
    return


def plot_y_action_correlation(observation, action):
    y_hist = observation[1].numpy()
    y_action = action[1].numpy()
    plt.scatter(y_hist, y_action)
    return

def minimize_mse_y(dataset):
    y_actions = []
    for i in range(200):
        action = dataset[i]["action"]
        y_action = float(action[1])
        y_actions.append(y_action)
    print(y_actions)
    print(len(y_actions))
    y_actions = torch.tensor(y_actions)
    pred = torch.mean(y_actions)
    print("mean is", pred)
    mse = ((y_actions - pred)**2).mean()
    print("mse is", mse)


num_overlay = 100 


# plot observations
for i in range(num_overlay):
    data = dataset[i]
    plot_x_object_correlation(data["observation"])

plt.legend()
plt.xlabel("first_measured_x")
plt.ylabel("gt_object_pose_x")
plt.xlim(0.5105,0.51075)
#plt.savefig('foo3.pdf')
plt.show()

for i in range(num_overlay):
    data = dataset[i]
    plot_y_object_correlation(data["observation"])

plt.legend()
plt.xlabel("first_measured_y")
plt.ylabel("gt_object_pose_y")
plt.xlim(0.015,0.016)
#plt.savefig('foo3.pdf')
plt.show()

for i in range(num_overlay):
    data = dataset[i]
    plot_y_action_correlation(data["observation"], data["action"])

plt.legend()
plt.xlabel("first_measured_y")
plt.ylabel("first_action_y")
plt.xlim(0.015,0.016)
#plt.savefig('foo3.pdf')
plt.show()

dataset.set_force_state_same()
for i in range(num_overlay):
    data = dataset[i]
    plot_y_action_correlation(data["observation"], data["action"])

plt.legend()
plt.xlim(0.015,0.016)
plt.xlabel("first_measured_y_forced_same")
plt.ylabel("first_action_y")
#plt.savefig('foo3.pdf')
plt.show()


minimize_mse_y(dataset)


sys.exit(0)

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
