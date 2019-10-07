from __future__ import print_function

# system
import os

# ROS
import rospy

# spartan
import spartan.utils.utils as spartan_utils

# imitation_agent
from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent
from imitation_agent.deploy.mlp_ee_position_agent import MLPPositionAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

# torch
import torch

import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.set_cuda_visible_devices([0])

def get_most_recent_network():
    base = "/home/peteflo/data/pdc/imitation/trained_models/mlp_position"
    most_recent_dir = sorted(os.listdir(base))[-1]
    longest_chkpt = sorted(os.listdir(os.path.join(base,most_recent_dir)))[-2] # "tensorboard is last"
    return os.path.join(base,most_recent_dir,longest_chkpt)

def load_network():
    #network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/mlp_position/2019-07-13-21-09-41/iteration-101480.pth"
    network_chkpt = get_most_recent_network()

    print("loading net:", network_chkpt)

    network = torch.load(network_chkpt)
    return network

def construct_imitation_episode(config):
    action_function = ActionFunctionFactory.action_from_config(config)
    
    observation_function = ObservationFunctionFactory.get_function(config)
    #observation_function = ObservationFunctionFactory.observation_from_config(config)
    imitation_episode = ImitationEpisode(config,
                                         type="online",
                                         action_function=action_function,
                                         observation_function=observation_function)

    return action_function, observation_function, imitation_episode

if __name__ == "__main__":
    rospy.init_node("task_space_control_agent")

    network = load_network()
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()

    config = network._policy_net._config

    print(config["use_gt_object_pose"], "is use_gt_object_pose")
    
    action_function, observation_function, episode = construct_imitation_episode(config)


    task_space_control_agent = ROSTaskSpaceControlAgent()
    task_space_control_agent.move_to_starting_position()

    mlp_position_agent = MLPPositionAgent(network, observation_function, episode, config)
    task_space_control_agent.control_function = mlp_position_agent.compute_control_action
    task_space_control_agent.control_rate = 25

    # run the node
    task_space_control_agent.run()
