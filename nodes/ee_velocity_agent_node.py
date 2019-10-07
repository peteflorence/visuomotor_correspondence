# system
import os

# ROS
import rospy

# spartan
import spartan.utils.utils as spartan_utils

# imitation_agent
from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent
from imitation_agent.deploy.ee_velocity_agent import EEVelocityAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

# torch
import torch

# another stable one is 2019-06-18-20-24-02

def load_network():
    # network_chkpt = "/home/manuelli/data/pdc/imitation/trained_models/mlp_ee_vel/2019-06-17-18-53-48/epoch-999.pth"

    # network_chkpt = "/home/manuelli/data/pdc/imitation/trained_models/mlp_ee_vel/2019-06-20-20-36-36/epoch-999.pth"

    # network_chkpt = "/home/manuelli/data/pdc/imitation/trained_models/mlp_ee_vel/2019-06-20-21-22-58/epoch-999.pth"

    # 2019-06-24-18-51-59
    network_chkpt = "/home/manuelli/data/pdc/imitation/trained_models/mlp_ee_vel/2019-06-24-19-05-54/epoch-999.pth"
    network = torch.load(network_chkpt)
    return network

def construct_imitation_episode(config):
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()
    logs_dir_path = os.path.join(data_dir, "pdc/imitation/logs")

    action_function = ActionFunctionFactory.get_function(config)

    observation_function = ObservationFunctionFactory.get_function(config)

    imitation_episode = ImitationEpisode(config,
                                         type="online",
                                         action_function=action_function,
                                         observation_function=observation_function)

    return action_function, observation_function, imitation_episode

if __name__ == "__main__":
    rospy.init_node("task_space_control_agent")

    network = load_network()
    action_function, observation_function, episode = construct_imitation_episode(config)
    ee_velocity_agent = EEVelocityAgent(network, observation_function, episode)

    task_space_control_agent = ROSTaskSpaceControlAgent()
    task_space_control_agent.control_function = ee_velocity_agent.compute_control_action

    # run the node
    task_space_control_agent.move_to_starting_position()
    task_space_control_agent.run()
