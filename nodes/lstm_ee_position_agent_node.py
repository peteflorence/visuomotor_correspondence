from __future__ import print_function

# system
import os
import time

# ROS
import rospy

# spartan
import spartan.utils.utils as spartan_utils

# imitation_agent
from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent
from imitation_agent.deploy.lstm_ee_position_agent import LSTMPositionAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

# torch
import torch


#TASK_NAME = "SHOE_FLIP"
#TASK_NAME = "MANY_SHOE_FLIP"
#TASK_NAME = "PUSH_BOX"
#TASK_NAME = "HANG_HAT"
TASK_NAME = "GRAB_PLATE"

REAL_ROBOT_PROCESSED_DIR = "/home/peteflo/data/pdc/imitation/real_push_box/2019-08-06-21-06-13/processed"

def get_most_recent_network():
    base = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/trained_models/lstm_standard")
    most_recent_dir = sorted(os.listdir(base))[-1]
    longest_chkpt = sorted(os.listdir(os.path.join(base,most_recent_dir)))[-2] # "tensorboard is last"
    return os.path.join(base,most_recent_dir,longest_chkpt)

def load_network():
    #network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-06-25-14-35-15/iter-5100.pth"

    # flip shoe
    if TASK_NAME == "SHOE_FLIP":
        network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-08-07-01-55-39/iter-200001.pth"
    elif TASK_NAME == "MANY_SHOE_FLIP":
        network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-08-09-00-45-18/iter-200001.pth"
    elif TASK_NAME == "PUSH_BOX":
        network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-08-07-05-33-17/iter-200000.pth"
    elif TASK_NAME == "HANG_HAT":
        network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-08-09-05-32-57/iter-200001.pth"
    elif TASK_NAME == "GRAB_PLATE":
        network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-08-10-19-29-55/iter-231546.pth"
    
    # trained to do 20 boxes, taking in ground truth pose
    #network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-06-26-03-52-23/iter-032249.pth"

    # trained to do 2 boxes, with vision (only 1 ref descriptor)
    #network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-06-26-19-49-43/iter-002725.pth"

    # trained to do 20 boxes, with vision (only 1 ref descriptor)
    #network_chkpt = "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-06-27-00-02-06/iter-005000.pth"

    #network_chkpt = get_most_recent_network()

    print("loading net:", network_chkpt)

    network = torch.load(network_chkpt)
    return network

def construct_imitation_episode(config):
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()

    action_function = ActionFunctionFactory.action_from_config(config)

    # observation_function = ObservationFunctionFactory.observation_from_config(config)
    observation_function = ObservationFunctionFactory.get_function(config)

    imitation_episode = ImitationEpisode(config,
                                         type="online",
                                         action_function=action_function,
                                         observation_function=observation_function,
                                         processed_dir=REAL_ROBOT_PROCESSED_DIR)

    return action_function, observation_function, imitation_episode



def shoe_flip_reset():
    above_table_pre_grasp = task_space_control_agent._stored_poses_dict["Grasping"]["above_table_pre_grasp"]
    task_space_control_agent._robot_service.moveToJointPosition(above_table_pre_grasp, maxJointDegreesPerSecond=40, timeout=5)
    import time; time.sleep(0.3)


def push_box_reset():
    """
    Abusing global scopes but this works.
    """
    above_table_pre_grasp = task_space_control_agent._stored_poses_dict["Grasping"]["above_table_pre_grasp"]
    task_space_control_agent._robot_service.moveToJointPosition(above_table_pre_grasp, maxJointDegreesPerSecond=30, timeout=5)

    # close gripper
    gripper_goal_pos = 0.0
    #time.sleep(1.0)
    task_space_control_agent._gripper_driver.sendGripperCommand(gripper_goal_pos, speed=0.2, timeout=0.01)
    #time.sleep(1.0)

    # move to start position
    start_pose = task_space_control_agent._stored_poses_dict["imitation"]["push_box_start_REAL"]
    success = task_space_control_agent._robot_service.moveToJointPosition(start_pose, maxJointDegreesPerSecond=20, timeout=5)
    print("Moved to push box start")
    print("Sleeping so you can put box in position...")
    time.sleep(0.5)

def hang_hat_reset():
    above_table_pre_grasp = task_space_control_agent._stored_poses_dict["Grasping"]["above_table_pre_grasp"]
    task_space_control_agent._robot_service.moveToJointPosition(above_table_pre_grasp, maxJointDegreesPerSecond=70, timeout=5)
    
    task_space_control_agent._gripper_driver.sendGripperCommand(0.1, speed=0.1, timeout=0.01)
    print("sent open goal to gripper")

    start_pose = task_space_control_agent._stored_poses_dict["imitation"]["hat_task_start"]
    success = task_space_control_agent._robot_service.moveToJointPosition(start_pose, maxJointDegreesPerSecond=70, timeout=5)
    print("ready to hang the hat!")

def grab_plate_reset():
    print("sleeping for a second then opening!")
    time.sleep(1.5)
    task_space_control_agent._gripper_driver.sendGripperCommand(0.1, speed=0.2, timeout=0.01)
    print("get out of the way!")
    time.sleep(0.5)
    above_table_pre_grasp = task_space_control_agent._stored_poses_dict["Grasping"]["above_table_pre_grasp"]
    task_space_control_agent._robot_service.moveToJointPosition(above_table_pre_grasp, maxJointDegreesPerSecond=20, timeout=5)
    lstm_position_agent.mouse_manager.release_mouse_focus()
    key = raw_input("'g' to go again, anything else quits")
    if key != "g":
        import sys; sys.exit(0)
    print("going again!")
    lstm_position_agent.mouse_manager.grab_mouse_focus()



if __name__ == "__main__":
    import dense_correspondence_manipulation.utils.utils as pdc_utils
    pdc_utils.set_cuda_visible_devices([0])

    rospy.init_node("task_space_control_agent")

    network = load_network()
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()

    config = network._policy_net._config
    config["real_robot"] = True

    print(config["model"]["config"]["num_ref_descriptors"])
    print("Is num_ref_descriptors")
    
    action_function, observation_function, episode = construct_imitation_episode(config)


    task_space_control_agent = None
    

    if TASK_NAME == "SHOE_FLIP" or TASK_NAME == "MANY_SHOE_FLIP":  
        config["software_safety"] = dict()
        config["software_safety"]["translation_threshold"] = 0.02
        config["software_safety"]["rotation_threshold_degrees"] = 4
        config["deploy_image_topic"] = "/camera_d415_02/color/image_raw"
        task_space_control_agent = ROSTaskSpaceControlAgent(config=config)

        # Note that starting position is not the same as reset.
        # On the start we want to open/close gripper.
        # But not on resets.
        task_space_control_agent.move_to_starting_position()
        reset_function = shoe_flip_reset

    elif TASK_NAME == "PUSH_BOX":
        config["gripper_width_default"] = 0.0
        config["deploy_image_topic"] = "/camera_d415_01/color/image_raw"
        task_space_control_agent = ROSTaskSpaceControlAgent(config=config)

        # Same start as reset.
        push_box_reset()
        reset_function = push_box_reset

    elif TASK_NAME == "HANG_HAT":
        config["software_safety"] = dict()
        config["software_safety"]["translation_threshold"] = 0.02
        config["software_safety"]["rotation_threshold_degrees"] = 4
        config["deploy_image_topic"] = "/camera_d415_01/color/image_raw"
        task_space_control_agent = ROSTaskSpaceControlAgent(config=config)
        # Same start as reset.
        hang_hat_reset()
        reset_function = hang_hat_reset

    elif TASK_NAME == "GRAB_PLATE":
        config["software_safety"] = dict()
        config["software_safety"]["translation_threshold"] = 0.02
        config["software_safety"]["rotation_threshold_degrees"] = 4
        config["deploy_image_topic"] = "/camera_d415_02/color/image_raw"
        task_space_control_agent = ROSTaskSpaceControlAgent(config=config)
        
        task_space_control_agent.move_to_starting_position()
        reset_function = grab_plate_reset


    lstm_position_agent = LSTMPositionAgent(network, observation_function, episode, config, reset_function=reset_function)
    task_space_control_agent.control_function = lstm_position_agent.compute_control_action
    task_space_control_agent.control_rate = 100



    # run the node
    task_space_control_agent.run()
