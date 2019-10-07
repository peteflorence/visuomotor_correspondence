from __future__ import print_function

# system
import numpy as np
import sys
import time

# torch
import torch

# ros
import rospy
import tf2_ros

# spartan
import spartan.utils.ros_utils as ros_utils
from robot_msgs.msg import CartesianGoalPoint
import spartan.utils.transformations as transformations
from spartan.utils import constants

# imitation_agent
import imitation_agent.config.parameters as parameters
from imitation_agent.deploy.utils import make_cartesian_gains_msg, tf_matrix_from_pose 
from imitation_agent.deploy import ros_imitation_parser
from imitation_agent.loss_functions.loss_functions import parse_mdn_params
from imitation_agent.utils import utils as imitation_agent_utils

import imitation_agent.dataset.dataset_utils as dataset_utils

from imitation_agent.deploy.software_safety import SoftwareSafety

class MLPPositionAgent(object):

    def __init__(self, network, observation_function, imitation_episode, config):
        self.network = network
        self.network.eval()
        self.network.cuda()

        self._config = config
        self._unsafe = False

        self._observation_function = observation_function
        self.imitation_episode = imitation_episode

        self.frame_name = "iiwa_link_ee" # end effector frame name
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.cache_reference_pose()
        self.software_safety = SoftwareSafety(config)
        self.software_safety.set_initial_goal(self.latest_interp_goal["T_W_E"])
        self.SIMULATION_SAFETY_CONFIG = False

        self.spartan_dataset = dataset_utils.build_spartan_dataset("")

        self._T_W_C_default = imitation_agent_utils.get_T_W_Cnominal_from_config(config)
        self._gripper_width_default = imitation_agent_utils.get_gripper_width_default_from_config(config)



    def cache_reference_pose(self):
        """
        This is the pose that all logs start from.
        """
        print("I better be home now!")
        for i in range(10):
            if i == 9:
                print("Couldn't find robot pose")
                sys.exit(0)
            try:
                ee_pose_above_table = ros_utils.poseFromROSTransformMsg(
                    self.tfBuffer.lookup_transform("base", self.frame_name, rospy.Time()).transform)
                self.ee_tf_above_table = tf_matrix_from_pose(ee_pose_above_table)
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Trouble looking up robot pose ...")
                time.sleep(0.1)

        self.latest_interp_goal = dict()
        self.latest_interp_goal["T_W_E"] = self.ee_tf_above_table

    def run_model(self, agent): # no return
        """
        Run the network.
        """
        data_list = agent.data_list
        self.imitation_episode.set_state_dict(data_list)

        # extract RGB image
        image_data = agent.get_latest_images()
        rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(image_data['rgb_image'])

        idx = len(data_list) - 1
        obs = self._observation_function(self.imitation_episode, idx, T_noise=np.eye(4))
        obs = torch.tensor(obs)
        obs = obs.type(torch.FloatTensor).cuda()
        obs = obs.unsqueeze(0) # shape (1, obs_dim)

        with torch.no_grad():
            # push it through the network
            input_package = dict()
            input_package["observation"] = obs
            input_package["image"] = rgb_tensor.unsqueeze(0).cuda()
            actions = self.network.forward(input_package)

      
        actions = actions.cpu().squeeze().detach()

        T_W_C = self.imitation_episode.get_entry(idx)['T_W_C']
        action_dict = ros_imitation_parser.compute_action(actions, self._config, T_W_C=T_W_C,
                                                          T_W_C_default=self._T_W_C_default,
                                                          gripper_width_default=self._gripper_width_default)


        # self.actions_goal = actions
        self.actions_goal = action_dict

    def should_run_new_model(self): # type -> bool
        """
        Based on downsample rate, should we run the model?
        """
        return True


    def compute_control_action(self, agent, #type ROSTaskSpaceControlAgent
                               ): # type -> (robot_msgs.msg.CartesianGoalPoint, float)

        if self.should_run_new_model():
            self.run_model(agent)

        goal = self.actions_goal
        msg = CartesianGoalPoint()
        msg.ee_frame_id = parameters.ee_frame_id
        msg.use_end_effector_velocity_mode = False
        msg.gain = make_cartesian_gains_msg(5., 10.)  # set the gains

        # extract position and quaternion for the EE frame
        xyz = goal["T_W_E"][:3, 3]
        quat = transformations.quaternion_from_matrix(goal["T_W_E"])

        msg.xyz_point.header.frame_id = "world"
        msg.xyz_point.point.x = xyz[0]
        msg.xyz_point.point.y = xyz[1]
        msg.xyz_point.point.z = xyz[2]

        msg.quaternion.w = quat[0]
        msg.quaternion.x = quat[1]
        msg.quaternion.y = quat[2]
        msg.quaternion.z = quat[3]

        if self.SIMULATION_SAFETY_CONFIG:
            self._unsafe = self.software_safety.determine_if_unsafe(msg)
        else:
            self.software_safety.sys_exit_if_not_safe(msg)

        return msg, goal['gripper_width']

    def set_T_W_C_default(self,
                          T,  # homogeneous transform, np.array shape [4,4]
                          ):
        """
        Sets the default T_W_cmd used for parsing the action
        :param T:
        :type T:
        :return:
        :rtype:
        """
        self._T_W_C_default = T

    def set_gripper_width_default(self, val):
        self._gripper_width_default = val




