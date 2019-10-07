from __future__ import print_function

# system
import numpy as np
import sys
import time

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
# torch
import torch

# ros
import rospy
import tf2_ros

# spartan
import spartan.utils.ros_utils as ros_utils
from spartan.utils import constants
from robot_msgs.msg import CartesianGoalPoint
import spartan.utils.transformations as transformations

# imitation_agent
import imitation_agent.config.parameters as parameters
from imitation_agent.deploy.utils import make_cartesian_gains_msg, tf_matrix_from_pose 
from imitation_agent.deploy import ros_imitation_parser
from imitation_agent.loss_functions.loss_functions import parse_mdn_params
from imitation_agent.utils import utils as imitation_agent_utils

import imitation_agent.dataset.dataset_utils as dataset_utils

from imitation_agent.deploy.software_safety import SoftwareSafety


USE_DEPLOY_DEMO = False
BAG_DATA = False


LSTM_TRAIN_RATE = 5.0 # in Hz
TASK_SPACE_STREAMING_TOPIC_RATE = 100.0 # in Hz
# TASK_SPACE_STREAMING_TOPIC_RATE = 30.0 # Hz
DOWNSAMPLE_RATE = TASK_SPACE_STREAMING_TOPIC_RATE/LSTM_TRAIN_RATE

if USE_DEPLOY_DEMO:
    # ---> needed for deploy demo
    import spartan.utils.utils as spartan_utils
    import sys
    import os
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    teleop_dir = os.path.join(spartan_source_dir, "src/catkin_projects/simple_teleop")
    sys.path.append(teleop_dir)
    from teleop_mouse_manager import TeleopMouseManager
    import robot_msgs.msg
    sys.path.append(os.path.join(spartan_source_dir,"src/catkin_projects/imitation_tools/scripts"))
    from capture_imitation_data_client import start_bagging_imitation_data_client, stop_bagging_imitation_data_client
    # ---> needed for deploy demo


class LSTMPositionAgent(object):

    def __init__(self, network, observation_function, imitation_episode, config, reset_function=None):
        self._reset_function = reset_function
        self._gripper_width_default = imitation_agent_utils.get_gripper_width_default_from_config(config)

        self.network = network
        #self.network._policy_net._non_vision_index_start = 0
        self.network.eval()
        self.network.cuda()
        self.network.set_states_initial()

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

        self.downsample_rate = DOWNSAMPLE_RATE
        self.first_time_running_model = True

        self.spartan_dataset = dataset_utils.build_spartan_dataset("")

        if BAG_DATA:
            bagging_started = start_bagging_imitation_data_client()
            if not bagging_started:
                raise ValueError("bagging failed to start")

            rospy.sleep(2.0)

        if USE_DEPLOY_DEMO:
            self.mouse_manager = TeleopMouseManager()

        self._T_W_C_default = imitation_agent_utils.get_T_W_Cnominal_from_config(config)
        self._T_W_E_init_command_sent = False


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
                print("Troubling looking up robot pose...")
                time.sleep(0.1)

        self.latest_interp_goal = dict()
        self.latest_interp_goal["T_W_E"] = self.ee_tf_above_table
        T_W_E = np.copy(self.ee_tf_above_table)
        self.initial_pose = {"T_W_E": T_W_E, "gripper_width": self._gripper_width_default}

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
        T_W_C = self.imitation_episode.get_entry(idx)['T_W_C']
        obs = torch.tensor(self._observation_function(self.imitation_episode, idx, T_noise=np.eye(4)))
        obs = obs.type(torch.FloatTensor).cuda()
        obs = obs.unsqueeze(0)  # shape (1, obs_dim)

        with torch.no_grad():
            # push it through the network
            input_package = dict()
            input_package["observation"] = obs
            input_package["image"] = rgb_tensor.unsqueeze(0).cuda()
            actions = self.network.forward(input_package)

        if self._config["regression_type"] == "MDN":
            num_gaussians = self._config["num_gaussians"]
            L = 1
            A = self.network.action_size
            pi, sigma, mu = parse_mdn_params(actions, num_gaussians, L, A)
            pi_np, sigma_np, mu_np = pi.cpu().detach().numpy(), sigma.cpu().detach().numpy(), mu.detach().cpu().numpy()

            #EXPERIMENT
            # if pi_np[0,0] > pi_np[0,1]:
            #     #pi_np[0,0] = 1.0
            #     #pi_np[0,1] = 0.0
            #     sampled = mu_np[0,0]
            # else:
            #     pi_np[0,0] = 0.0
            #     pi_np[0,1] = 1.0
            #     sampled = mu_np[0,1]

            def gumbel_sample(x, axis=1):
                """
                x shape: N, num_gaussians (numpy.ndarray)
                returns: N,               (numpy.ndarray)
                """
                z = np.random.gumbel(loc=0, scale=1, size=x.shape)
                return (np.log(x) + z).argmax(axis=axis)

            k = gumbel_sample(pi_np**8)

            L = pi_np.shape[0]
            indices = (np.arange(L), k)
            rn = np.random.randn(L,self.network.action_size)
            sampled = rn * (sigma_np**8)[indices] + mu_np[indices]
                    
            actions = torch.from_numpy(sampled).squeeze()
            # print(actions)
            # print(type(actions))
            # print(actions.shape)
        else:
            actions = actions.cpu().squeeze().detach()

        # compute target transform

        # actions is the output of the network?
        action_dict = ros_imitation_parser.compute_action(actions, self._config, T_W_C=T_W_C,
                                                          T_W_C_default=self._T_W_C_default,
                                                          gripper_width_default=self._gripper_width_default)


        # self.actions_goal = actions
        self.actions_goal = action_dict
        self.update_interpolation_goals()

    def update_interpolation_goals(self): # no return
        """
        Update goals with interpolation
        """
        self.interp_index = 0

        # list of dicts
        # dict has keys {"ee_goal_pose": 4 x 4 homogeneous transform,
        # "gripper_width": float}
        #
        self.interp_goals = []

        if self.latest_interp_goal is None:
            for i in range(int(self.downsample_rate)):
                self.interp_goals.append(self.actions_goal)

        else:
            T_W_E_init = self.latest_interp_goal["T_W_E"]
            T_W_E_final = self.actions_goal["T_W_E"]


            R_stack = np.zeros([2,3,3])
            R_stack[0,:,:] = T_W_E_init[:3, :3]
            R_stack[1,:,:] = T_W_E_final[:3, :3]
            key_rots = Rotation.from_dcm(R_stack)
            key_times = [0, self.downsample_rate]
            slerp = Slerp(key_times, key_rots)
            interp_times = range(1, int(self.downsample_rate)+1)
            interp_times = 1.0*np.array(interp_times)
            interp_rots = slerp(interp_times)
            interp_rots_dcm = interp_rots.as_dcm()

            self.interp_goals = [None]*int(self.downsample_rate)
            for i in range(int(self.downsample_rate)):
                T_W_E = np.eye(4)
                T_W_E[:3, :3] = interp_rots_dcm[i, :, :]
                T_W_E[:3, 3] = T_W_E_init[:3, 3] + (i+1)*1.0/(self.downsample_rate) * (T_W_E_final[:3, 3] - T_W_E_init[:3, 3])
                # T_W_E[:3,3] = T_W_E_final[:3, 3]
                d = dict()
                d['T_W_E'] = T_W_E
                d['T_W_C'] = np.matmul(T_W_E, constants.T_E_cmd)
                d['gripper_width'] = self.actions_goal['gripper_width']
                self.interp_goals[i] = d

    def should_run_new_model(self): # type -> bool
        """
        Based on downsample rate, should we run the model?
        """
        if self.first_time_running_model:
            self.first_time_running_model = False
            return True

        if self.interp_index >= (int(self.downsample_rate) - 1):
            return True

        return False


    def compute_control_action(self, agent, #type ROSTaskSpaceControlAgent
                               ): # type -> (robot_msgs.msg.CartesianGoalPoint, float)

        if USE_DEPLOY_DEMO:
            events = self.mouse_manager.get_events()

            if events["escape"]:
                if BAG_DATA:
                    stop_bagging_imitation_data_client()
                sys.exit(0)

            if events["r"]:
                self._reset_function()
                import time; time.sleep(0.2) # this is just to give me a fraction of second more to reset
                self.first_time_running_model = True
                self._T_W_E_init_command_sent = False
                print("made it through resets")
                sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming', robot_msgs.srv.StartStreamingPlan)
                init = robot_msgs.srv.StartStreamingPlanRequest()
                res = sp(init)

                self.cache_reference_pose()
                self.software_safety.set_initial_goal(self.latest_interp_goal["T_W_E"])
                self.network.set_states_initial()

                goal = self.latest_interp_goal
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

                self.software_safety.sys_exit_if_not_safe(msg)

                return msg, self._gripper_width_default



        if not self._T_W_E_init_command_sent:
            self._T_W_E_init_command_sent = True
            goal = self.initial_pose
        else:
            if self.should_run_new_model():
                self.run_model(agent)

            self.latest_interp_goal = self.interp_goals[self.interp_index]

            self.interp_index += 1
            if self.interp_index >= int(self.downsample_rate):
                self.interp_index -= 1

            goal = self.interp_goals[self.interp_index]
            


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
                          T, # homogeneous transform, np.array shape [4,4]
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

    @property
    def config(self):
        return self._config




