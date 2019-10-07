from __future__ import print_function

import time
import os
import numpy as np
import pandas as pd
import random
import torch

import rospy
import std_srvs.srv
import visualization_msgs.msg

import robot_msgs.msg
from spartan.utils import utils as spartan_utils
import spartan.utils.transformations as transformations
from spartan.utils.ros_utils import SimpleSubscriber, RobotService, poseFromROSPoseMsg

from imitation_tools.simulation_manager import SimulationManager
from imitation_tools.external_wrench import ExternalWrench
from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent

from imitation_agent.deploy.lstm_ee_position_agent import LSTMPositionAgent
from imitation_agent.deploy.mlp_ee_position_agent import MLPPositionAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.tasks import push_box
from imitation_agent.evaluation.dataframe_wrapper import PandaDataFrameWrapper
from imitation_agent.training import train_utils

import dense_correspondence_manipulation.utils.utils as pdc_utils
#pdc_utils.set_cuda_visible_devices([0]) # in general, good to not use same gpu as for rendering




"""
How to use:

network = #
observation_function = #
episode = #
config = #

def create_imitation_agent():
    return LSTMPositionAgent(network, observation_function, episode, config)

evaluator = PushBoxEvaluator(create_imitation_agent)

evaluator.run_loop_of_deploys()

"""


# IF LUCAS
PROCESSED_DIR = "/home/manuelli/data_ssd/imitation/logs/push_box/2019-07-30-19-18-08/processed"
# IF PETE
# PROCESSED_DIR = "/home/peteflo/data/pdc/imitation/move_to_box_careful/2019-07-10-15-13-54/processed"

LOGGING_DIR_ROOT = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/deploy_evaluation/experiment_03")


def sample_initial_pose_uniform():
    q0 = np.array([0.61, 0.0, 0.05, 0.0, 0.0, 1.57])

    x_val = random.uniform(0.58, 0.62)
    y_val = random.uniform(-0.24, -0.24 + 0.1)
    yaw_val = random.uniform(np.pi/2.0 - np.deg2rad(30), np.pi/2.0 + np.deg2rad(30))

    # x_val = 0.58
    # y_val = -0.24
    # yaw_val = np.pi/2.0 + np.deg2rad(30)

    q0[0] = x_val
    q0[1] = y_val
    q0[5] = yaw_val

    return q0

class PushBoxEvaluator(object):

    def __init__(self, create_network_agent_function, # type func
                       logging_dir=None): # type str
        """
        param create_network_agent_function: This is a function which returns a
        fresh constructed imitation agent.
        """

        self._create_network_agent_function = create_network_agent_function
        self._sim_manager = SimulationManager()
        self._add_external_wrench = True

        if logging_dir is None:
            self._logging_dir = os.path.join(spartan_utils.getSpartanSourceDir(),"sandbox/move_to_box_test")
        else:
            self._logging_dir = logging_dir

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        stored_poses_file = os.path.join(spartan_utils.getSpartanSourceDir(),
                                         "src/catkin_projects/station_config/RLG_iiwa_1/stored_poses.yaml")
        self._stored_poses_dict = spartan_utils.getDictFromYamlFilename(stored_poses_file)

        self._dataframe = None

        self._dataframe_list = []

        self._setup_subscribers()
        self.external_wrench = ExternalWrench()

    def _setup_subscribers(self):
        self._box_position_subscriber = SimpleSubscriber("/scene_graph/update",
                                                         visualization_msgs.msg.InteractiveMarkerUpdate,
                                                         externalCallback=self._on_object_pose_msg)
        self._box_position_subscriber.start()

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def _on_object_pose_msg(self, msg):
        if len(msg.poses) != 1:
            return

        data = msg.poses[0]
        if not "base_link_sugar" in data.name:
            return

        # box to world transform

        pos, quat = poseFromROSPoseMsg(data.pose)
        T_W_B = transformations.quaternion_matrix(quat)
        T_W_B[:3, 3] = pos

        self._object_pose = dict()
        self._object_pose["T_W_B"] = T_W_B

    
    def run_loop_of_deploys(self, num_to_deploy):
        """
        Call this when ready to kick off the evaluation!
        """

        for deploy_idx in range(num_to_deploy):
            print("DEPLOY_IDX", deploy_idx)
            self.run_single_deploy(deploy_idx)


    def run_single_deploy(self, deploy_idx, rate, config=None):
        
        # create and save the sim_config file
        sim_config, sim_config_file = self.create_new_sim_config_file(deploy_idx)
        q0 = sim_config["instances"][0]["q0"]
        pos = np.array(q0[0:3])
        rpy = np.array(q0[3:6])

        # initialize sim?
        self._sim_manager.set_up(sim_config_file)

        
        # load a ros_task_space_control_agent
        task_space_control_agent = ROSTaskSpaceControlAgent(config=config)
        # wait for sim to start
        task_space_control_agent._robot_joint_states_subscriber.waitForNextMessage()


        # close gripper
        task_space_control_agent.gripper_driver.closeGripper()

        # move to starting position
        start_pose = self._stored_poses_dict["imitation"]["push_box_start_left"]
        success = task_space_control_agent.robot_service.moveToJointPosition(start_pose, maxJointDegreesPerSecond=60, timeout=5)


        # start running control
        service_proxy = rospy.ServiceProxy('plan_runner/init_task_space_streaming',
        robot_msgs.srv.StartStreamingPlan)

        init = robot_msgs.srv.StartStreamingPlanRequest()
        res = service_proxy(init)

        def cleanup():
            rospy.wait_for_service("plan_runner/stop_plan")
            sp = rospy.ServiceProxy('plan_runner/stop_plan',
                                    std_srvs.srv.Trigger)
            init = std_srvs.srv.TriggerRequest()
            sp(init)
            print("Done cleaning up and stopping streaming plan")


        # TODO: how to sim_config this later?
        rate = rospy.Rate(rate)

        # create a fresh agent
        imitation_agent = self._create_network_agent_function()
        imitation_agent.imitation_episode.set_online_sim_config_dict(sim_config)
        imitation_agent.SIMULATION_SAFETY_CONFIG = True

        # set T_W_C_default for this task
        imitation_agent.set_T_W_C_default(push_box.T_W_C_default)
        imitation_agent.set_gripper_width_default(0.0)

        task_space_control_agent.control_function = imitation_agent.compute_control_action


        termination_type = None

        # LOOP:
        start_time = time.time()
        task_relevant_state = None

        times_list = []
        object_pose_list = [] # list of length 16 lists
        while not rospy.is_shutdown():

            # step the agent
            # add disturbance if necessary
            if self._add_external_wrench:
                self.external_wrench.step(self._object_pose["T_W_B"])

            task_space_control_agent.step()

            # check task status
            task_relevant_state = dict()
            task_relevant_state["T_W_B"] = self._object_pose["T_W_B"]

            # record information
            times_list.append(time.time() - start_time)
            object_pose_list.append(self._object_pose["T_W_B"].tolist())

            if self.conditional_terminate_on_unstable(imitation_agent._unsafe):
                termination_type = "UNSTABLE"
                print("TERMINATING BECAUSE UNSTABLE")
                break

            if self.conditional_terminate_on_state(task_relevant_state):
                termination_type = "STATE"
                break

            if self.conditional_terminate_on_time(start_time):
                termination_type = "TIME"
                break

            rate.sleep()

        print("termination_type:", termination_type)
        duration = time.time() - start_time

        # compute reward based only on final relevant state
        state = task_relevant_state
        reward_data = push_box.compute_reward(state["T_W_B"])

        save_data = dict()
        save_data['index'] = deploy_idx
        save_data['state'] = state
        save_data['termination_type'] = termination_type
        save_data['reward'] = reward_data['reward']
        save_data['reward_data'] = reward_data
        save_data["object_position"] = pos.tolist()
        save_data["object_rpy"] = rpy.tolist()
        save_data['times'] = times_list
        save_data['object_poses'] = object_pose_list

        self.save_single_deploy_data(save_data, deploy_idx, duration)

        cleanup()

        self._sim_manager.tear_down()


    def conditional_terminate_on_state(self, state, # type dict
                                       ): # type -> bool

        return push_box.should_terminate(state["T_W_B"], demonstration=False)

    def conditional_terminate_on_time(self, start_time):
        max_time_allowable = 30
        
        if (time.time() - start_time > max_time_allowable):
            return True

        return False

    def conditional_terminate_on_unstable(self, unstable_flag):
        if unstable_flag:
            return True

        return False

    def save_single_deploy_data(self, data, deploy_idx, duration):
        """
        This should write to disk! (And just overwrite / append)
        """
        results_csv_file = os.path.join(self._logging_dir, "results.csv")

        if os.path.exists(results_csv_file):
            self._dataframe = pd.read_csv(results_csv_file, index_col=0)

        index = data['index']
        reward = data['reward']
        reward_data = data["reward_data"]
        state = data['state']

        dfw = PushBoxPandasTemplate()
        dfw.set_value("date_time", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
        dfw.set_value("reward", reward)
        dfw.set_value("termination_type", data["termination_type"])
        dfw.set_value("y_error", reward_data["y_error"])
        dfw.set_value("angle_error", reward_data["angle_error"])
        dfw.set_value("object_position", [data["object_position"]])
        dfw.set_value("object_rpy", [data["object_rpy"]])
        dfw.set_value("index", deploy_idx)
        dfw.set_value("termination_time", duration)
        dfw.set_value("times", [data['times']])
        dfw.set_value("object_poses", [data["object_poses"]])


        print("duration:", data['times'][-1])

        if self._dataframe is None:
            self._dataframe = dfw.dataframe
        else:
            print(len(self._dataframe))
            self._dataframe = self._dataframe.append(dfw.dataframe, sort=False)
        print(len(self._dataframe))

        #df = pd.concat(self._dataframe_list)
        self._dataframe.to_csv(results_csv_file)

        # save additional data

        index_str = str(index).zfill(5)
        json_data = dict()
        keys = ["index", "termination_type", "reward", "object_position",
                "object_rpy", "times", "object_poses"]
        for key in keys:
            json_data[key] = data[key]

        json_filename = os.path.join(self._logging_dir, "data_%s.json" %(index_str))
        spartan_utils.save_to_json(json_data, json_filename)



    def create_new_sim_config_file(self, index):
        sim_config_file = os.path.join(spartan_utils.getSpartanSourceDir(), "src/catkin_projects/drake_iiwa_sim/config/sim_config_externalcameras_manipulation.yaml")

        sim_config = spartan_utils.getDictFromYamlFilename(sim_config_file)
        
        object_pose = self.sample_initial_pose()

        sugar_box_dict = {'model': "sugar",
                                   'q0': object_pose.tolist(),
                                   'fixed': False,
                                   'body_name': "base_link_sugar"}
        sim_config["instances"] = [sugar_box_dict]

        index_str = str(index).zfill(5)

        sim_config_save_file = os.path.join(self._logging_dir, "sim_config_"+index_str+".yaml")
        spartan_utils.saveToYaml(sim_config, sim_config_save_file)
        return sim_config, sim_config_save_file

    def sample_initial_pose(self):
        q0 = np.array([0.61, 0.0, 0.05, 0.0, 0.0, 1.57])

        x_val = random.uniform(0.58, 0.62)
        y_val = random.uniform(-0.24, -0.24 + 0.1)
        yaw_val = random.uniform(np.pi / 2.0 - np.deg2rad(30), np.pi / 2.0 + np.deg2rad(30))

        q0[0] = x_val
        q0[1] = y_val
        q0[5] = yaw_val

        return q0

class PushBoxPandasTemplate(PandaDataFrameWrapper):
    columns = ['date_time',
               'reward',
               'termination_type',
               'object_position',
               'object_rpy',
               'y_error',
               'angle_error',
               'index',
               'termination_time',
               'times',
               'object_poses',
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, PushBoxPandasTemplate.columns)



def get_most_recent_network_lstm_position():
    base = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/trained_models/lstm_standard")
    most_recent_dir = sorted(os.listdir(base))[-1]
    longest_chkpt = sorted(os.listdir(os.path.join(base,most_recent_dir)))[-2] # "tensorboard is last"
    return os.path.join(base,most_recent_dir,longest_chkpt)

def get_most_recent_network_mlp_posiiton():
    base = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/trained_models/mlp_position")
    most_recent_dir = sorted(os.listdir(base))[-1]
    longest_chkpt = sorted(os.listdir(os.path.join(base,most_recent_dir)))[-2] # "tensorboard is last"
    return os.path.join(base,most_recent_dir,longest_chkpt)

def load_network_lstm_position(network_chkpt=None):
    
    if network_chkpt is None:
        network_chkpt = get_most_recent_network_lstm_position()
    #network_chkpt =  "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-06-27-02-43-28/iter-026204.pth" # vision
    #network_chkpt =  "/home/peteflo/data/pdc/imitation/trained_models/lstm_standard/2019-07-03-03-54-39/iter-020100.pth" # pose in

    readable_name = network_chkpt.split("lstm_standard/")[-1].split("/iter")[0]

    print("loading net:", network_chkpt)
    network = torch.load(network_chkpt)
    return network, readable_name

def load_network_mlp_position(network_chkpt=None):

    if network_chkpt is None:
        network_chkpt = get_most_recent_network_mlp_posiiton()

    readable_name = network_chkpt.split("mlp_position/")[-1].split("/iter")[0]


    print("loading net:", network_chkpt)
    network = torch.load(network_chkpt)
    return network, readable_name

def construct_imitation_episode(config):

    action_function = ActionFunctionFactory.action_from_config(config)

    observation_function = ObservationFunctionFactory.get_function(config)
    # observation_function = ObservationFunctionFactory.observation_from_config(config)

    imitation_episode = ImitationEpisode(config,
                                         type="online",
                                         action_function=action_function,
                                         observation_function=observation_function,
                                         processed_dir=PROCESSED_DIR)
    return action_function, observation_function, imitation_episode


def test():
    
    rospy.init_node("push_box_evaluator")
    network, readable_name = load_network()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return LSTMPositionAgent(network, observation_function, episode, config)

    logging_dir = os.path.join(LOGGING_DIR_ROOT,readable_name)
    evaluator = PushBoxEvaluator(create_imitation_agent, logging_dir = logging_dir)

    num_deploys = 2000
    train_utils.make_deterministic()
    np.random.seed(0)
    evaluator.run_loop_of_deploys(num_deploys)



def run_one_lstm_position_agent(deploy_idx, seed=None, network_chkpt=None, logging_dir=None):
    rospy.init_node("push_box_evaluator")
    
    network, readable_name = load_network_lstm_position(network_chkpt)
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()
    config = network._policy_net._config

    # print(network.logs_config.keys())
    # print("num train logs", len(network.logs_config["logs"]))
    # quit()

    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return LSTMPositionAgent(network, observation_function, episode, config)

    logging_dir = os.path.join(logging_dir,readable_name)
    evaluator = PushBoxEvaluator(create_imitation_agent, logging_dir = logging_dir)

    if seed is not None:
        evaluator.set_random_seed(seed)

    num_deploys = 1
    evaluator.run_single_deploy(deploy_idx, rate=100, config=config)


def run_one_mlp_position_agent(deploy_idx, seed=None, network_chkpt=None, logging_dir=None):
    rospy.init_node("push_box_evaluator")
    
    network, readable_name = load_network_mlp_position(network_chkpt)
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return MLPPositionAgent(network, observation_function, episode, config)

    logging_dir = os.path.join(logging_dir, readable_name)
    evaluator = PushBoxEvaluator(create_imitation_agent, logging_dir = logging_dir)

    if seed is not None:
        evaluator.set_random_seed(seed)

    num_deploys = 1
    print("running single deploy")
    evaluator.run_single_deploy(deploy_idx, rate=30, config=config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlp",
        "-m",
        dest="run_one_mlp_position",
        action="store_true",
        required=False,
        help="Pass if you'd like to run an mlp position"
    )
    parser.add_argument(
        "--lstm",
        "-l",
        dest="run_one_lstm_position",
        required=False,
        action='store_true',
        help="Pass if you'd like to run an mlp position"
    )
    parser.add_argument(
        "--seed",
        "-s",
        dest="random_seed",
        required=False,
        default=0,
        help="Pass if you'd like to set the random seed"
    )
    parser.add_argument(
        "--net",
        "-n",
        dest="network_chkpt",
        required=False,
        help="Pass if you'd like to run a particular network_chkpt"
    )
    parser.add_argument(
        "--deploy-idx",
        "-d",
        dest="deploy_idx",
        required=False,
        default=1,
        help="Set the deploy index"
    )
    parser.add_argument(
        "--logging_dir",
        "-logging_dir",
        dest="logging_dir",
        required=False,
        default=LOGGING_DIR_ROOT,
        help="specify the logging directory"
    )

    args = parser.parse_args()    
    print(args)

    if args.run_one_mlp_position:
        print("you want to run an mlp position")
        run_one_mlp_position_agent(int(args.deploy_idx), seed=int(args.random_seed), network_chkpt=args.network_chkpt, logging_dir=args.logging_dir)
    elif args.run_one_lstm_position:
        print("you want to run an lstm position")
        run_one_lstm_position_agent(int(args.deploy_idx), seed=int(args.random_seed), network_chkpt=args.network_chkpt, logging_dir=args.logging_dir)
    else:
        print("running test")

    #test()
