from __future__ import print_function

import time
import os
import numpy as np
import pandas as pd

import torch

import rospy
import std_srvs.srv

import robot_msgs.msg
from spartan.utils import utils as spartan_utils
import spartan.utils.transformations as transformations

from imitation_tools.simulation_manager import SimulationManager
from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent

from imitation_agent.deploy.lstm_ee_position_agent import LSTMPositionAgent
from imitation_agent.deploy.mlp_ee_position_agent import MLPPositionAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.tasks import move_to_box, flip_box
from imitation_agent.evaluation.dataframe_wrapper import PandaDataFrameWrapper
from imitation_agent.training import train_utils
from imitation_agent.dataset.convex_hull_helper import ConvexHullHelper

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

evaluator = MoveToBoxEvaluator(create_imitation_agent)

evaluator.run_loop_of_deploys()

"""

# IF LUCAS
#PROCESSED_DIR = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716/2019-07-16-01-09-23/processed"
# IF PETE
PROCESSED_DIR = "/home/peteflo/data/pdc/imitation/move_to_box_careful/2019-07-10-15-13-54/processed"

# LOGGING_DIR_ROOT = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/deploy_evaluation/experiment_03")
LOGGING_DIR_ROOT = os.path.join(spartan_utils.getSpartanSourceDir(), 'sandbox', 'flip_box_evaluation')


class MoveToBoxEvaluator(object):

    def __init__(self, create_network_agent_function, # type func
                 logging_dir=None,
                 convex_hull=None, # ConvexHullHelper object
                 ): # type str
        """
        param create_network_agent_function: This is a function which returns a
        fresh constructed imitation agent.
        """

        self._create_network_agent_function = create_network_agent_function
        self._sim_manager = SimulationManager()
        self._convex_hull_helper = convex_hull

        if logging_dir is None:
            self._logging_dir = os.path.join(spartan_utils.getSpartanSourceDir(),"sandbox/move_to_box_test")
        else:
            self._logging_dir = logging_dir

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        self._dataframe = None

        self._dataframe_list = []

        self.termination_time = 15

        self._flip_box_termination_data = dict()
        self._flip_box_termination_data["last_time_not_vertical"] = time.time()


    def set_random_seed(self, seed):
        np.random.seed(seed)

    
    def run_loop_of_deploys(self, num_to_deploy):
        """
        Call this when ready to kick off the evaluation!
        """

        for deploy_idx in range(num_to_deploy):
            print("DEPLOY_IDX", deploy_idx)
            self.run_single_deploy(deploy_idx)


    def run_single_deploy(self, deploy_idx, rate, config):
        
        # create and save the config file
        sim_config, sim_config_file = self.create_new_config_file(deploy_idx)
        q0 = sim_config["instances"][0]["q0"]
        pos = np.array(q0[0:3])
        rpy = np.array(q0[3:6])


        T_W_B = transformations.euler_matrix(rpy[0], rpy[1], rpy[2])
        T_W_B[0:3, 3] = pos

        # initialize sim?
        self._sim_manager.set_up(sim_config_file)


        # load a ros_task_space_control_agent
        task_space_control_agent = ROSTaskSpaceControlAgent(config =config)
        # wait for sim to start
        task_space_control_agent._robot_joint_states_subscriber.waitForNextMessage()

        task_space_control_agent.move_to_starting_position()


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

        rate = rospy.Rate(rate)

        # configure the agent a bit
        # create a fresh agent
        imitation_agent = self._create_network_agent_function()
        imitation_agent.SIMULATION_SAFETY_CONFIG = True
        imitation_agent.imitation_episode.set_online_sim_config_dict(sim_config)
        task_space_control_agent.control_function = imitation_agent.compute_control_action

        termination_type = None

        # LOOP:
        start_time = time.time()
        while not rospy.is_shutdown():

            # step the agent
            task_space_control_agent.step()

            # check task status
            task_relevant_state = self.get_task_relevant_state(task_space_control_agent)

            if self.conditional_terminate_on_unstable(imitation_agent._unsafe):
                termination_type = "UNSTABLE"
                print("TERMINATING BECAUSE UNSTABLE")
                break

            if self.conditional_terminate_on_state(task_relevant_state):
                print("TERMINATING ON STATE")
                termination_type = "STATE"
                break

            if self.conditional_terminate_on_time(start_time):
                print("TERMINATING ON TIME")
                termination_type = "TIME"
                break

            rate.sleep()

        duration = time.time() - start_time

        # compute reward based only on final relevant state
        state = task_relevant_state
        if not self.use_flip:
            reward_data = move_to_box.compute_reward(state["T_W_B"], state["T_W_E"])
        else:
            #reward_data = flip_box.compute_reward(state["T_W_B"])
            reward_data = flip_box.compute_reward(state["T_W_B"])

        save_data = dict()
        save_data['index'] = deploy_idx
        save_data['state'] = state
        save_data['termination_type'] = termination_type
        save_data['reward'] = reward_data['reward']
        save_data['reward_data'] = reward_data
        save_data["object_position"] = pos
        save_data["object_rpy"] = rpy
        save_data["T_W_B"] = state["T_W_B"]

        self.save_single_deploy_data(save_data, deploy_idx, duration)

        cleanup()

        self._sim_manager.tear_down()

    
    def get_task_relevant_state(self, ros_task_space_control_agent):
        state = ros_task_space_control_agent.data_list[-1]
        T_W_E_dict = state['observations']['ee_to_world']

        T_W_E = spartan_utils.homogenous_transform_from_dict(T_W_E_dict)

        # also T_W_B
        T_W_B = spartan_utils.homogenous_transform_from_dict(state["observations"]['object_pose_cheat_data'])
        return {"T_W_E": T_W_E,
                "T_W_B": T_W_B}



    def conditional_terminate_on_state(self, state, # type dict
                                       ): # type -> bool

        terminate = False

        if self.use_flip:
            success = flip_box.is_sucessful(state["T_W_B"])

            angle_error_to_target = flip_box.angle_error_to_target(state["T_W_B"])
            # print("\n")
            # print("box_vertical:", success)
            # print("angle error (degrees)", np.rad2deg(angle_error_to_target))
            if success:
                # print("box flip success")
                if (time.time() - self._flip_box_termination_data["last_time_not_vertical"]) > 2.0:
                    print("box has been vertical for designated time, terminating")
                    terminate = True
            else:
                self._flip_box_termination_data["last_time_not_vertical"] = time.time()

        else:
            reward_data = move_to_box.compute_reward(state["T_W_B"], state["T_W_E"])
            print("reward:", reward_data['reward'])
            reward_threshold = -1.2
            if (reward_data['reward'] > reward_threshold):
                terminate = True

        return terminate


    def conditional_terminate_on_time(self, start_time):

        if (time.time() - start_time > self.termination_time):
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

        dfw = None
        if not self.use_flip:
            dfw = MoveToBoxPandasTemplate()
            dfw.set_value("date_time", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
            dfw.set_value("reward", reward)
            dfw.set_value("termination_type", data["termination_type"])
            dfw.set_value("position_error", reward_data["pos"])
            dfw.set_value("angle_error", reward_data["angle"])
            dfw.set_value("object_position", [data["object_position"].tolist()])
            dfw.set_value("object_rpy", [data["object_rpy"].tolist()])
            dfw.set_value("index", deploy_idx)
            dfw.set_value("termination_time", duration)
        else:
            dfw = FlipBoxPandasTemplate()
            dfw.set_value("date_time", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
            dfw.set_value("reward", reward)
            dfw.set_value("termination_type", data["termination_type"])

            dfw.set_value("object_position", [data["object_position"].tolist()])
            dfw.set_value("object_rpy", [data["object_rpy"].tolist()])
            dfw.set_value("deploy_index", deploy_idx)
            dfw.set_value("termination_time", duration)

            dfw.set_value("success", reward_data["success"])
            dfw.set_value("reward", reward_data["reward"])
            dfw.set_value("angle_error", reward_data["angle_error"])
            dfw.set_value("angle_error_degrees", reward_data["angle_error_degrees"])


            final_pos = data["T_W_B"][:3, 3]
            final_quat = transformations.quaternion_from_matrix(data["T_W_B"])
            dfw.set_value("final_pos_x", final_pos[0])
            dfw.set_value("final_pos_y", final_pos[1])
            dfw.set_value("final_pos_z", final_pos[2])

            dfw.set_value("final_quat_w", final_quat[0])
            dfw.set_value("final_quat_x", final_quat[1])
            dfw.set_value("final_quat_y", final_quat[2])
            dfw.set_value("final_quat_z", final_quat[3])


        if self._dataframe is None:
            self._dataframe = dfw.dataframe
        else:
            print(len(self._dataframe))
            self._dataframe = pd.concat([self._dataframe, dfw.dataframe])


        print(len(self._dataframe))

        #df = pd.concat(self._dataframe_list)
        self._dataframe.to_csv(results_csv_file)


    def create_new_config_file(self, index):
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


    def set_use_se2(self, use_se2):
        self.use_se2 = use_se2

    def set_use_flip(self, use_flip):
        self.use_flip = use_flip
        if use_flip:
            self.termination_time = 20

    def sample_initial_pose(self):
        print("SAMPLING INITIAL POSE")
        print("\n \n")

        q0 = np.array([0.61, 0.0, 0.05, 0.0, 0.0, 1.57])
        while True:

            x_range = np.array([0.6, 0.7])
            y_range = np.array([-0.2, 0.2])

            x_val = np.random.random(1)*(x_range[1] - x_range[0]) + x_range[0]
            y_val = np.random.random(1) * (y_range[1] - y_range[0]) + y_range[0]

            q0[0] = x_val
            q0[1] = y_val

            if self.use_se2:
                theta_range = np.array([-np.deg2rad(45),np.deg2rad(45)])
                theta_sampled_delta = np.random.random(1) * (theta_range[1] - theta_range[0]) + theta_range[0]
                print(theta_sampled_delta, "is theta delta")
                q0[-1] = 1.57 + theta_sampled_delta

            # SET SPECIFIC FOR DEBUGGING
            #q0[0:3] = np.asarray([0.6632670558646401, -0.02846374967378179, 0.05])

            if self._convex_hull_helper is None:
                break
            else:
                if self._convex_hull_helper.contains_point(q0[0:2]):
                    break

        print("q0:", q0)
        return q0


class MoveToBoxPandasTemplate(PandaDataFrameWrapper):
    columns = ['date_time',
               'reward',
               'termination_type',
               'object_position',
               'object_rpy',
               'position_error',
               'angle_error',
               'index',
               'termination_time'
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, MoveToBoxPandasTemplate.columns)

class FlipBoxPandasTemplate(PandaDataFrameWrapper):
    columns = ['date_time',
               'reward',
               'termination_type',
               'object_position',
               'object_rpy',
               'angle_error',
               "success",
               'angle_error_degrees',
               'deploy_index',
               'termination_time',
               'final_pos_x',
               'final_pos_y',
               'final_pos_z',
               'final_quat_w',
               'final_quat_x',
               'final_quat_y',
               'final_quat_z',
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, FlipBoxPandasTemplate.columns)



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
    
    rospy.init_node("move_to_box_evaluator")
    network, readable_name = load_network()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return LSTMPositionAgent(network, observation_function, episode, config)

    logging_dir = os.path.join(spartan_utils.getSpartanSourceDir(),"sandbox",readable_name)
    evaluator = MoveToBoxEvaluator(create_imitation_agent, logging_dir = logging_dir)

    num_deploys = 2000
    train_utils.make_deterministic()
    np.random.seed(0)
    evaluator.run_loop_of_deploys(num_deploys)



def run_one_lstm_position_agent(deploy_idx, seed=None, network_chkpt=None, use_se2=False, use_flip=False, logging_dir=None, training_poses_file=None):
    rospy.init_node("move_to_box_evaluator")
    
    network, readable_name = load_network_lstm_position(network_chkpt)
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()
    config = network._policy_net._config

    # overwrite the software safety params
    config["software_safety"] = dict()
    config["software_safety"]["rotation_threshold_degrees"] = 3.0
    config["software_safety"]["translation_threshold"] = 0.02


    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return LSTMPositionAgent(network, observation_function, episode, config)

    convex_hull = None
    if training_poses_file is not None:
        print("loading training poses from file")
        training_object_poses = np.load(training_poses_file)
        convex_hull = ConvexHullHelper(training_object_poses)

    logging_dir = os.path.join(logging_dir, readable_name)
    evaluator = MoveToBoxEvaluator(create_imitation_agent, logging_dir=logging_dir, convex_hull=convex_hull)

    if seed is not None:
        evaluator.set_random_seed(seed)
    
    evaluator.set_use_se2(use_se2)
    evaluator.set_use_flip(use_flip)

    num_deploys = 1
    evaluator.run_single_deploy(deploy_idx, rate=100, config=config) 


def run_one_mlp_position_agent(deploy_idx, seed=None, network_chkpt=None, use_se2=False, use_flip=False, logging_dir=None, training_poses_file=None):
    rospy.init_node("move_to_box_evaluator")
    
    network, readable_name = load_network_mlp_position(network_chkpt)
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return MLPPositionAgent(network, observation_function, episode, config)

    convex_hull = None
    if training_poses_file is not None:
        training_object_poses = np.load(training_poses_file)
        convex_hull = ConvexHullHelper(training_object_poses)

    logging_dir = os.path.join(logging_dir, readable_name)
    evaluator = MoveToBoxEvaluator(create_imitation_agent, logging_dir=logging_dir, convex_hull=convex_hull)

    if seed is not None:
        evaluator.set_random_seed(seed)

    evaluator.set_use_se2(use_se2)
    evaluator.set_use_flip(use_flip)


    num_deploys = 1
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
        "--se2",
        "-se2",
        dest="use_se2",
        required=False,
        default=False,
        action='store_true',
        help="Pass if you'd like to sample over se2 instead of just r2"
    )
    parser.add_argument(
        "--flip",
        "-flip",
        dest="use_flip",
        required=False,
        default=False,
        action='store_true',
        help="Pass if you'd like to flip the box"
    )

    parser.add_argument(
        "--logging_dir",
        "-logging_dir",
        dest="logging_dir",
        required=False,
        default=LOGGING_DIR_ROOT,
        help="specify the logging directory"
    )

    parser.add_argument(
        "--training_poses_file",
        "-training_poses_file",
        dest="training_poses_file",
        required=False,
        default=None,
        help="(optional) full path to npy file containing object poses used during training. Will only sample new poses that lie in the convex hull"
    )

    args = parser.parse_args()    
    print(args)

    if args.run_one_mlp_position:
        print("you want to run an mlp position")
        run_one_mlp_position_agent(int(args.deploy_idx), seed=int(args.random_seed), network_chkpt=args.network_chkpt, use_se2=args.use_se2, use_flip=args.use_flip, logging_dir=args.logging_dir, training_poses_file=args.training_poses_file)
    elif args.run_one_lstm_position:
        print("you want to run an lstm position")
        run_one_lstm_position_agent(int(args.deploy_idx), seed=int(args.random_seed), network_chkpt=args.network_chkpt, use_se2=args.use_se2, use_flip=args.use_flip, logging_dir=args.logging_dir, training_poses_file=args.training_poses_file)
    else:
        print("running test")

    #test()
