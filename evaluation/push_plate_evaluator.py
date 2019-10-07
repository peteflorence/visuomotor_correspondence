from __future__ import print_function

import time
import os
import numpy as np
import pandas as pd
import random
import torch
import ros_numpy

import rospy
import std_srvs.srv
import visualization_msgs.msg

import robot_msgs.msg
from spartan.utils import utils as spartan_utils
import spartan.utils.transformations as transformations

from imitation_tools.simulation_manager import SimulationManager
from imitation_tools.external_wrench import ExternalWrench

from imitation_agent.deploy.ros_task_space_control_agent import ROSTaskSpaceControlAgent
from imitation_agent.deploy.lstm_ee_position_agent import LSTMPositionAgent
from imitation_agent.deploy.mlp_ee_position_agent import MLPPositionAgent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.tasks import push_plate
from imitation_agent.evaluation.dataframe_wrapper import PandaDataFrameWrapper
from imitation_agent.training import train_utils
from imitation_agent.dataset.convex_hull_helper import ConvexHullHelper
from imitation_agent.tasks.push_plate import sample_initial_pose

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
PROCESSED_DIR = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716/2019-07-16-01-09-23/processed"
# IF PETE
# PROCESSED_DIR = "/home/peteflo/data/pdc/imitation/move_to_box_careful/2019-07-10-15-13-54/processed"

# LOGGING_DIR_ROOT = os.path.join(spartan_utils.get_data_dir(), "pdc/imitation/deploy_evaluation/experiment_03")
LOGGING_DIR_ROOT = os.path.join(spartan_utils.getSpartanSourceDir(), 'sandbox', 'push_plate_evaluation')


class PushPlateEvaluator(object):

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
        self._external_wrench = ExternalWrench(topic="plate_external_wrench", object_type="plate")
        self._add_disturbance = True

        if logging_dir is None:
            self._logging_dir = os.path.join(spartan_utils.getSpartanSourceDir(),"sandbox/push_plate_test")
        else:
            self._logging_dir = logging_dir

        if not os.path.isdir(self._logging_dir):
            os.makedirs(self._logging_dir)

        self._dataframe = None

        self._dataframe_list = []

        self.termination_time = 30
        self.setup_publishers()

    def setup_publishers(self):
        self._rviz_marker_publisher = rospy.Publisher("visualization_marker", visualization_msgs.msg.Marker, queue_size=1)

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)


    def visualize_goal_location(self, send_time=0.5):
        """
        Visualizes goal location in RVIZ using a sphere
        :return:
        :rtype:
        """


        msg = visualization_msgs.msg.Marker()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base"
        msg.type = visualization_msgs.msg.Marker.SPHERE
        msg.pose = ros_numpy.geometry.numpy_to_pose(push_plate.T_W_P_goal)
        msg.id = 0

        scale = 0.03
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale

        msg.color.g = 1.0
        msg.color.a = 0.75


        rate = rospy.Rate(100)
        start_time = time.time()
        while (time.time() - start_time) < send_time:
            self._rviz_marker_publisher.publish(msg)
            rate.sleep()

    
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


        T_W_O = transformations.euler_matrix(rpy[0], rpy[1], rpy[2])
        T_W_O[0:3, 3] = pos

        # initialize sim?
        self._sim_manager.set_up(sim_config_file)
        # wait for sim to start



        # load a ros_task_space_control_agent
        task_space_control_agent = ROSTaskSpaceControlAgent(config =config)
        # wait for sim to start
        task_space_control_agent._robot_joint_states_subscriber.waitForNextMessage()
        rospy.sleep(1.0)


        # move to the starting position
        print("moving to starting position")
        above_table_pre_grasp = task_space_control_agent._stored_poses_dict["Grasping"]["above_table_pre_grasp"]
        success = task_space_control_agent._robot_service.moveToJointPosition(above_table_pre_grasp, maxJointDegreesPerSecond=60, timeout=5)
        task_space_control_agent._gripper_driver.closeGripper()

        if not success:
            raise ValueError("Failed to move to starting position")


        task_space_control_agent._gripper_driver.sendCloseGripperCommand()
        self.visualize_goal_location()


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
        print("\n\n----------Creating Imitation Agent--------\n\n")
        imitation_agent = self._create_network_agent_function()
        imitation_agent.SIMULATION_SAFETY_CONFIG = True
        imitation_agent.set_gripper_width_default(0.0) # zero gripper width default
        imitation_agent.imitation_episode.set_online_sim_config_dict(sim_config)
        task_space_control_agent.control_function = imitation_agent.compute_control_action
        print("\n\n----------Finished Creating Imitation Agent--------\n\n")

        # start running control
        service_proxy = rospy.ServiceProxy('plan_runner/init_task_space_streaming',
                                           robot_msgs.srv.StartStreamingPlan)

        init = robot_msgs.srv.StartStreamingPlanRequest()
        res = service_proxy(init)

        termination_type = None

        # LOOP:
        start_time = time.time()
        task_relevant_state = None
        times = []
        object_poses = []
        while not rospy.is_shutdown():


            # print("\n\n----test-----\n\n")
            # step the agent
            task_space_control_agent.step()

            # check task status
            task_relevant_state = self.get_task_relevant_state(task_space_control_agent)
            times.append(time.time() - start_time)
            object_poses.append(task_relevant_state["T_W_O"].tolist())

            if self.conditional_terminate_on_unstable(imitation_agent._unsafe):
                termination_type = "UNSTABLE"
                print("TERMINATING BECAUSE UNSTABLE")
                break

            if self.conditional_terminate_on_state(task_relevant_state):
                print("TERMINATING BASED ON STATE")
                termination_type = "STATE"
                break

            if self.conditional_terminate_on_time(start_time):
                print("TERMINATE BASED ON TIME")
                termination_type = "TIME"
                break


            #add disturbance
            if self._add_disturbance:
                self._external_wrench.step(task_relevant_state["T_W_O"])

            rate.sleep()

        duration = time.time() - start_time


        T_W_O_final = task_relevant_state["T_W_O"]
        distance_to_goal = push_plate.compute_distance_to_goal(task_relevant_state["T_W_O"])
        save_data = {"index": deploy_idx,
                     "state": task_relevant_state,
                     "termination_type": termination_type,
                     "distance_to_goal": distance_to_goal,
                     "object_pose_final": T_W_O_final,
                     "duration": duration,
                     "object_pose_start": q0,
                     "times": times,
                     "object_poses": object_poses}

        self.save_single_deploy_data(save_data)


        # # compute reward based only on final relevant state
        # state = task_relevant_state
        # if not self.use_flip:
        #     reward_data = move_to_box.compute_reward(state["T_W_O"], state["T_W_E"])
        # else:
        #     #reward_data = flip_box.compute_reward(state["T_W_O"])
        #     reward_data = flip_box.compute_reward(state["T_W_O"])
        #
        # save_data = dict()
        # save_data['index'] = deploy_idx
        # save_data['state'] = state
        # save_data['termination_type'] = termination_type
        # save_data['reward'] = reward_data['reward']
        # save_data['reward_data'] = reward_data
        # save_data["object_position"] = pos
        # save_data["object_rpy"] = rpy
        # save_data["T_W_O"] = state["T_W_O"]
        #
        # self.save_single_deploy_data(save_data, deploy_idx, duration)

        cleanup()

        self._sim_manager.tear_down()

    
    def get_task_relevant_state(self, ros_task_space_control_agent):
        state = ros_task_space_control_agent.data_list[-1]
        T_W_E_dict = state['observations']['ee_to_world']

        T_W_E = spartan_utils.homogenous_transform_from_dict(T_W_E_dict)

        # also T_W_O
        T_W_O = spartan_utils.homogenous_transform_from_dict(state["observations"]['object_pose_cheat_data'])
        return {"T_W_E": T_W_E,
                "T_W_O": T_W_O}



    def conditional_terminate_on_state(self, state, # type dict
                                       ): # type -> bool

        # print("Plate position", state["T_W_O"][:3, 3])
        dist_to_goal, terminate = push_plate.should_terminate(state["T_W_O"])
        # print("dist to goal", dist_to_goal)
        return terminate



    def conditional_terminate_on_time(self, start_time):

        if (time.time() - start_time > self.termination_time):
            return True

        return False

    def conditional_terminate_on_unstable(self, unstable_flag):
        if unstable_flag:
            return True

        return False


    def save_single_deploy_data(self, data):
        """
        This should write to disk! (And just overwrite / append)
        """
        results_csv_file = os.path.join(self._logging_dir, "results.csv")

        if os.path.exists(results_csv_file):
            self._dataframe = pd.read_csv(results_csv_file, index_col=0)

        index = data['index']
        q0 = data["object_pose_start"]


        dfw = PushPlatePandasTemplate()
        dfw.set_value("date_time", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
        dfw.set_value("distance_to_goal", data["distance_to_goal"])
        dfw.set_value("termination_type", data["termination_type"])
        dfw.set_value("index", index)
        dfw.set_value("duration", data["duration"])
        dfw.set_value("initial_pos_x", q0[0])
        dfw.set_value("initial_pos_y", q0[1])
        dfw.set_value("initial_pos_z", q0[2])


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
        keys = ["index", "termination_type", "distance_to_goal", "times", "object_poses"]
        for key in keys:
            json_data[key] = data[key]

        json_filename = os.path.join(self._logging_dir, "data_%s.json" %(index_str))
        spartan_utils.save_to_json(json_data, json_filename)


    def create_new_config_file(self, index):
        sim_config_file = os.path.join(spartan_utils.getSpartanSourceDir(), "src/catkin_projects/drake_iiwa_sim/config/sim_config_externalcameras_manipulation.yaml")

        sim_config = spartan_utils.getDictFromYamlFilename(sim_config_file)
        
        q0 = sample_initial_pose(fixed_initial_pose=False)

        plate_dict = {'model': "ikea_dinera_plate_8in",
                      'q0': q0.tolist(),
                      'fixed': False,
                      'body_name': "ikea_dinera_plate_8in"}
        sim_config["instances"] = [plate_dict]

        q0 = np.array([0.29510, 0.23508, 0.1, 0.0, 0.0, 0.0])
        cube_dict = {'model': "foam_brick",
                     'q0': q0.tolist(),
                     'fixed': False,
                     'body_name': "base_link"}

        sim_config["instances"].append(cube_dict)

        index_str = str(index).zfill(5)

        sim_config_save_file = os.path.join(self._logging_dir, "sim_config_"+index_str+".yaml")
        spartan_utils.saveToYaml(sim_config, sim_config_save_file)
        return sim_config, sim_config_save_file




class PushPlatePandasTemplate(PandaDataFrameWrapper):
    columns = ['date_time',
               'distance_to_goal',
               'termination_type',
               'index',
               'duration',
               'initial_pos_x',
               'initial_pos_y',
               'initial_pos_z',
               ]

    def __init__(self):
        PandaDataFrameWrapper.__init__(self, PushPlatePandasTemplate.columns)



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
    
    rospy.init_node("push_plate_evaluator")
    network, readable_name = load_network()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return LSTMPositionAgent(network, observation_function, episode, config)

    logging_dir = os.path.join(spartan_utils.getSpartanSourceDir(),"sandbox",readable_name)
    evaluator = PushPlateEvaluator(create_imitation_agent, logging_dir = logging_dir)

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
    evaluator = PushPlateEvaluator(create_imitation_agent, logging_dir=logging_dir, convex_hull=convex_hull)

    if seed is not None:
        evaluator.set_random_seed(seed)

    num_deploys = 1
    evaluator.run_single_deploy(deploy_idx, rate=100, config=config) 


def run_one_mlp_position_agent(deploy_idx, seed=None, network_chkpt=None, logging_dir=None, training_poses_file=None):
    rospy.init_node("move_to_box_evaluator")
    
    network, readable_name = load_network_mlp_position(network_chkpt)
    network.unset_use_precomputed_features()
    network.unset_use_precomputed_descriptor_images()
    config = network._policy_net._config
    
    action_function, observation_function, episode = construct_imitation_episode(config)

    def create_imitation_agent():
        return MLPPositionAgent(network, observation_function, episode, config)


    logging_dir = os.path.join(logging_dir, readable_name)
    evaluator = PushPlateEvaluator(create_imitation_agent, logging_dir=logging_dir)

    if seed is not None:
        evaluator.set_random_seed(seed)

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
