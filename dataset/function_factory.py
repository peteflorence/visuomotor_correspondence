# system
import numpy as np
import functools
import torch

# spartan
import spartan.utils.utils as spartan_utils
from spartan.utils import constants
# imitation_agent
from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.utils import utils as imitation_agent_utils

import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.add_dense_correspondence_to_python_path()
from dense_correspondence.correspondence_tools import correspondence_finder


"""
OBSERVATIONS
"""


def get_translation_observation_x(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['translation']["x"]


def get_translation_observation_y(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['translation']["y"]


def get_translation_observation_z(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['translation']["z"]

def get_rpy_observation_roll(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['rpy']['roll']

def get_rpy_observation_pitch(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['rpy']['pitch']

def get_rpy_observation_yaw(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['rpy']['yaw']

def get_quaternion_observation_w(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['quaternion']['w']


def get_quaternion_observation_x(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['quaternion']['x']


def get_quaternion_observation_y(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['quaternion']['y']


def get_quaternion_observation_z(obs_action_data):
    return obs_action_data['observations']['ee_to_world']['quaternion']['z']


def get_gripper_observation_force(obs_action_data):
    return obs_action_data['observations']['gripper_state']['current_force']


def get_gripper_observation_speed(obs_action_data):
    return obs_action_data['observations']['gripper_state']['current_speed']


def get_gripper_observation_width(obs_action_data):
    return obs_action_data['observations']['gripper_state']['width']



def get_ee_to_world(data, # type dict entry from states.yaml
                    ): # type -> numpy.ndarray 4 x 4 homogeneous transform
    transform_dict = data['observations']['ee_to_world']
    transform = spartan_utils.homogenous_transform_from_dict(transform_dict)
    return transform


def get_transformed_end_effector_points(episode, ee_points, idx):
    """
    Returns N x 3 array
    :param idx:
    :type idx:
    :return:
    :rtype:
    """

    data = episode.get_entry(idx)
    T_EE_world = get_ee_to_world(data)
    ee_points_transformed = spartan_utils.apply_homogenous_transform_to_points(T_EE_world, ee_points.T).T

    return ee_points_transformed


# DEBUG-ONLY FUNCTIONS

# ---> static
def get_gt_starting_object_pose_x(obs_action_data):
    return obs_action_data['debug_observations']['object_starting_pose']['x']

def get_gt_starting_object_pose_y(obs_action_data):
    return obs_action_data['debug_observations']['object_starting_pose']['y']

def get_gt_starting_object_pose_z(obs_action_data):
    return obs_action_data['debug_observations']['object_starting_pose']['z']


# --> dynamic
def get_gt_dynamic_object_translation_x(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['position']['x']

def get_gt_dynamic_object_translation_y(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['position']['y']

def get_gt_dynamic_object_translation_z(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['position']['z']

def get_gt_dynamic_object_quaternion_w(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['quaternion']['w']

def get_gt_dynamic_object_quaternion_x(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['quaternion']['x']

def get_gt_dynamic_object_quaternion_y(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['quaternion']['y']

def get_gt_dynamic_object_quaternion_z(obs_action_data):
    return obs_action_data['observations']['object_pose_cheat_data']['quaternion']['z']



class ObservationFunctionFactory(object):
    """
    Helper class that constructs observation parsing functions from configs
    """

    @staticmethod
    def get_function(config,  # dict
                     ):  # type function
        """
        Returns a function that takes inputs (episode, idx)
        :param type:
        :type type:
        :param config:
        :type config:
        :return:
        :rtype:
        """

        return getattr(ObservationFunctionFactory, config["observation"]["type"])(config)

    @staticmethod
    def ee_position_history_observation(config=None,  # type dict
                                        ):
        """
        Returns history of EE positions (expressed in terms of 3 points on the end-effector)
        :param episode: ImitationEpisode object
        :type episode:
        :param idx:
        :type idx:
        :return: numpy.ndarray M x N x 3 where M = history length, N = num points (typically 3)
        :rtype:
        """

        # pre-processing
        ee_points_list = config['observation']['config']['ee_points']
        num_points = len(ee_points_list)

        ee_points = np.zeros([num_points, 3])
        for i in xrange(num_points):
            ee_points[i, :] = np.array(ee_points_list[i])

        observation_get_functions = []
        if config["use_gt_object_pose"]:
            observation_get_functions.append(get_gt_starting_object_pose_x)
            observation_get_functions.append(get_gt_starting_object_pose_y)
            observation_get_functions.append(get_gt_starting_object_pose_z)

        index_map = dict()
        # construct the actual function
        def func(episode, idx, **kwargs): # accepts arbitrary kwargs

            T_noise = kwargs['T_noise'] # noise transform (same for all at the moment)

            idx_for_history = np.array(config['observation']['config']['history'])
            idx_for_history += idx  # shift it to current

            # clip it so we don't spill off the ends
            idx_for_history = np.clip(idx_for_history, 0, len(episode) - 1)

            M = len(idx_for_history)
            N = ee_points.shape[0]

            ee_points_prev = np.zeros([M, N, 3])

            for i, idx_prev in enumerate(idx_for_history):
                T_W_C = episode.get_entry(idx_prev)["T_W_C"]
                T_W_C = np.matmul(T_W_C, T_noise) # apply noise
                ee_points_prev[i, :, :] = spartan_utils.apply_homogenous_transform_to_points(T_W_C, ee_points.T).T

            ee_points_prev = ee_points_prev.flatten()
            ee_points_prev = ImitationEpisode.convert_to_flattened_torch_tensor(ee_points_prev)

            obs = torch.FloatTensor([])
            obs = torch.cat((obs, ee_points_prev))

            data = episode.get_entry(idx)
            T_W_E = spartan_utils.homogenous_transform_from_dict(data['observations']["ee_to_world"])
            T_W_C = np.matmul(T_W_E, constants.T_E_cmd)
            T_W_C = np.matmul(T_W_C, T_noise) # apply noise

            T_W_Cnominal = imitation_agent_utils.get_T_W_Cnominal_from_config(config)
            T_Cnominal_C = np.matmul(np.linalg.inv(T_W_Cnominal), T_W_C)
            angle_axis_relative_to_nominal = spartan_utils.angle_axis_from_rotation_matrix(T_Cnominal_C[:3, :3])


            if "angle_axis_relative_to_nominal" in config["observation"]["config"]:
                if config["observation"]["config"]["angle_axis_relative_to_nominal"]["x"]:
                    obs = torch.cat((obs, torch.FloatTensor([angle_axis_relative_to_nominal[0]])))
                if config["observation"]["config"]["angle_axis_relative_to_nominal"]["y"]:
                    obs = torch.cat((obs, torch.FloatTensor([angle_axis_relative_to_nominal[1]])))
                if config["observation"]["config"]["angle_axis_relative_to_nominal"]["z"]:
                    obs = torch.cat((obs, torch.FloatTensor([angle_axis_relative_to_nominal[2]])))

            if config["use_gt_object_pose"]:
                obs_gt_pose = [get(data) for get in observation_get_functions]
                obs_gt_pose = torch.FloatTensor(obs_gt_pose)

                if config["project_pose_into_camera"]:
                    obs_gt_pose[2] = 0.0 # HACK: drop it onto table
                    camera_to_world = episode.get_camera_pose_matrix(config["camera_num"])
                    K = episode.get_K_matrix(config["camera_num"])
                    uv = correspondence_finder.pinhole_projection_world_to_image(obs_gt_pose, K, camera_to_world=camera_to_world)
                    obs_gt_pose_projected = torch.FloatTensor([uv[0],uv[1]]) 
                    obs = torch.cat((obs, obs_gt_pose_projected))
                else:
                    obs = torch.cat((obs, obs_gt_pose))

            if "use_dynamic_gt_object_pose" in config["observation"]["config"]:
                config_local = config["observation"]["config"]["use_dynamic_gt_object_pose"]
                object_pose_vec = []

                T_W_B = spartan_utils.homogenous_transform_from_dict(data['observations']['object_pose_cheat_data'])
                T_Bnominal_B = np.matmul(np.linalg.inv(constants.T_W_B_init), T_W_B)

                pos = T_W_B[:3, 3]
                angle_axis_relative_to_nominal = spartan_utils.angle_axis_from_rotation_matrix(T_Bnominal_B[:3, :3])

                if config_local["translation"]["x"]:
                    object_pose_vec.append(pos[0])
                if config_local["translation"]["y"]:
                    object_pose_vec.append(pos[1])
                if config_local["translation"]["z"]:
                    object_pose_vec.append(pos[2])

                if config_local["angle_axis_relative_to_nominal"]["x"]:
                    object_pose_vec.append(angle_axis_relative_to_nominal[0])

                if config_local["angle_axis_relative_to_nominal"]["y"]:
                    object_pose_vec.append(angle_axis_relative_to_nominal[1])

                if config_local["angle_axis_relative_to_nominal"]["z"]:
                    object_pose_vec.append(angle_axis_relative_to_nominal[2])

                obs = torch.cat((obs, torch.FloatTensor(object_pose_vec)))

            if "gt_object_points" in config["observation"]["config"]:

                T_W_B = spartan_utils.homogenous_transform_from_dict(data['observations']['object_pose_cheat_data'])

                for point in config["observation"]["config"]["gt_object_points"]:
                    point = spartan_utils.apply_homogenous_transform_to_points(T_W_B, np.asarray(point).T).T
                    point_torch = torch.from_numpy(point).float()

                    if config["observation"]["config"]["project_gt_object_points_into_camera"]:
                        camera_to_world = episode.get_camera_pose_matrix(config["camera_num"])
                        K = episode.get_K_matrix(config["camera_num"])
                        uv = correspondence_finder.pinhole_projection_world_to_image(point, K, camera_to_world=camera_to_world)
                        obs_gt_pose_projected = torch.FloatTensor([uv[0],uv[1]]) 
                        obs = torch.cat((obs, obs_gt_pose_projected))
                    else:
                        obs = torch.cat((obs, point_torch))


            # need to keep gripper width last
            # or refactor noise augmentation
            if config["observation"]["config"]["gripper"]["width"]:
                obs = torch.cat((obs, torch.FloatTensor([get_gripper_observation_width(data)])))

            return obs

        return func

    @staticmethod
    def observation_from_config(config):

        obs_config = config["observation"]["config"]

        observation_get_functions = []

        if obs_config["translation"]["x"]:
            observation_get_functions.append(get_translation_observation_x)

        if obs_config["translation"]["y"]:
            observation_get_functions.append(get_translation_observation_y)

        if obs_config["translation"]["z"]:
            observation_get_functions.append(get_translation_observation_z)

        if obs_config["rpy"]["roll"]:
            observation_get_functions.append(get_rpy_observation_roll)

        if obs_config["rpy"]["pitch"]:
            observation_get_functions.append(get_rpy_observation_pitch)

        if obs_config["rpy"]["yaw"]:
            observation_get_functions.append(get_rpy_observation_yaw)

        if obs_config["quaternion"]["w"]:
            observation_get_functions.append(get_quaternion_observation_w)

        if obs_config["quaternion"]["x"]:
            observation_get_functions.append(get_quaternion_observation_x)

        if obs_config["quaternion"]["y"]:
            observation_get_functions.append(get_quaternion_observation_y)

        if obs_config["quaternion"]["z"]:
            observation_get_functions.append(get_quaternion_observation_z)

        if obs_config["gripper"]["force"]:
            observation_get_functions.append(get_gripper_observation_force)

        if obs_config["gripper"]["speed"]:
            observation_get_functions.append(get_gripper_observation_speed)

        if obs_config["gripper"]["width"]:
            observation_get_functions.append(get_gripper_observation_width)

        try:
            # ANALYSIS-ONLY USE IN SIM, DYNAMIC
            if obs_config["object_pose_cheat_data"]["translation"]:
                observation_get_functions.append(get_gt_dynamic_object_translation_x)
                observation_get_functions.append(get_gt_dynamic_object_translation_y)
                observation_get_functions.append(get_gt_dynamic_object_translation_z)

            if obs_config["object_pose_cheat_data"]["quaternion"]:
                observation_get_functions.append(get_gt_dynamic_object_quaternion_w)
                observation_get_functions.append(get_gt_dynamic_object_quaternion_x)
                observation_get_functions.append(get_gt_dynamic_object_quaternion_y)
                observation_get_functions.append(get_gt_dynamic_object_quaternion_z)


        # ANALYSIS-ONLY USE IN SIM, STATIC
            if config["use_gt_object_pose"]:
                observation_get_functions.append(get_gt_starting_object_pose_x)
                observation_get_functions.append(get_gt_starting_object_pose_y)
                observation_get_functions.append(get_gt_starting_object_pose_z)

        except KeyError:
            print "assuming this is an old network..."




        def func(episode, idx):

            obs_action_data = episode.get_entry(idx)
            obs = [get(obs_action_data) for get in observation_get_functions]

            if config["project_pose_into_camera"]:
                world_pos = obs[-3:]
                world_pos[2] = 0.0 # HACK: drop it onto table
                camera_to_world = episode.get_camera_pose_matrix(config["camera_num"])
                K = episode.get_K_matrix(config["camera_num"])

                uv = correspondence_finder.pinhole_projection_world_to_image(world_pos, K, camera_to_world=camera_to_world)
                del obs[-3:]
                obs.append(uv[0])
                obs.append(uv[1])


            return obs

        return func


"""
ACTIONS
"""


def get_translation_setpoint_x(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['position']['x']

def get_translation_setpoint_y(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['position']['y']

def get_translation_setpoint_z(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['position']['z']

def get_rpy_setpoint_roll(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['rpy']['roll']

def get_rpy_setpoint_pitch(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['rpy']['pitch']

def get_rpy_setpoint_yaw(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['rpy']['yaw']

def get_quaternion_setpoint_w(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['quaternion']['w']

def get_quaternion_setpoint_x(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['quaternion']['x']

def get_quaternion_setpoint_y(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['quaternion']['y']

def get_quaternion_setpoint_z(obs_action_data):
    return obs_action_data['actions_offset']['ee_setpoint']['quaternion']['z']

def get_gripper_setpoint_width(obs_action_data):
    return obs_action_data['actions_offset']['gripper_setpoint']['width']

def get_translation_delta_x(obs_action_data):
    return obs_action_data["actions_offset"]['delta_to_EE_frame']['translation_world_frame'][0]

def get_translation_delta_y(obs_action_data):
    return obs_action_data["actions_offset"]['delta_to_EE_frame']['translation_world_frame'][1]

def get_translation_delta_z(obs_action_data):
    return obs_action_data["actions_offset"]['delta_to_EE_frame']['translation_world_frame'][2]

def get_angle_axis_delta_x(obs_action_data):
    return obs_action_data["actions_offset"]["delta_to_EE_frame"]["angle_axis_world_frame"][0]

def get_angle_axis_delta_y(obs_action_data):
    return obs_action_data["actions_offset"]["delta_to_EE_frame"]["angle_axis_world_frame"][1]

def get_angle_axis_delta_z(obs_action_data):
    return obs_action_data["actions_offset"]["delta_to_EE_frame"]["angle_axis_world_frame"][2]


"""
COMMAND
"""


def get_ee_linear_velocity_cmd_expressed_in_world(data):
    """

    :param data: Single entry in the states.yaml file
    :type data:
    :return:
    :rtype:
    """
    d = data['control']['linear_velocity_cmd_expressed_in_world']
    return np.array([d['x'], d['y'], d['z']])


def get_ee_angular_velocity_cmd_expressed_in_world(data):
    """

    :param data: Single entry in the states.yaml file
    :type data:
    :return:
    :rtype: np.array size 3
    """

    d = data['control']['angular_velocity_cmd_expressed_in_world']
    return np.array([d['x'], d['y'], d['z']])


class ActionFunctionFactory(object):
    """
    Helper class that constructs action parsing functions from configs
    """

    @staticmethod
    def get_function(config,  # dict
                     ):  # type function
        """
        Returns a function that takes inputs (episode, idx)
        :param type:
        :type type:
        :param config:
        :type config:
        :return:
        :rtype:
        """

        return getattr(ActionFunctionFactory, config["action"]["type"])(config)

    @staticmethod
    def ee_velocity_cmd_world_frame(config=None,  # type dict
                                    ):  # type np.ndarray 6 x 1, (ang_vel, lin_vel)
        """
        Returns commanded ee velocity in world frame
        :param episode: ImitationEpisode object
        :type episode:
        :param idx:
        :type idx:
        :return:
        :rtype:
        """

        def func(episode, idx):
            data = episode.get_entry(idx)
            lin_vel = get_ee_linear_velocity_cmd_expressed_in_world(data)
            ang_vel = get_ee_angular_velocity_cmd_expressed_in_world(data)

            ee_velocity = np.concatenate((ang_vel, lin_vel))

            ee_velocity = ImitationEpisode.convert_to_flattened_torch_tensor(ee_velocity)

            return ee_velocity

        return func

    @staticmethod
    def action_from_config(config):

        action_config = config['action']['config']

        def func(episode, idx, **kwargs):
            T_noise = kwargs["T_noise"]
            data = episode.get_entry(idx)
            T_W_C = data["T_W_C"]
            T_W_C = np.matmul(T_W_C, T_noise)
            T_W_C_cmd = data["actions_offset"]["T_W_C"]

            # given T_W_C and T_W_C_cmd we can extract everything we want


            # compute delta translation and rotation
            T_C_cmd = np.matmul(np.linalg.inv(T_W_C), T_W_C_cmd)
            translation_delta = T_C_cmd[:3, 3]
            translation_delta_world_frame = np.dot(T_W_C[:3, :3], translation_delta)
            angle_axis_delta = spartan_utils.angle_axis_from_rotation_matrix(T_C_cmd[:3, :3])

            # global
            translation = T_W_C_cmd[:3, 3]

            # euler, global
            rpy = spartan_utils.transformations.euler_from_matrix(T_W_C_cmd)

            T_W_Cnominal = imitation_agent_utils.get_T_W_Cnominal_from_config(config)
            T_Cnominal_C_cmd = np.matmul(np.linalg.inv(T_W_Cnominal), T_W_C_cmd)
            angle_axis_relative_to_nominal = spartan_utils.angle_axis_from_rotation_matrix(T_Cnominal_C_cmd[:3, :3])

            # extract the action
            action = []

            if action_config["translation"]["x"]:
                action.append(translation[0])

            if action_config["translation"]["y"]:
                action.append(translation[1])

            if action_config["translation"]["z"]:
                action.append(translation[2])

            if action_config["translation_delta"]["x"]:
                action.append(translation_delta[0])

            if action_config["translation_delta"]["y"]:
                action.append(translation_delta[1])

            if action_config["translation_delta"]["z"]:
                action.append(translation_delta[2])

            if action_config["translation_delta_world_frame"]["x"]:
                action.append(translation_delta_world_frame[0])

            if action_config["translation_delta_world_frame"]["y"]:
                action.append(translation_delta_world_frame[1])

            if action_config["translation_delta_world_frame"]["z"]:
                action.append(translation_delta_world_frame[2])

            if action_config["rpy"]["roll"]:
                action.append(rpy[0])

            if action_config["rpy"]["pitch"]:
                action.append(rpy[1])

            if action_config["rpy"]["yaw"]:
                action.append(rpy[2])

            if action_config["angle_axis_delta"]["x"]:
                action.append(angle_axis_delta[0])

            if action_config["angle_axis_delta"]["z"]:
                action.append(angle_axis_delta[1])

            if action_config["angle_axis_delta"]["z"]:
                action.append(angle_axis_delta[2])

            if "angle_axis_relative_to_nominal" in config["action"]["config"]:
                if config["action"]["config"]["angle_axis_relative_to_nominal"]["x"]:
                    action.append(angle_axis_relative_to_nominal[0])

                if config["action"]["config"]["angle_axis_relative_to_nominal"]["y"]:
                    action.append(angle_axis_relative_to_nominal[1])

                if config["action"]["config"]["angle_axis_relative_to_nominal"]["z"]:
                    action.append(angle_axis_relative_to_nominal[2])


            if action_config['gripper']['width']:
                action.append(data['actions_offset']['gripper_setpoint']['width'])

            return action

        return func
