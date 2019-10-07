from __future__ import print_function

import spartan.utils.utils as spartan_utils
from spartan.utils import transformations
import os

from imitation_agent.dataset.imitation_episode import ImitationEpisode
from imitation_agent.dataset.imitation_episode_dataset import ImitationEpisodeDataset
from imitation_agent.dataset.imitation_episode_sequence_dataset import ImitationEpisodeSequenceDataset
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.utils.visibility_utils import check_sugar_box_in_frame


USE_STATIC = False


def sugar_box_visible(logs_dir_path, log_name, camera_num, urange=None, vrange=None):
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    logs_config = {"logs": [log_name],
                   "test_logs": [log_name]}

    
    config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence.yaml")
    config = spartan_utils.getDictFromYamlFilename(config_yaml)
    config["filtering"]["filter_no_movement"] = False
    config["use_gt_object_pose"] = True
    action_function = ActionFunctionFactory.action_from_config(config)

    obs_function = ObservationFunctionFactory.get_function(config)

    dataset = ImitationEpisodeDataset(logs_dir_path,
                                      logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)

    episode = dataset.episodes[log_name]
    T_W_camera = episode.get_camera_pose_matrix(camera_num)
    K = episode.get_K_matrix(camera_num)

    for idx in xrange(0, len(episode)):
        entry = episode.get_entry(idx)

        # if static
        if USE_STATIC:
            static_rotation_only = transformations.euler_matrix(0.0, 0.0, 1.57, axes='sxyz')
            static_rotation_only[0,3] = entry['debug_observations']['object_starting_pose']['x']
            static_rotation_only[1,3] = entry['debug_observations']['object_starting_pose']['y']
            static_rotation_only[2,3] = 2.00476981e-02
            T_W_B = static_rotation_only

        else:
            T_W_B = spartan_utils.homogenous_transform_from_dict(entry['observations']['object_pose_cheat_data'])
        
        in_frame = check_sugar_box_in_frame(K, T_W_camera=T_W_camera, T_W_B=T_W_B, urange=urange, vrange=vrange)

        if not in_frame:
            print("sugar box not in frame at idx %d. image file is" %(idx) )
            return False


    # print("all %d images had box in frame" %(len(episode)))
    return True





if __name__ == "__main__":

    logs_dir_path = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716"
    log_name = "2019-07-16-02-56-04"
    camera_num = 0
    sugar_box_visible(logs_dir_path, log_name, camera_num)