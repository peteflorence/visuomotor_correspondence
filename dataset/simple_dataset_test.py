from __future__ import print_function

import spartan.utils.utils as spartan_utils
import os

from imitation_agent.dataset.imitation_episode_dataset import ImitationEpisodeDataset
from imitation_agent.dataset.imitation_episode_sequence_dataset import ImitationEpisodeSequenceDataset
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory



def test_sequence_dataset():
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()

    logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_then_flip_7_22")
    logs_config_yaml = os.path.join(spartan_source_dir,
                                    "modules/imitation_agent/config/task/move_to_box_then_flip_7_22_small.yaml")

    logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

    config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence.yaml")
    config = spartan_utils.getDictFromYamlFilename(config_yaml)

    config["use_vision"] = False

    obs_function = ObservationFunctionFactory.get_function(config)
    action_function = ActionFunctionFactory.action_from_config(config)

    print(logs_dir_path)

    dataset = ImitationEpisodeSequenceDataset(logs_dir_path, logs_config, config, action_function, obs_function)

    print(len(dataset))

    for i in range(10):
        data = dataset[i]
        print(type(data))
        print(data.keys())
        print(data['images'].shape)
        print("observation.shape", data['observations'].shape)
        print("action.shape", data['actions'].shape)

def test_stateless_dataset():
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()

    logs_dir_path = os.path.join(data_dir, "pdc/imitation/real_push_box")
    logs_config_yaml = os.path.join(spartan_source_dir,
                                    "modules/imitation_agent/config/task/real_push_box.yaml")

    logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

    config_yaml = os.path.join(imitation_src_dir, "experiments", "lstm_grab_plate.yaml")
    config = spartan_utils.getDictFromYamlFilename(config_yaml)


    config["use_vision"] = False
    config["filtering"]["filter_no_movement"] = False

    action_function = ActionFunctionFactory.action_from_config(config)

    obs_function = ObservationFunctionFactory.get_function(config)

    dataset = ImitationEpisodeDataset(logs_dir_path,
                                      logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)

    
    print(len(dataset), "is len")


    for i in range(10):
        data = dataset[i]
        print("\n\n")
        print(type(data))
        print(data.keys())
        print(data['image'].shape)
        print("observation.shape", data['observation'].shape)
        print("action.shape", data['action'].shape)

if __name__ == "__main__":
    #test_sequence_dataset()
    test_stateless_dataset()
