from __future__ import print_function

import spartan.utils.utils as spartan_utils
import os
import numpy as np
import argparse

# imitation_agent
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.dataset.imitation_episode_sequence_dataset import ImitationEpisodeSequenceDataset
from imitation_agent.dataset.imitation_episode_dataset import ImitationEpisodeDataset
from imitation_agent.training import train_utils
from imitation_agent.dataset import dataset_utils

spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")

LOGS_DIR_PATH = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716"
LOGS_CONFIG_FILE = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_then_flip_0716_box_in_frame.yaml")

# LOGS_DIR_PATH = "/home/manuelli/data_ssd/imitation/logs/push_box"
# LOGS_CONFIG_FILE = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/push_box.yaml")

LOGS_CONFIG = spartan_utils.getDictFromYamlFilename(LOGS_CONFIG_FILE)
NUM_LOGS = 100
LOGS_CONFIG_DOWNSAMPLED = train_utils.deterministic_downsample(LOGS_CONFIG, NUM_LOGS)

CONFIG_YAML = os.path.join(imitation_src_dir, "experiments", "05", "05_lstm_sequence.yaml")
CONFIG = spartan_utils.getDictFromYamlFilename(CONFIG_YAML)

obs_function = ObservationFunctionFactory.get_function(CONFIG)
action_function = ActionFunctionFactory.action_from_config(CONFIG)
dataset = ImitationEpisodeDataset(LOGS_DIR_PATH, LOGS_CONFIG_DOWNSAMPLED, CONFIG,
                                  action_function=action_function,
                                  observation_function=obs_function)

def run(save_file=None):

    # just xy
    object_poses = dataset_utils.get_object_starting_poses(dataset)
    object_poses = np.array(object_poses)

    if save_file is None:
        save_file = "object_poses.npy"

    np.save(save_file, object_poses)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file",
        "-save_file",
        dest="save_file",
        required=False,
        help="(optional) location to save output"
    )

    args = parser.parse_args()
    run(save_file=args.save_file)


