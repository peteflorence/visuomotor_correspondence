from imitation_agent.imitation_episode import ImitationEpisodeConfig, ImitationEpisode
import numpy as np

def test_imitation_episode():
    path_to_processed_dir = "/home/manuelli/spartan/data_volume/pdc/imitation/logs/2019-06-12-23-04-40/processed"
    indices_for_previous_ee_positions = np.array([0, -1, -2, -4])
    ee_points = np.zeros([3,3])

    config = ImitationEpisodeConfig(path_to_processed_dir=path_to_processed_dir, indices_for_previous_ee_positions=indices_for_previous_ee_positions)
