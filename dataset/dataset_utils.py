import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.add_dense_correspondence_to_python_path()
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.dataset.dynamic_spartan_dataset import DynamicSpartanDataset

def build_spartan_dataset(logs_root_path):
    """
    This is just to keep the ancestral dataloader happy.
    The only thing that matters is the logs_root_path, I think. It should be an absolute path
    """
    mini_config_expanded = dict()
    single_object_scene_config = dict()
    single_object_scene_config["train"] = ["2019-04-10-22-12-43"] # Hack - doesn't need to change?
    single_object_scene_config["test"]  = ["2019-04-10-22-12-43"] # Hack - doesn't need to change?
    single_object_scene_config["object_id"] = "test_object"
    single_object_scene_dict = dict()
    single_object_scene_dict["new_object"] = single_object_scene_config
    mini_config_expanded["single_object"] = single_object_scene_dict
    multi_object_scene_config = dict()
    multi_object_scene_config["train"] = []
    multi_object_scene_config["test"] = []
    mini_config_expanded["multi_object"] = multi_object_scene_config
    mini_config_expanded["logs_root_path"] = logs_root_path
    
    return DynamicSpartanDataset(config_expanded=mini_config_expanded)


def get_object_starting_poses(dataset, # ImitationEpisodeDataset or ImitationEpisodeSequenceDataset
                              ): # return -> numpy.array shape [N,2] with N = # episodes
    object_poses = []
    for log_name in dataset.episodes.keys():
        episode = dataset.episodes[log_name]
        object_pose_trained = episode.sim_config_dict["instances"][0]["q0"]
        object_poses.append([object_pose_trained[0], object_pose_trained[1]])

    return object_poses
