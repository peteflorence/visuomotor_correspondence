import os
import attr

@attr.s
class SingleEpisodeDirectoryStructure(object):

    path_to_processed_dir = attr.ib()
        
    @property
    def state_file(self):
        return os.path.join(self.path_to_processed_dir, "states.json")

    @property
    def scene_name(self):
        return os.path.dirname(self.path_to_processed_dir)

    def expand_relative_path(self, rel_path):
        return os.path.join(self.path_to_processed_dir, rel_path)

    def get_camera_info_yaml(self, camera_num):
        return os.path.join(self.path_to_processed_dir, "images_camera_"+str(camera_num), "camera_info.yaml")

    @property
    def sim_config_file(self):
    	raw_log_dir = os.path.join(self.scene_name, "raw")
    	return os.path.join(raw_log_dir, "sim_config.yaml")