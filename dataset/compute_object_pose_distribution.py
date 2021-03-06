from __future__ import print_function

import spartan.utils.utils as spartan_utils
import os

from imitation_agent.dataset.directory_structure import SingleEpisodeDirectoryStructure



spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")

LOGS_DIR_PATH = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716"
LOGS_CONFIG_FILE = os.path.join(imitation_src_dir, 'config/task/move_to_box_then_flip_0716.yaml')

# LOGS_DIR_PATH = "/home/manuelli/data_ssd/imitation/logs/push_box"
# LOGS_CONFIG_FILE = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/push_box.yaml")

LOGS_CONFIG = spartan_utils.getDictFromYamlFilename(LOGS_CONFIG_FILE)


def run():

    good_logs = []
    box_pos_list = []
    for log_name in LOGS_CONFIG["logs"]:
        ds = SingleEpisodeDirectoryStructure(os.path.join(LOGS_DIR_PATH, log_name, "processed"))
        sim_config = spartan_utils.getDictFromYamlFilename(ds.sim_config_file)
        state_file = ds.state_file
        state_dict = spartan_utils.read_json(ds.state_file)
        state_dict[0]["object_pose_chea"]
        sugar_box_data = sim_config["instances"][0]

    good_test_logs = []
    for log_name in LOGS_CONFIG["test_logs"]:
        if not sugar_box_visible(log_name=log_name, logs_dir_path=LOGS_DIR_PATH, camera_num=CAMERA_NUM, urange=URANGE, vrange=VRANGE):
            print("not visible in log name: %s" %(log_name))
        else:
            good_test_logs.append(log_name)



    logs_config_filtered = {"logs": good_logs, "test_logs": good_test_logs}
    spartan_utils.saveToYaml(logs_config_filtered, "logs_config.yaml")

    print("\n\n-----------------")
    print("num logs: %d, after filtering: %d" %(len(LOGS_CONFIG["logs"]), len(good_logs)))
    print("num test logs: %d, after filtering: %d" % (len(LOGS_CONFIG["test_logs"]), len(good_test_logs)))

if __name__=="__main__":
    run()


