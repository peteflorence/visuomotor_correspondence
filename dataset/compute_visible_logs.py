from __future__ import print_function

import spartan.utils.utils as spartan_utils
import os

from imitation_agent.utils.visibility_checker import sugar_box_visible


spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")

# LOGS_DIR_PATH = "/home/manuelli/data/pdc/imitation/logs_flip_sugar/move_to_box_then_flip_0716"
# LOGS_CONFIG_FILE = os.path.join(imitation_src_dir, 'config/task/move_to_box_then_flip_0716.yaml')

LOGS_DIR_PATH = "/home/peteflo/data/pdc/imitation/move_to_box_se2"
LOGS_CONFIG_FILE = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_se2_box_in_frame.yaml")

LOGS_CONFIG = spartan_utils.getDictFromYamlFilename(LOGS_CONFIG_FILE)

# full image (i.e., for DD)
#URANGE = [0, 640]
#VRANGE = [0, 480]

#settings for E2E
#URANGE = [80, 640-80] # centered
#URANGE = [0, 480]   # for camera_right, i.e. CAMERA_NUM = 1
URANGE = [160, 640] # for camera_left,  i.e. CAMERA_NUM = 0

VRANGE = [0, 480]

CAMERA_NUM = 0

def run():

    good_logs = []
    for log_name in LOGS_CONFIG["logs"]:
        if not sugar_box_visible(log_name=log_name, logs_dir_path=LOGS_DIR_PATH, camera_num=CAMERA_NUM, urange=URANGE, vrange=VRANGE):
            print("not visible in log name: %s" %(log_name))
        else:
            good_logs.append(log_name)

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


