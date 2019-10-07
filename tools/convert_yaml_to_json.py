import os
import sys
import spartan.utils.utils as spartan_utils

import yaml
from yaml import CLoader
import json

# MINITEST
# #states_yaml = "states.yaml"
# states_yaml = "/home/peteflo/data/pdc/imitation/logs_sugar/2019-06-25-16-15-20/processed/states.yaml"
# states_json = "states.json"
# with open(states_yaml, 'r') as yaml_in, open(states_json, "w") as json_out:
#         yaml_object = yaml.load(yaml_in, Loader=CLoader) # yaml_object will be a list or a dict
#         json.dump(yaml_object, json_out)
#         print "wrote to json", states_json

# hey = json.load(file(states_json))
# print hey
# print hey["0"]["actions"]["ee_setpoint"]["position"]["x"]
# print type(hey["0"]["actions"]["ee_setpoint"]["position"]["x"])

# #
# print "quit"
# sys.exit(0)



spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_trunc_gauss_wait")

logs_config_yaml = os.path.join(spartan_source_dir, "modules/imitation_agent/config/task/move_to_box_trunc_gauss_wait.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

for log_name in logs_config["logs"]:
    states_yaml = os.path.join(logs_dir_path,log_name,"processed","states.yaml")
    states_json = os.path.join(logs_dir_path,log_name,"processed","states.json")

    with open(states_yaml, 'r') as yaml_in, open(states_json, "w") as json_out:
        yaml_object = yaml.load(yaml_in, Loader=CLoader) # yaml_object will be a list or a dict
        json.dump(yaml_object, json_out)
        print "wrote to json", states_json
