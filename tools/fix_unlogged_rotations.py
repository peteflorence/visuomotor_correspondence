import os
import sys
import spartan.utils.utils as spartan_utils
from spartan.utils import transformations
import numpy as np

import yaml
from yaml import CLoader
import json


spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_then_flip_0716")

logs_config_yaml = os.path.join(spartan_source_dir, "modules/imitation_agent/config/task/move_to_box_then_flip_0716.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

ee_tf_above_table = np.asarray([[-0.01389096,  0.38503428,  0.92279773,  0.51080192],
                                [ 0.04755021,  0.92209702, -0.38402613,  0.01549765],
                                [-0.99877226,  0.03854474, -0.03111728,  0.50188255],
                                [ 0.,          0.,          0.,          1.,       ]])


import matplotlib.pyplot as plt



def quat_to_euler(quat):
    quat = np.asarray([quat["w"],quat["x"],quat["y"],quat["z"]])
    x = transformations.quaternion_matrix(quat)
    T_delta = np.dot(x,np.linalg.inv(ee_tf_above_table))
    euler = transformations.euler_from_matrix(T_delta[0:3,0:3], 'syxz')
    return euler

for log_name in logs_config["logs"]:
    print log_name

    rolls = []
    pitches = []
    yaws = []

    rolls_before =[]
    pitches_before = []
    yaws_before = []

    rolls_measured = []



    states_json = os.path.join(logs_dir_path,log_name,"processed","states.json")
    states_yaml = os.path.join(logs_dir_path,log_name,"processed","states.yaml")
    states = json.load(file(states_json))
    for i in range(len(states)):
        i = str(i)
        quat_cmd = states[i]["actions"]["ee_setpoint"]["quaternion"]
        euler_cmd = quat_to_euler(quat_cmd)

        pitches.append(euler_cmd[0])
        rolls.append(euler_cmd[1])
        yaws.append(euler_cmd[2])

        rolls_before.append(states[i]["actions"]["ee_setpoint"]["rpy"]["roll"]*1.0)
        pitches_before.append(states[i]["actions"]["ee_setpoint"]["rpy"]["pitch"]*1.0)
        yaws_before.append(states[i]["actions"]["ee_setpoint"]["rpy"]["yaw"]*1.0)

        print euler_cmd
        states[i]["actions"]["ee_setpoint"]["rpy"]["pitch"] = euler_cmd[0]
        states[i]["actions"]["ee_setpoint"]["rpy"]["roll"] = euler_cmd[1]
        states[i]["actions"]["ee_setpoint"]["rpy"]["yaw"] = euler_cmd[2]

        quat_measured = states[i]["observations"]["ee_to_world"]["quaternion"]
        euler_measured = quat_to_euler(quat_measured)

        print euler_measured

        states[i]["observations"]["ee_to_world"]["rpy"] = dict()
        states[i]["observations"]["ee_to_world"]["rpy"]["pitch"] = euler_measured[0]
        states[i]["observations"]["ee_to_world"]["rpy"]["roll"] = euler_measured[1]
        states[i]["observations"]["ee_to_world"]["rpy"]["yaw"] = euler_measured[2]

        rolls_measured.append(euler_measured[1])


    [rolls_before, pitches_before, yaws_before] = [np.asarray(x) for x in [rolls_before, pitches_before, yaws_before]]



    rolls = np.asarray(rolls)
    pitches = np.asarray(pitches)
    yaws = np.asarray(yaws)

    t = np.arange(0,rolls.shape[0],1)

    # plt.plot(t,rolls_measured,label="rolls_measured")
    # plt.legend()
    # plt.show()

    # plt.plot(t,rolls_before,label="rolls_before")
    # plt.plot(t,pitches_before,label="pitches_before")
    # plt.plot(t,yaws_before,label="yaws_before")
    # plt.legend()
    # plt.show()


    # plt.plot(t,rolls,label="roll")
    # plt.plot(t,pitches,label="pitch")
    # plt.plot(t,yaws,label="yaws")
    # plt.legend()
    # plt.show()


    print "writing!, ", states_yaml, states_json

    with open(states_json, "w") as json_out:
        json.dump(states, json_out)
        #yaml.dump(states, yaml_out, default_flow_style=False)
