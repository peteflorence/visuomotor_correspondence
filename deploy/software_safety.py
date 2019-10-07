import sys
import numpy.linalg as LA
import numpy as np

# spartan
import robot_msgs.msg
from spartan.utils import transformations
from spartan.utils import utils as spartan_utils

from imitation_agent.utils import utils as imitation_agent_utils

class SoftwareSafety(object):

	def __init__(self, config):
		self._config = config
		params = imitation_agent_utils.get_software_safety_params_from_config(config)
		self._translation_threshold = params["translation_threshold"]
		self._rotation_threshold = np.deg2rad(params["rotation_threshold_degrees"])

	def set_initial_goal(self, T_W_E, ## EE to world transform
	                     ):

		self._T_W_E_prev = T_W_E

	def kill(self):
		sys.exit(0)

	def determine_if_unsafe(self, msg):
		translation_vector = np.asarray([msg.xyz_point.point.x,  msg.xyz_point.point.y,  msg.xyz_point.point.z])

		quat_msg = msg.quaternion
		quat = np.array([quat_msg.w, quat_msg.x, quat_msg.y, quat_msg.z])

		T_W_E = transformations.quaternion_matrix(quat)
		T_W_E[:3, 3] = translation_vector

		dT = np.matmul(np.linalg.inv(T_W_E), self._T_W_E_prev)
		angle_axis = spartan_utils.angle_axis_from_rotation_matrix(dT[:3, :3])


		unsafe_commmand = False
		type_str = ""
		value = 0

		if np.linalg.norm(dT[:3, 3]) > self._translation_threshold:
			unsafe_commmand = True
			type_str = "TRANSLATION"
			value = np.linalg.norm(dT[:3, 3])
		elif np.linalg.norm(angle_axis) > self._rotation_threshold:
			unsafe_commmand = True
			type_str = "ROTATION"
			value = np.linalg.norm(angle_axis)

		self._T_W_E_prev = T_W_E

		if unsafe_commmand:
			print "SOFTWARE SAFETY MAX EXCEEDED: ", type_str
			print "DESIRED VALUE: ", value


		return unsafe_commmand


	def sys_exit_if_not_safe(self, msg): # CartesionGoalPoint msg
		"""
		If the goal msg is not safe, 
		then this class will take down the full Python stack
		via sys.exit(0).
		"""
		unsafe_commmand = self.determine_if_unsafe(msg)

		if unsafe_commmand:
			self.kill()





