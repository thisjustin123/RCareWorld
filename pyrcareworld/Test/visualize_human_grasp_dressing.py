from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
import random
import numpy as np

from pyrcareworld.agents.robot import Robot
from pyrcareworld.envs.rhi_vis_env import RhiVisEnv
from pyrcareworld.envs.rhi_vis_env import rotate_matrix


def lerp(a, b, t):
    return a + (b - a) * t


START_POS = [0.231999993, 1.16499996, -0.75]
END_POS = [0.573000014, 0.772000015, -0.75]
HANDOFF_POS = [0.128999993, 0.486000001, -0.47]
HUMAN_END_POS = [0.0839999989, 0.307999998, -0.503000021]
OBJ_GRASP_OFFSET = [0, -0.15, 0]

# To be used with the Point Cloud Viz scene.
if __name__ == "__main__":
    env = RhiVisEnv(
        executable_file="@editor",
        start_pos=START_POS,
        end_pos=END_POS,
        handoff_pos=HANDOFF_POS,
        human_end_pos=HUMAN_END_POS,
        obj_grab_offset=OBJ_GRASP_OFFSET,
        point_cloud_path="pyrcareworld/Test/evaluate_human_fk_good1.npy",
    )
    env.demo_dressing()
