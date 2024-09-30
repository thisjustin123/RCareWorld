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
    point_cloud = np.load("pyrcareworld/Test/.out_points/9_limit_3_Learned.npy")
    env = RhiVisEnv(
        executable_file="@editor",
        start_pos=START_POS,
        end_pos=END_POS,
        handoff_pos=HANDOFF_POS,
        human_end_pos=HUMAN_END_POS,
        obj_grab_offset=OBJ_GRASP_OFFSET,
        point_cloud=point_cloud,
    )

    cloud_aligner = env.create_object(id=909, name="cloud_aligner", is_in_scene=True)
    align_pos = np.array(cloud_aligner.getPosition())
    align_rot = np.array(cloud_aligner.getRotation())
    env.rotate_point_cloud(x=align_rot[0], y=align_rot[1], z=align_rot[2])
    env.scale_point_cloud(x=1, y=1, z=-1)
    env.reset_midpoint(align_pos)
    env.make_cloud(name="instatest", radius=0.6)

    env.demo_bathing()
