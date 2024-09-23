from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
import random
import numpy as np

from pyrcareworld.agents.robot import Robot
from pyrcareworld.envs.rhi_vis_env import RhiVisEnv

if __name__ == "__main__":
    env = RhiVisEnv(
        executable_file="@editor",
    )
    for _ in range(50):
        shoulder_x = random.random() * 90 + 90
        shoulder_y = random.random() * 90
        shoulder_z = random.random() * 90
        elbow = random.random() * 90

        env.fk_step(
            shoulder_angles=[shoulder_x, shoulder_y, shoulder_z],
            elbow_angles=[elbow, 0, 0],
        )

    env.fk_vis_and_close(save=True)
