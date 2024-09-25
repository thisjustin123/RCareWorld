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

    shoulder_angles = np.load("pyrcareworld/Test/shoulder_elbow_angles.npy")
    # Scramble
    shoulder_angles = np.random.permutation(shoulder_angles)
    # Take every Nth
    N = 50
    shoulder_angles = shoulder_angles[::N]

    for rot_vector in shoulder_angles:
        shoulder_x = rot_vector[0]
        shoulder_y = 180 - rot_vector[2]
        shoulder_z = rot_vector[1]
        elbow = rot_vector[3]

        print(shoulder_x, shoulder_y, shoulder_z, elbow)

        env.fk_step(
            shoulder_angles=[shoulder_x, shoulder_y, shoulder_z],
            elbow_angles=[elbow, 0, 0],
        )

    env.fk_end(save=True, visualize=False)

    # Read pyrcareworld/Test/evaluate_human_fk.npy
    as_np = np.load("pyrcareworld/Test/evaluate_human_fk.npy")
    # Demo: Can read points afterwards.
    env.cloud_manager.set_radius(0.3)
    env.cloud_manager.make_cloud(points=as_np, name="Human FK")

    for _ in range(300):
        env.step()
