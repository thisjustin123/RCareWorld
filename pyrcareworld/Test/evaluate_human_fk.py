from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
import random
import numpy as np

from pyrcareworld.agents.robot import Robot
from pyrcareworld.envs.rhi_vis_env import RhiVisEnv

CLOUD_OFFSET = np.array([0.34, 0.45, -0.03])
INPUT_ANGLES = "pyrcareworld/Test/ziang_2_8_limit.npy"
OUTPUT_FILE = "pyrcareworld/Test/evaluate_human_fk.npy"
COMPUTE_NEW_FK = True

if __name__ == "__main__":
    env = RhiVisEnv(
        executable_file="@editor",
    )

    shoulder_angles = np.load(INPUT_ANGLES)
    # Scramble
    shoulder_angles = np.random.permutation(shoulder_angles)
    # Take every Nth
    N = 10
    shoulder_angles = shoulder_angles[::N]

    if COMPUTE_NEW_FK:
        for rot_vector in shoulder_angles:
            # 1, 180 - 2, 0 is close but rotation is like flipped 180 on some axis.
            shoulder_x = rot_vector[1]
            shoulder_y = 180 - rot_vector[2]
            shoulder_z = rot_vector[0]
            elbow = rot_vector[3]

            print(shoulder_x, shoulder_y, shoulder_z, elbow)

            env.fk_step(
                shoulder_angles=[shoulder_x, shoulder_y, shoulder_z],
                elbow_angles=[elbow, 0, 0],
            )
        env.fk_end(save=True, visualize=False)

    # Read pyrcareworld/Test/evaluate_human_fk.npy
    as_np = np.load(OUTPUT_FILE)
    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    as_np = np.matmul(rotation_matrix, as_np.T).T
    rotation_matrix_z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    as_np = np.matmul(rotation_matrix_z, as_np.T).T
    as_np[:, 2] = -as_np[:, 2]
    theta = np.radians(30)  # Convert 30 degrees to radians
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    as_np = np.matmul(rotation_matrix, as_np.T).T

    as_np += CLOUD_OFFSET

    # Demo: Can read points afterwards.
    env.cloud_manager.set_radius(0.7)
    env.cloud_manager.make_cloud(points=as_np, name="Human FK")

    for _ in range(999999):
        env.step()
