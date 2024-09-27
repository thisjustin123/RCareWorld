from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
import random
from scipy.spatial.transform import Rotation as R
import numpy as np

from pyrcareworld.agents.robot import Robot
from pyrcareworld.envs.rhi_vis_env import RhiVisEnv


def rotation_matrix_x(angle):
    angle = np.radians(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def rotation_matrix_y(angle):
    angle = np.radians(angle)
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


def rotmat2euler(rot_matrix, seq="ZXY"):
    rotation = R.from_matrix(rot_matrix)
    return rotation.as_euler(seq, degrees=True)


CLOUD_OFFSET = np.array([0.34, 0.45, -0.03]) + np.array(
    [0.143999994, 0.213, -0.651000023]
)
INPUT_ANGLES = "pyrcareworld/Test/shoulder_elbow_angles.npy"
OUTPUT_FILE = "pyrcareworld/Test/shoulder_elbow_points.npy"
COMPUTE_NEW_FK = True
REALIGN_FILES = [
    "pyrcareworld/Test/5_limit_2.npy",
    "pyrcareworld/Test/ziang_2_8_limit.npy",
]

if __name__ == "__main__":
    env = RhiVisEnv(executable_file="@editor", output_file=OUTPUT_FILE)

    shoulder_angles = np.load(INPUT_ANGLES)
    # Scramble
    shoulder_angles = np.random.permutation(shoulder_angles)
    # Take every Nth
    N = 30
    shoulder_angles = shoulder_angles[::N]

    if COMPUTE_NEW_FK:
        for rot_vector in shoulder_angles:
            if INPUT_ANGLES in REALIGN_FILES:
                shoulder_aa, shoulder_fe, shoulder_rot, elbow_flexion = rot_vector
                shoulder_aa = -shoulder_aa
                shoulder_rot -= 90
                shoulder_rot = -shoulder_rot
                local_rot_mat = (
                    rotation_matrix_y(90)
                    @ rotation_matrix_x(shoulder_aa)
                    @ rotation_matrix_y(shoulder_fe)
                    @ rotation_matrix_x(shoulder_rot)
                )
                transformed_angles = rotmat2euler(local_rot_mat, seq="YZX")
                shoulder_x = transformed_angles[0] - 90
                shoulder_y = transformed_angles[1]
                shoulder_z = 180 - transformed_angles[2]
                elbow = elbow_flexion
                rot_vector = [shoulder_x, shoulder_y, shoulder_z, elbow]

            shoulder_x = rot_vector[0]
            shoulder_y = -(180 - rot_vector[2])
            shoulder_z = -rot_vector[1]
            elbow = -rot_vector[3]

            # Use to test an angle.
            TEST = [-45, 90, 0, -45]

            env.fk_step(
                shoulder_angles=[TEST[0], TEST[1], TEST[2]],
                elbow_angles=[TEST[3], 0, 0],
            )
        env.fk_end(save=True, visualize=False)

    # Read pyrcareworld/Test/evaluate_human_fk.npy
    as_np = np.load(OUTPUT_FILE)
    as_np += CLOUD_OFFSET

    # Demo: Can read points afterwards.
    env.cloud_manager.set_radius(0.7)
    env.cloud_manager.make_cloud(points=as_np, name="Human FK")

    for _ in range(999999):
        env.step()
