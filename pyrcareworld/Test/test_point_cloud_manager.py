from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
import random
import numpy as np


def generate_random_points(count: int = 100):
    """
    Generates and returns `count` random points between -1 and 1.
    """

    list = []
    for _ in range(count):
        list.append(
            np.array(
                [
                    random.random() * 2 - 1,
                    random.random() * 2 - 1,
                    random.random() * 2 - 1,
                ]
            )
        )

    return np.array(list)


# To be used with the Point Cloud Viz scene.
if __name__ == "__main__":
    env = RCareWorld(executable_file="@editor")
    cloud_manager = PointCloudManager(
        env=env, id=100, name="RCareWorld", is_in_scene=True
    )

    # Example: Make a 100 point cloud with default settings.
    cloud_manager.make_cloud(points=generate_random_points())

    # Example: Scale radius down, then make another.
    cloud_manager.set_radius(radius=0.3)
    cloud_manager.make_cloud(points=generate_random_points())

    # Example: Set position and radius.
    cloud_manager.set_cloud_pos(pos=[-2, 0, -2])
    cloud_manager.make_cloud(points=generate_random_points())

    while True:
        env.step()
