from pyrcareworld.objects import RCareWorldBaseObject
import numpy as np


class PointCloudManager(RCareWorldBaseObject):
    """
    An RCareWorld Point Cloud Manager. Default ID 100.
    """

    def __init__(self, env, id: int, name: str, is_in_scene: bool = False):
        super().__init__(env=env, id=id, name=name, is_in_scene=is_in_scene)

    def make_cloud(self, points: list, name: str):
        """
        Makes a point cloud.

        Args:
            points: A list of 3D points, relative to the point cloud. The point cloud starts at (0, 0, 0), unless `set_cloud_pos` is called.
        """
        self.env.instance_channel.set_action(
            "MakeCloud",
            id=self.id,
            positions=np.array(points).reshape(-1).tolist(),
            name=name,
        )

    def set_cloud_pos(self, pos: list):
        """
        Sets the position of the point cloud.

        Args:
            pos: The position of the point cloud. Any clouds made will now be relative to this position.
        """
        self.env.instance_channel.set_action("SetCloudPos", id=self.id, position=pos)

    def set_radius(self, radius: float):
        """
        Sets the radius of the point cloud.

        Args:
            radius: The radius of the point cloud.
        """
        self.env.instance_channel.set_action("SetRadius", id=self.id, radius=radius)

    def remove_cloud(self, name: str):
        """
        Removes a point cloud.

        Args:
            name: The name of the point cloud.
        """
        self.env.instance_channel.set_action("RemoveCloud", id=self.id, name=name)
