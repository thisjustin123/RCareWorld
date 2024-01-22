from pyrcareworld.envs import RCareWorld
import numpy as np
import pybullet as p
import math


class SkeletonEnv(RCareWorld):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        self.robot = self.create_robot(315893, [3158930], "kinova_gen3_7dof")
        self.init_pose_obj = self.create_object(6666, "Ini", True)
        ini_world_pose = self.init_pose_obj.getPosition()
        ini_world_rot = self.init_pose_obj.getQuaternion()
        self.robot.moveTo(ini_world_pose, ini_world_rot)
        self.skin = self.create_skin(id=114514, name="Skin", is_in_scene=True)

    def step(self):
        pose = self.init_pose_obj.getPosition()
        rot = self.init_pose_obj.getQuaternion()
        self.robot.moveTo(pose, rot)
        skin_info = self.skin.getInfo()

        force_on_skeleton = {}
        for i in range(len(skin_info["skeleton_ids"])):
            skeleton_id = skin_info["skeleton_ids"][i]
            if skeleton_id == -1:
                continue

            if skeleton_id not in force_on_skeleton:
                force_on_skeleton[skeleton_id] = 0
            force_on_skeleton[skeleton_id] += skin_info["forces"][i]
          
        if (not hasattr(self, "prev_force_on_skeleton") or force_on_skeleton != self.prev_force_on_skeleton):
          print("Forces along IDs are now:", force_on_skeleton)

        self.prev_force_on_skeleton = force_on_skeleton
        self._step()

    def demo(self):
        for i in range(10000000):
            self.step()


if __name__ == "__main__":
    env = SkeletonEnv()
    env.demo()
