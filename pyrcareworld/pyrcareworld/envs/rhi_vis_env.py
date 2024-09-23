import random
import numpy as np

from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
from pyrcareworld.agents.robot import Robot


def lerp(a, b, t):
    return a + (b - a) * t


class RhiVisEnv(RCareWorld):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        start_pos: list = [0, 0, 0],
        end_pos: list = [0, 0, 0],
        handoff_pos: list = [0, 0, 0],
        human_end_pos: list = [0, 0, 0],
        obj_grab_offset: list = [0, 0, 0],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            **kwargs,
        )

        self.start_pos = start_pos
        self.end_pos = end_pos
        self.handoff_pos = handoff_pos
        self.human_end_pos = human_end_pos
        self.obj_grab_offset = obj_grab_offset
        self.person = self.create_human(id=85042, name="Human", is_in_scene=True)
        self.list = []

    def generate_random_points(self, count: int = 100):
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

        return np.array(list) / 5

    def demo(self):
        cloud_manager = PointCloudManager(
            env=self, id=100, name="RCareWorld", is_in_scene=True
        )
        robot: Robot = self.create_robot(
            id=221584,
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[0, 0, -0.721000016],
            gripper_list=[2215840],
        )
        target_object = self.create_object(id=2, name="Duster", is_in_scene=True)
        start_pos = np.array(self.start_pos)
        end_pos = np.array(self.end_pos)

        duster_pos = np.array(target_object.getPosition())

        for _ in range(50):
            self.step()
            robot.directlyMoveTo(start_pos)

        for i in range(150):
            pos = lerp(start_pos, duster_pos, i / 150)
            robot.directlyMoveTo(pos)
            self.step()

        robot.GripperClose()

        for _ in range(60):
            robot.directlyMoveTo(pos)
            self.step()

        # Here's where you'd visualize point cloud.

        # TODO: Vis point cloud.

        cloud_manager.set_radius(radius=0.3)
        cloud_manager.make_cloud(
            points=self.generate_random_points() + np.array([0.1, 0.35, -0.5]),
            name="cloud",
        )

        for _ in range(100):
            robot.directlyMoveTo(pos)
            self.step()

        gripper_pos = np.array(duster_pos)
        invis_target_pos = np.array(self.handoff_pos)

        for i in range(150):
            pos = lerp(gripper_pos, invis_target_pos, i / 150)
            robot.directlyMoveTo(pos)
            self.step()

        cloud_manager.remove_cloud(name="cloud")

        # Here's where the human should grab the object.

        # TODO: Make human grasp object.
        self.person.ik_move_to(
            position=np.array(target_object.getPosition())
            + np.array(self.obj_grab_offset)
        )

        for _ in range(70):
            self.step()

        self.person.gripper_close()
        robot.GripperOpen()

        for _ in range(40):
            self.step()

        self.person.ik_move_to(self.human_end_pos)

        invis_target_pos = np.array(self.handoff_pos)

        for i in range(150):
            pos = lerp(invis_target_pos, end_pos, i / 150)
            robot.directlyMoveTo(pos)
            self.step()

        for i in range(300):
            robot.directlyMoveTo(end_pos)
            self.step()

    def fk_step(
        self,
        shoulder_angles: list,
        elbow_angles: list,
    ):
        self.person.setJointRotationByNameDirectly(
            joint_name="RightUpperArm",
            position=shoulder_angles,
        )
        self.person.setJointRotationByNameDirectly("RightLowerArm", elbow_angles)
        self.step()

        self.list.append(self.person.getJointPositionByName("RightHand"))

    def fk_vis_and_close(self, save: bool = False):
        as_np = np.array(self.list)

        cloud_manager = PointCloudManager(
            env=self, id=100, name="RCareWorld", is_in_scene=True
        )

        cloud_manager.make_cloud(points=as_np, name="cloud")

        if save:
            # Save as_np as a file named evaluate_human_fk.npy
            np.save("pyrcareworld/Test/evaluate_human_fk.npy", as_np)

        for _ in range(300):
            self.step()
