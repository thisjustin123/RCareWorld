import random
import numpy as np

from pyrcareworld.envs.base_env import RCareWorld
from pyrcareworld.agents.pointcloudmanager import PointCloudManager
from pyrcareworld.agents.robot import Robot


def lerp(a, b, t):
    return a + (b - a) * t


def rotate_matrix(points, x_angle, y_angle, z_angle):
    # Convert angles from degrees to radians
    x_rad = np.radians(x_angle)
    y_rad = np.radians(y_angle)
    z_rad = np.radians(z_angle)

    # Rotation matrix around the X axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(x_rad), -np.sin(x_rad)],
            [0, np.sin(x_rad), np.cos(x_rad)],
        ]
    )

    # Rotation matrix around the Y axis
    Ry = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad)],
            [0, 1, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad)],
        ]
    )

    # Rotation matrix around the Z axis
    Rz = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0],
            [np.sin(z_rad), np.cos(z_rad), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Apply the rotation matrix to the points
    rotated_points = np.dot(points, R.T)

    return rotated_points


class RhiVisEnv(RCareWorld):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        point_cloud: np.ndarray = None,
        start_pos: list = [0, 0, 0],
        end_pos: list = [0, 0, 0],
        handoff_pos: list = [0, 0, 0],
        human_end_pos: list = [0, 0, 0],
        obj_grab_offset: list = [0, 0, 0],
        point_cloud_path: str = None,
        output_file: str = "pyrcareworld/Test/evaluate_human_fk.npy",
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
        self.output_file = output_file
        self.handoff_pos = handoff_pos
        self.human_end_pos = human_end_pos
        self.point_cloud = point_cloud
        self.obj_grab_offset = obj_grab_offset
        self.person = self.create_human(id=85042, name="Human", is_in_scene=True)
        self.cloud_manager = PointCloudManager(
            env=self, id=100, name="RCareWorld", is_in_scene=True
        )
        self.cloud_manager.set_radius(0.3)
        self.point_cloud_path = point_cloud_path
        if point_cloud_path is not None:
            self.point_cloud = self.load_point_cloud()
        self.list = []

    def load_point_cloud(self):
        as_np = np.load(self.point_cloud_path)
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
        return as_np

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
        self.cloud_manager.set_radius(radius=0.5)
        if self.point_cloud is not None:
            CLOUD_OFFSET = np.array([0.34, 0.45, -0.03])
            as_np = self.point_cloud
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
            self.point_cloud = self.point_cloud + CLOUD_OFFSET
            self.cloud_manager.make_cloud(
                points=as_np.tolist(),
                name="cloud",
            )
        else:
            CLOUD_OFFSET = np.array([0.54, 0.25, 0])
            self.cloud_manager.make_cloud(
                points=self.generate_random_points() + CLOUD_OFFSET,
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

        self.cloud_manager.remove_cloud(name="cloud")

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

    def demo_bathing(self):
        robot: Robot = self.create_robot(
            id=221584,
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[-0.425999999, 0.619000018, 0.606999993],
            gripper_list=[221584],
        )
        CLOUD_OFFSET = np.array([0.187999994, 1.18200004, -0.44600001])
        GRASP_POINT = np.array([-0.108999997, 0.742999971, 0.361000001])
        cloud = self.point_cloud
        cloud = rotate_matrix(
            cloud,
            x_angle=0,
            y_angle=180,
            z_angle=0,
        )
        cloud = rotate_matrix(
            cloud,
            x_angle=0,
            y_angle=0,
            z_angle=-90,
        )
        cloud += CLOUD_OFFSET
        self.cloud_manager.make_cloud(points=cloud, name="Human FK")

        for _ in range(30):
            robot.directlyMoveTo(GRASP_POINT)
            self.step()

        robot.directlyMoveTo(GRASP_POINT)
        robot.GripperClose()

        for _ in range(30):
            self.step()

        while True:
            robot.directlyMoveTo(GRASP_POINT)
            self.step()

    def demo_dressing(self):
        robot: Robot = self.create_robot(
            id=221584,
            robot_name="kinova_gen3_7dof-robotiq85",
            base_pos=[-0.231, 0, 0],
            gripper_list=[221584],
        )
        CLOUD_OFFSET = np.array([-0.198, 0.63, -0.371])
        GRASP_POINT = np.array([0.00300000003, 0.358999999, 0.583999991])
        CLOTH_GRASP_GOAL = GRASP_POINT + np.array([0, 1, 0])
        cloud = rotate_matrix(
            self.point_cloud,
            x_angle=0,
            y_angle=-150,
            z_angle=0,
        )
        cloud += CLOUD_OFFSET
        self.cloud_manager.make_cloud(points=cloud, name="Human FK")

        for _ in range(30):
            robot.directlyMoveTo(GRASP_POINT)
            self.step()

        robot.directlyMoveTo(GRASP_POINT)
        robot.GripperClose()

        for _ in range(30):
            self.step()

        for i in range(300):
            robot.directlyMoveTo(lerp(GRASP_POINT, CLOTH_GRASP_GOAL, i / 300))
            self.step()

        while True:
            robot.directlyMoveTo(CLOTH_GRASP_GOAL)
            self.step()

    def demo_human_only(self):
        CLOUD_OFFSET = np.array([-0.198, 0.63, -0.371])
        cloud = rotate_matrix(
            self.point_cloud,
            x_angle=0,
            y_angle=-150,
            z_angle=0,
        )
        cloud += CLOUD_OFFSET
        self.cloud_manager.make_cloud(points=cloud, name="Human FK")

        while True:
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
        self.person.setJointRotationByNameDirectly(
            joint_name="RightLowerArm", position=elbow_angles
        )
        self.step()

        self.list.append(self.person.getJointPositionByName("RightHand"))

    def fk_end(self, save: bool = False, visualize: bool = False):
        DISCARD = 2
        as_np = np.array(self.list[DISCARD:])

        if save:
            # Save as_np as a file named evaluate_human_fk.npy
            np.save(self.output_file, as_np)

        if visualize:
            for _ in range(300):
                self.step()
