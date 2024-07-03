from pathlib import Path
from simulation.base_sim import BaseSim
import logging
import numpy as np

from real_robot_env.robot.hardware_azure import Azure
from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from real_robot_env.robot.utils.keyboard import KeyManager
from real_robot_env.real_robot_env import RealRobotEnv

import cv2
import time

DELTA_T = 0.034

logger = logging.getLogger(__name__)


class RealRobot(BaseSim):
    def __init__(self, device: str):
        super().__init__(seed=-1, device=device)

        self.p4 = FrankaArm(
            name="p4",
            ip_address="141.3.53.154",
            port=50053,
            control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL,
            hz=100
        )
        assert self.p4.connect(), f"Connection to {self.p4.name} failed"

        self.p4_hand = FrankaHand(name="p4_hand", ip_address="141.3.53.154", port=50054)
        assert self.p4_hand.connect(), f"Connection to {self.p4_hand.name} failed"

        self.cam0 = Azure(device_id=0)
        self.cam1 = Azure(device_id=1)

        self.i = 0

    # def test_agent(self, agent):
    #     logger.info("Starting trained model evaluation on real robot")
    #
    #     km = KeyManager()
    #
    #     while km.key != 'q':
    #         print("Press 's' to start a new evaluation, or 'q' to quit")
    #         km.pool()
    #
    #         while km.key not in ['s', 'q']:
    #             km.pool()
    #
    #         if km.key == 's':
    #             print()
    #             agent.reset()
    #
    #             assert self.cam0.connect(), f"Connection to {self.cam0.name} failed"
    #             assert self.cam1.connect(), f"Connection to {self.cam1.name} failed"
    #
    #             print("Starting evaluation. Press 'd' to stop current evaluation")
    #
    #             km.pool()
    #             while km.key != 'd':
    #                 km.pool()
    #
    #                 obs = self.__get_obs()
    #                 pred_action = agent.predict(obs, if_vision=True).squeeze()
    #
    #                 pred_joint_pos = pred_action[:7]
    #                 pred_gripper_command = pred_action[-1]
    #
    #                 pred_gripper_command = 1 if pred_gripper_command > 0 else -1
    #
    #                 self.p4.go_to_within_limits(goal=pred_joint_pos)
    #                 self.p4_hand.apply_commands(width=pred_gripper_command)
    #                 time.sleep(DELTA_T)
    #
    #             print()
    #             logger.info("Evaluation done. Resetting robots")
    #             # time.sleep(1)
    #
    #             self.cam0.close()
    #             self.cam1.close()
    #             self.p4.reset()
    #             self.p4_hand.reset()
    #
    #     print()
    #     logger.info("Quitting evaluation")
    #
    #     km.close()
    #     self.p4.close()
    #     self.p4_hand.reset()

    def test_agent(self, agent):
        env = RealRobotEnv(
            robot_name="p4",
            robot_ip_address="141.3.53.154",
            robot_arm_port=50053,
            robot_gripper_port=50054
        )

        logger.info("Starting trained model evaluation on real robot")

        km = KeyManager()

        while km.key != 'q':
            print("Press 's' to start a new evaluation, or 'q' to quit")
            km.pool()

            while km.key not in ['s', 'q']:
                km.pool()

            if km.key == 's':
                print()

                agent.reset()
                obs, _ = env.reset()

                print("Starting evaluation. Press 'd' to stop current evaluation")

                km.pool()
                while km.key != 'd':
                    km.pool()
                    pred_action = agent.predict((obs.cam0_img, obs.cam1_img), if_vision=True).squeeze()
                    obs, *_ = env.step(pred_action)
                    time.sleep(DELTA_T)

                print()
                logger.info("Evaluation done. Resetting robots")

                env.reset()

        print()
        logger.info("Quitting evaluation")

        km.close()
        env.close()

    def __get_obs(self):

        img0 = self.cam0.get_sensors()["rgb"][:, :, :3]  # remove depth
        img1 = self.cam1.get_sensors()["rgb"][:, :, :3]

        img0 = cv2.resize(img0, (512, 512))[:, 100:370]
        img1 = cv2.resize(img1, (512, 512))

        # cv2.imshow('0', img0)
        # cv2.imshow('1', img1)
        # cv2.waitKey(0)

        processed_img0 = cv2.resize(img0, (128, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0
        processed_img1 = cv2.resize(img1, (256, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0

        return (processed_img0, processed_img1)

