#!/usr/bin/env python3
"""
Created on Mon May  9 18:22:56 2022

@author: nvilla
"""

import pybullet_data
import pybullet as p  # PyBullet simulator
import numpy as np
from scipy.spatial.transform import Rotation as R
# import os


class BulletRobot:
    def __init__(self, 
                 controlledJoints,
                 modelPath,
                 URDF_filename,
                 simuStep,
                 rmodelComplete,
                 robotPose = [0.0, 0.0, 1.01927],
                 inertiaOffset = True,
                 talos = True):
        p.connect(p.GUI)  # Start the client for PyBullet
        p.setTimeStep(simuStep)
        p.setGravity(0, 0, -9.81)  # Set gravity (disabled by default)

        # place CoM of root link ## TODO: check placement
        robotStartPosition = robotPose
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        if talos:
            p.setAdditionalSearchPath(modelPath + "/talos_data/robots/")
        else:
            p.setAdditionalSearchPath(modelPath + "/solo_description/robots/")

        self.robotId = p.loadURDF(
            URDF_filename,
            robotStartPosition,
            robotStartOrientation,
            useFixedBase=False,
        )

        # Load horizontal plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        
        self.localInertiaPos = np.array([0, 0, 0])
        if inertiaOffset:
            self.localInertiaPos = p.getDynamicsInfo(self.robotId, -1)[3]  # of the base link

        # leg_left (45-50), leg_right (52-57), torso (0-1), arm_left (11-17),
        # gripper_left (21), arm_right (28-34), gripper_right (38), head (3,4).
        self.bulletJointNames = [
            p.getJointInfo(self.robotId, i)[1].decode()
            for i in range(p.getNumJoints(self.robotId))
        ]
        self.JointIndicesComplete = [
            self.bulletJointNames.index(rmodelComplete.names[i])
            for i in range(2, rmodelComplete.njoints)
        ]

        # Joints controlled with crocoddyl
        self.bulletControlledJoints = [
            i
            for i in self.JointIndicesComplete
            if p.getJointInfo(self.robotId, i)[1].decode() in controlledJoints
        ]

        # Disable default position controler in torque controlled joints
        # Default controller will take care of other joints
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bulletControlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bulletControlledJoints],
        )

        # Augment friction to forbid feet sliding for Talos
        p.changeDynamics(self.robotId, 50, lateralFriction=100, spinningFriction=30)
        p.changeDynamics(self.robotId, 57, lateralFriction=100, spinningFriction=30)

        # Augment friction to forbid feet sliding for Solo
        p.changeDynamics(self.robotId, 3, lateralFriction=100, spinningFriction=30)
        p.changeDynamics(self.robotId, 7, lateralFriction=100, spinningFriction=30)
        p.changeDynamics(self.robotId, 11, lateralFriction=100, spinningFriction=30)
        p.changeDynamics(self.robotId, 15, lateralFriction=100, spinningFriction=30)

    def initializeJoints(self, q0CompleteStart):
        # Initialize position in pyBullet
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=[
                q0CompleteStart[0] + self.localInertiaPos[0],
                q0CompleteStart[1] + self.localInertiaPos[1],
                q0CompleteStart[2] + self.localInertiaPos[2],
            ],
            ornObj=q0CompleteStart[3:7],
        )
        initial_joint_positions = np.array(q0CompleteStart[7:].flat).tolist()
        for i in range(len(initial_joint_positions)):
            p.enableJointForceTorqueSensor(self.robotId, i, True)
            p.resetJointState(
                self.robotId, self.JointIndicesComplete[i], initial_joint_positions[i]
            )

    def resetState(self, q0Start):
        # Initialize position in pyBullet
        p.resetBasePositionAndOrientation(
            self.robotId,
            posObj=[
                q0Start[0] + self.localInertiaPos[0],
                q0Start[1] + self.localInertiaPos[1],
                q0Start[2] + self.localInertiaPos[2],
            ],
            ornObj=q0Start[3:7],
        )
        for i in range(len(self.bulletControlledJoints)):
            p.resetJointState(
                self.robotId, self.bulletControlledJoints[i], q0Start[i + 7]
            )

    def resetReducedState(self, q0Start):
        # Initialize position in pyBullet
        for i in range(len(self.bulletControlledJoints)):
            p.resetJointState(
                self.robotId, self.bulletControlledJoints[i], q0Start[i]
            )

    def addStairs(self, path, position, orientation):
        p.setAdditionalSearchPath(path)
        self.stepId = p.loadURDF("step/step.urdf")
        p.resetBasePositionAndOrientation(
            self.stepId, posObj=position, ornObj=orientation
        )

    def execute(self, torques):
        p.setJointMotorControlArray(
            self.robotId,
            self.bulletControlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )
        p.stepSimulation()
    
    def execute_velocity(self, velocity, max_forces):
        p.setJointMotorControlArray(
            self.robotId,
            self.bulletControlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=velocity,
            forces=max_forces,
        )
        p.stepSimulation()
        
    def changeCamera(self,cameraDistance,cameraYaw,cameraPitch,cameraTargetPos):
        p.resetDebugVisualizerCamera(cameraDistance,
                                     cameraYaw,
                                     cameraPitch,
                                     cameraTargetPos)

    def measureState(self):
        jointStates = p.getJointStates(
            self.robotId, self.JointIndicesComplete
        )  # State of all joints
        baseState = p.getBasePositionAndOrientation(self.robotId)
        baseVel = p.getBaseVelocity(self.robotId)

        # Joint vector for Pinocchio
        q = np.hstack(
            [
                baseState[0],
                baseState[1],
                [jointStates[i_joint][0] for i_joint in range(len(jointStates))],
            ]
        )
        v = np.hstack(
            [
                baseVel[0],
                baseVel[1],
                [jointStates[i_joint][1] for i_joint in range(len(jointStates))],
            ]
        )
        rotation = R.from_quat(q[3:7])
        q[:3] -= rotation.as_matrix() @ self.localInertiaPos
        return q, v
    
    def addTable(self, path, position):
        p.setAdditionalSearchPath(path)
        self.tableId = p.loadURDF("table/table.urdf")
        p.resetBasePositionAndOrientation(self.tableId, posObj=position,ornObj=[0,0,0,1])

    def showSlope(self, position, orientation):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[2, 0.3, 0.01],
            rgbaColor=[0.0, 1.0, 0.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdRight = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=position,
            baseOrientation=orientation,
            useMaximalCoordinates=True,
        )
        
    def showHandToTrack(self, RH_pose):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05],
            rgbaColor=[0.0, 0.0, 1.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdHand = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                RH_pose[0],
                RH_pose[1],
                RH_pose[2],
            ],
            useMaximalCoordinates=True,
        )

    def showTargetToTrack(self, LF_pose, RF_pose):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.075, 0.001],
            rgbaColor=[0.0, 0.0, 1.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdRight = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                RF_pose.translation[0],
                RF_pose.translation[1],
                RF_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

        self.sphereIdLeft = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                LF_pose.translation[0],
                LF_pose.translation[1],
                LF_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )
    
    def showSoloFeet(self, FL_pose, FR_pose, HL_pose, HR_pose):
        visualShapeTarget = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.001],
            rgbaColor=[0.0, 0.0, 1.0, 1.0],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0.0, 0.0, 0.0],
        )

        self.sphereIdFL = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                FL_pose.translation[0],
                FL_pose.translation[1],
                FL_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

        self.sphereIdFR = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                FR_pose.translation[0],
                FR_pose.translation[1],
                FR_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

        self.sphereIdHL = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                HL_pose.translation[0],
                HL_pose.translation[1],
                HL_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

        self.sphereIdHR = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeTarget,
            basePosition=[
                HR_pose.translation[0],
                HR_pose.translation[1],
                HR_pose.translation[2],
            ],
            useMaximalCoordinates=True,
        )

    def moveMarkers(self, LF_trans, RF_trans):

        p.resetBasePositionAndOrientation(
            self.sphereIdRight,
            posObj=[
                RF_trans[0],
                RF_trans[1],
                RF_trans[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        p.resetBasePositionAndOrientation(
            self.sphereIdLeft,
            posObj=[
                LF_trans[0],
                LF_trans[1],
                LF_trans[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    
    def moveSoloFeet(self, FL_pose, FR_pose, HL_pose, HR_pose):

        p.resetBasePositionAndOrientation(
            self.sphereIdFL,
            posObj=[
                FL_pose[0],
                FL_pose[1],
                FL_pose[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        p.resetBasePositionAndOrientation(
            self.sphereIdFR,
            posObj=[
                FR_pose[0],
                FR_pose[1],
                FR_pose[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        p.resetBasePositionAndOrientation(
            self.sphereIdHL,
            posObj=[
                HL_pose[0],
                HL_pose[1],
                HL_pose[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        p.resetBasePositionAndOrientation(
            self.sphereIdHR,
            posObj=[
                HR_pose[0],
                HR_pose[1],
                HR_pose[2],
            ],
            ornObj=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    
    def close(self):
        p.disconnect()


if __name__ == "__main__":

    print("BulletRobot")
