import numpy as np
import pinocchio as pin
import ndcurves
import os 
import example_robot_data

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'

URDF_FILENAME = "talos_reduced.urdf"
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = np.cos(theta)
    x = x * np.sin(theta)
    y = y * np.sin(theta)
    z = z * np.sin(theta)
    return np.array([x, y, z, w])

def loadTalos():
    robotComplete = example_robot_data.load("talos")
    qComplete = robotComplete.model.referenceConfigurations["half_sitting"]

    locked_joints = [20,21,22,23,28,29,30,31]
    locked_joints += [32, 33]
    robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
    rmodel: pin.Model = robot.model
    q0 = rmodel.referenceConfigurations["half_sitting"]

    return robotComplete.model, rmodel, qComplete, q0

def addCoinFrames(rmodel, LF_id, RF_id):
    trans_FL = np.array([0.1, 0.075, 0])
    trans_FR = np.array([0.1, -0.075, 0])
    trans_HL = np.array([-0.1, 0.075, 0])
    trans_HR = np.array([-0.1, -0.075, 0])
    FL_contact_placement = pin.SE3.Identity()
    FL_contact_placement.translation = trans_FL
    FR_contact_placement = pin.SE3.Identity()
    FR_contact_placement.translation = trans_FR
    HL_contact_placement = pin.SE3.Identity()
    HL_contact_placement.translation = trans_HL
    HR_contact_placement = pin.SE3.Identity()
    HR_contact_placement.translation = trans_HR

    frame_contact_placement = [
        FL_contact_placement,
        FR_contact_placement,
        HL_contact_placement,
        HR_contact_placement
    ]
                            
    LF_FL_frame = pin.Frame("LF_FL", 
                            rmodel.frames[LF_id].parentJoint,
                            rmodel.frames[LF_id].parentFrame,
                            rmodel.frames[LF_id].placement * FL_contact_placement, pin.OP_FRAME)
    LF_FR_frame = pin.Frame("LF_FR", 
                            rmodel.frames[LF_id].parentJoint,
                            rmodel.frames[LF_id].parentFrame,
                            rmodel.frames[LF_id].placement * FR_contact_placement, pin.OP_FRAME)
    LF_HL_frame = pin.Frame("LF_HL", 
                            rmodel.frames[LF_id].parentJoint,
                            rmodel.frames[LF_id].parentFrame,
                            rmodel.frames[LF_id].placement * HL_contact_placement, pin.OP_FRAME)
    LF_HR_frame = pin.Frame("LF_HR", 
                            rmodel.frames[LF_id].parentJoint,
                            rmodel.frames[LF_id].parentFrame,
                            rmodel.frames[LF_id].placement * HR_contact_placement, pin.OP_FRAME)

    RF_FL_frame = pin.Frame("RF_FL", 
                            rmodel.frames[RF_id].parentJoint,
                            rmodel.frames[RF_id].parentFrame,
                            rmodel.frames[RF_id].placement * FL_contact_placement, pin.OP_FRAME)
    RF_FR_frame = pin.Frame("RF_FR", 
                            rmodel.frames[RF_id].parentJoint,
                            rmodel.frames[RF_id].parentFrame,
                            rmodel.frames[RF_id].placement * FR_contact_placement, pin.OP_FRAME)
    RF_HL_frame = pin.Frame("RF_HL", 
                            rmodel.frames[RF_id].parentJoint,
                            rmodel.frames[RF_id].parentFrame,
                            rmodel.frames[RF_id].placement * HL_contact_placement, pin.OP_FRAME)
    RF_HR_frame = pin.Frame("RF_HR", 
                            rmodel.frames[RF_id].parentJoint,
                            rmodel.frames[RF_id].parentFrame,
                            rmodel.frames[RF_id].placement * HR_contact_placement, pin.OP_FRAME)

    LF_FL_id = rmodel.addFrame(LF_FL_frame)
    LF_FR_id = rmodel.addFrame(LF_FR_frame)
    LF_HL_id = rmodel.addFrame(LF_HL_frame)
    LF_HR_id = rmodel.addFrame(LF_HR_frame)

    RF_FL_id = rmodel.addFrame(RF_FL_frame)
    RF_FR_id = rmodel.addFrame(RF_FR_frame)
    RF_HL_id = rmodel.addFrame(RF_HL_frame)
    RF_HR_id = rmodel.addFrame(RF_HR_frame)

    contact_ids = [LF_FL_id, LF_FR_id, LF_HL_id, LF_HR_id,
             RF_FL_id, RF_FR_id, RF_HL_id, RF_HR_id]

    return rmodel, contact_ids, frame_contact_placement

def save_trajectory(
    xs,
    us,
    com,
    LF_force,
    RF_force,
    LF_torque,
    RF_torque,
    time,
    LF_trans,
    RF_trans,
    LF_trans_ref,
    RF_trans_ref,
    L_measured,
    save_name=None,
    save_dir=DEFAULT_SAVE_DIR,
):
    """
    Saves data to a compressed npz file (binary)
    """
    simu_data = {}
    simu_data["xs"] = xs
    simu_data["us"] = us
    simu_data["com"] = com
    simu_data["LF_force"] = LF_force
    simu_data["RF_force"] = RF_force
    simu_data["LF_torque"] = LF_torque
    simu_data["RF_torque"] = RF_torque
    simu_data["LF_pose"] = LF_trans
    simu_data["RF_pose"] = RF_trans
    simu_data["LF_pose_ref"] = LF_trans_ref
    simu_data["RF_pose_ref"] = RF_trans_ref
    simu_data["L_measured"] = L_measured
    simu_data["time"] = time
    print("Compressing & saving data...")
    if save_name is None:
        save_name = "sim_data_NO_NAME" + str(time.time())
    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    save_path = save_dir + "/" + save_name + ".npz"
    np.savez_compressed(save_path, data=simu_data)
    print("Saved data to " + str(save_path) + " !")

def computeCoP(LF_pose, RF_pose, LF_force, LF_torque, RF_force, RF_torque):
    cop_total = np.zeros(3)
    total_z_force = 0
    if LF_force[2] > 1.:
        local_cop_left = np.array([-LF_torque[1] / LF_force[2],
                                  LF_torque[0] / LF_force[2], 
                                  0.0]
        )
        cop_total += (LF_pose.rotation @ local_cop_left +
                      LF_pose.translation) * LF_force[2]
        total_z_force += LF_force[2]
    if RF_force[2] > 1.:
        local_cop_right = np.array([-RF_torque[1] / RF_force[2],
                                   RF_torque[0] / RF_force[2], 
                                   0.0]
        )
        cop_total += (RF_pose.rotation @ local_cop_right + 
                      RF_pose.translation) * RF_force[2]
        total_z_force += RF_force[2]
    if (total_z_force < 1.): print("Zero force detected")
    cop_total /= total_z_force
    
    return cop_total

def load_data(npz_file):
    """
    Loads a npz archive of sim_data into a dict
    """
    d = np.load(npz_file, allow_pickle=True, encoding="latin1")
    return d["data"][()]

class footTrajectory:
    def __init__(
        self, start_pose_left, start_pose_right, T_ss, T_ds, nsteps, swing_apex, x_forward, y_forward, foot_angle, y_gap, z_height
    ):
        self.translationRight = np.array([x_forward, -y_gap - y_forward, z_height])
        self.translationLeft = np.array([x_forward, y_gap, z_height])
        self.rotationDiff = self.yawRotation(foot_angle)

        self.start_pose_left = start_pose_left
        self.start_pose_right = start_pose_right
        self.final_pose_left = start_pose_left
        self.final_pose_right = start_pose_right

        self.T_ds = T_ds
        self.T_ss = T_ss
        self.nsteps = nsteps
        self.swing_apex = swing_apex
    
    def updateForward(self, x_f_left, x_f_right, y_gap, y_forward, z_height_left, z_height_right, swing_apex):
        self.translationRight = np.array([x_f_right, -y_gap - y_forward, z_height_right])
        self.translationLeft = np.array([x_f_left, y_gap, z_height_left])
        self.swing_apex = swing_apex
    
    def updateTrajectory(self, takeoff_RF, takeoff_LF, land_RF, land_LF, LF_pose, RF_pose):
        if land_LF < 0:
            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = LF_pose.copy()
        
        if land_RF < 0:
            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = RF_pose.copy()
        
        if takeoff_RF < self.T_ds and takeoff_RF >= 0:
            #print("Update right trajectory")
            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = LF_pose.copy()
            yawLeft = self.extractYaw(LF_pose.rotation)
            self.final_pose_right.translation += self.yawRotation(yawLeft) @ self.translationRight
            self.final_pose_right.rotation = self.rotationDiff @ self.final_pose_right.rotation

            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = self.final_pose_right.copy()
            yawRight = self.extractYaw(self.final_pose_right.rotation)
            self.final_pose_left.translation += self.yawRotation(yawRight) @ self.translationLeft

        if takeoff_LF < self.T_ds and takeoff_LF >= 0:
            #print("Update left trajectory")
            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = RF_pose.copy()
            yawRight = self.extractYaw(RF_pose.rotation)
            self.final_pose_left.translation += self.yawRotation(yawRight) @ self.translationLeft

            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = self.final_pose_left.copy()
            yawLeft = self.extractYaw(self.final_pose_left.rotation)
            self.final_pose_right.translation += self.yawRotation(yawLeft) @ self.translationRight
            self.final_pose_right.rotation = self.rotationDiff @ self.final_pose_right.rotation
        
        swing_trajectory_left = self.defineBezier(
            self.swing_apex, 0, 1, self.start_pose_left, self.final_pose_left
        )
        swing_trajectory_right = self.defineBezier(
            self.swing_apex, 0, 1, self.start_pose_right, self.final_pose_right
        )

        LF_refs = (
            self.foot_trajectory(
                self.nsteps,
                land_LF,
                self.start_pose_left,
                self.final_pose_left,
                swing_trajectory_left,
                self.T_ss,
            )
            if land_LF > -1
            else ([self.start_pose_left for i in range(self.nsteps)])
        )

        RF_refs = (
            self.foot_trajectory(
                self.nsteps,
                land_RF,
                self.start_pose_right,
                self.final_pose_right,
                swing_trajectory_right,
                self.T_ss,
            )
            if land_RF > -1
            else ([self.start_pose_right for i in range(self.nsteps)])
        )

        return LF_refs, RF_refs


    def defineBezier(self, height, time_init, time_final, placement_init, placement_final):
        wps = np.zeros([3, 9])
        for i in range(4):  # init position. init vel,acc and jerk == 0
            wps[:, i] = placement_init.translation
        # compute mid point (average and offset along z)
        wps[:, 4] = placement_init.translation * 3/4 + placement_final.translation * 1/4
        wps[2, 4] += height
        for i in range(5, 9):  # final position. final vel,acc and jerk == 0
            wps[:, i] = placement_final.translation
        translation = ndcurves.bezier3(wps, time_init, time_final)
        pBezier = ndcurves.piecewise_SE3(
            ndcurves.SE3Curve(
                translation, placement_init.rotation, placement_final.rotation
            )
        )
        return pBezier
    
    def foot_trajectory(self, T, time_to_land, initial_pose, final_pose, trajectory_swing, TsingleSupport):
        placement = []
        for t in range(
            time_to_land, time_to_land - T, -1
        ):
            if t <= 0:
                placement.append(final_pose)
            elif t > TsingleSupport:
                placement.append(initial_pose)
            else:
                swing_pose = initial_pose.copy()
                swing_pose.translation = trajectory_swing.translation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                swing_pose.rotation = trajectory_swing.rotation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                placement.append(swing_pose)

        return placement
    
    def yawRotation(self, yaw):
        Ro = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        return Ro


    def extractYaw(self, Ro):
        return np.arctan2(Ro[1, 0], Ro[0, 0])

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def shapeState(q_current, v_current, nq, nxq, cj_ids):
    x_internal = np.zeros(nxq)
    x_internal[:7] = q_current[:7]
    x_internal[nq:nq + 6] = v_current[:6]
    i = 0
    for jointID in cj_ids:
        if jointID > 1:
            x_internal[i + 7] = q_current[jointID + 5]
            x_internal[nq + i + 6] = v_current[jointID + 4]
            i += 1
    
    return x_internal

def scan_list(list_Fs):
    for i in range(len(list_Fs)):
        list_Fs[i] -= 1
    if len(list_Fs) > 0 and list_Fs[0] == -1:
        list_Fs.remove(list_Fs[0])

def update_timings(land_LFs, land_RFs, takeoff_LFs, takeoff_RFs):
    scan_list(land_LFs)
    scan_list(land_RFs)
    scan_list(takeoff_LFs)
    scan_list(takeoff_RFs)
    land_LF = -1
    land_RF = -1
    takeoff_LF = -1
    takeoff_RF = -1
    if len(land_LFs) > 0:
        land_LF = land_LFs[0]
    if len(land_RFs) > 0:
        land_RF = land_RFs[0]
    if len(takeoff_LFs) > 0:
        takeoff_LF = takeoff_LFs[0]
    if len(takeoff_RFs) > 0:
        takeoff_RF = takeoff_RFs[0]
    return takeoff_RF, takeoff_LF, land_RF, land_LF

def compute_ID_references(space, rmodel, rdata, LF_id, RF_id, base_id, torso_id, x0_multibody, x_measured, LF_refs, RF_refs, dt):
    LF_vel_lin = pin.getFrameVelocity(rmodel,rdata,LF_id,pin.LOCAL).linear
    RF_vel_lin = pin.getFrameVelocity(rmodel,rdata,RF_id,pin.LOCAL).linear
    LF_vel_ang = pin.getFrameVelocity(rmodel,rdata,LF_id,pin.LOCAL).angular
    RF_vel_ang = pin.getFrameVelocity(rmodel,rdata,RF_id,pin.LOCAL).angular

    q_diff = -space.difference(x0_multibody, x_measured)[:rmodel.nv]
    dq_diff = -space.difference(x0_multibody, x_measured)[rmodel.nv:]
    LF_diff = np.zeros(6)
    LF_diff[:3] = LF_refs[0].translation - rdata.oMf[LF_id].translation
    LF_diff[3:] = -pin.log3(LF_refs[0].rotation.T @ rdata.oMf[LF_id].rotation)
    RF_diff = np.zeros(6)
    RF_diff[:3] = RF_refs[0].translation - rdata.oMf[RF_id].translation
    RF_diff[3:] = -pin.log3(RF_refs[0].rotation.T @ rdata.oMf[RF_id].rotation)

    dLF_diff = np.zeros(6)
    dLF_diff[:3] = (LF_refs[1].translation - LF_refs[0].translation) / dt - LF_vel_lin
    dLF_diff[3:] = pin.log3(LF_refs[0].rotation.T @ LF_refs[1].rotation) / dt -LF_vel_ang
    dRF_diff = np.zeros(6)
    dRF_diff[:3] = (RF_refs[1].translation - RF_refs[0].translation) / dt - RF_vel_lin
    dRF_diff[3:] = pin.log3(RF_refs[0].rotation.T @ RF_refs[1].rotation) / dt - RF_vel_ang

    base_diff = -pin.log3(RF_refs[0].rotation.T @ rdata.oMf[base_id].rotation)
    torso_diff = -pin.log3(RF_refs[0].rotation.T @ rdata.oMf[torso_id].rotation)
    dbase_diff = pin.log3(RF_refs[0].rotation.T @ RF_refs[1].rotation) / dt -pin.getFrameVelocity(rmodel,rdata,base_id,pin.LOCAL).angular
    dtorso_diff = pin.log3(RF_refs[0].rotation.T @ RF_refs[1].rotation) / dt -pin.getFrameVelocity(rmodel,rdata,torso_id,pin.LOCAL).angular

    return q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff