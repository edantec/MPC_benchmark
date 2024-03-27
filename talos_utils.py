import numpy as np
import pinocchio as pin
import proxsuite
import ndcurves
import os 
import example_robot_data

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'

URDF_FILENAME = "talos_reduced.urdf"
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

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
        self, start_pose_left, start_pose_right, T_ss, T_ds, nsteps, swing_apex, x_forward, y_gap, x_depth
    ):
        self.translationRight = np.array([x_forward, -y_gap, -x_depth])
        self.translationLeft = np.array([x_forward, y_gap, -x_depth])

        self.start_pose_left = start_pose_left
        self.start_pose_right = start_pose_right
        self.final_pose_left = start_pose_left
        self.final_pose_right = start_pose_right

        self.T_ds = T_ds
        self.T_ss = T_ss
        self.nsteps = nsteps
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
            self.final_pose_right.translation += self.translationRight

            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = self.final_pose_right.copy()
            self.final_pose_left.translation += self.translationLeft

        if takeoff_LF < self.T_ds and takeoff_LF >= 0:
            #print("Update left trajectory")
            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = RF_pose.copy()
            self.final_pose_left.translation += self.translationLeft

            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = self.final_pose_left.copy()
            self.final_pose_right.translation += self.translationRight
        
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
        wps[:, 4] = (placement_init.translation + placement_final.translation) / 2.0
        wps[2, 4] += height
        for i in range(5, 9):  # final position. final vel,acc and jerk == 0
            wps[:, i] = placement_final.translation
        translation = ndcurves.bezier(wps, time_init, time_final)
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

class IKSolver:
    def __init__(
        self, model, weights, sole_ids, base_id, torso_id, dt, low_limits, up_limits, verbose: bool
    ):
        n = model.nv
        neq = 0
        nin = model.nv - 6

        self.g = np.zeros(n)
        self.H = np.zeros((n,n))
        self.l = np.zeros(nin)
        self.u = np.zeros(nin)
        C = np.zeros((nin, n))
        C[:,6:] = np.eye(nin)
        self.weights = weights
        self.alpha = 0.1
        self.Klim = 0.5
        self.sole_ids = sole_ids
        self.base_id = base_id
        self.torso_id = torso_id
        self.dt = dt
        self.low_limits = low_limits
        self.up_limits = up_limits

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 10
        qp.init(self.H, self.g, None, None, C, self.l, self.u)
        self.qp = qp

        self.model = model
    
    def computeMatrice(self, data, q, dq_ref, dp_left, dp_right, dbase, dtorso, H):
        Jc_left = pin.getFrameJacobian(self.model, data, self.sole_ids[0], pin.LOCAL) 
        Jc_right = pin.getFrameJacobian(self.model, data, self.sole_ids[1], pin.LOCAL) 
        Jc_base = pin.getFrameJacobian(self.model, data, self.base_id, pin.LOCAL)[3:]
        Jc_torso = pin.getFrameJacobian(self.model, data, self.torso_id, pin.LOCAL)[3:]
        
        self.H = self.weights[0] * np.eye(self.model.nv)
        self.H += self.weights[1] * Jc_left.transpose() @ Jc_left
        self.H += self.weights[1] * Jc_right.transpose() @ Jc_right
        self.H += self.weights[2] * data.Ag.transpose() @ data.Ag
        self.H += self.weights[3] * Jc_base.transpose() @ Jc_base
        self.H += self.weights[3] * Jc_torso.transpose() @ Jc_torso

        self.g = self.alpha * self.weights[0] * dq_ref
        self.g -= self.weights[1] * dp_left.transpose() @ Jc_left
        self.g -= self.weights[1] * dp_right.transpose() @ Jc_right
        self.g -= self.weights[2] * H.transpose() @ data.Ag
        self.g -= self.weights[3] * dbase.transpose() @ Jc_base
        self.g -= self.weights[3] * dtorso.transpose() @ Jc_torso

        self.l = self.Klim * (self.low_limits - q[7:]) / self.dt
        self.u = self.Klim * (self.up_limits - q[7:]) / self.dt



    def solve(self, data, q, dq_ref, dp_left, dp_right, dbase, dtorso, H):
        self.computeMatrice(data, q, dq_ref, dp_left, dp_right, dbase, dtorso, H)

        self.qp.update(
            H=self.H,
            g=self.g,
            l=self.l,
            u=self.u,
            update_preconditioner=False,
        )
        self.qp.solve()
        qdot = self.qp.results.x[:]

        return qdot 

class IK2Solver:
    def __init__(
        self, model, weights, K_gains, sole_ids, base_id, torso_id, verbose: bool
    ):
        n = model.nv
        neq = 0
        nin = 0

        self.K_gains = K_gains

        self.g = np.zeros(n)
        self.H = np.zeros((n,n))
        self.l = None
        self.u = None
        C = None

        self.weights = weights
        self.alpha = 0.1
        self.Klim = 0.5
        self.sole_ids = sole_ids
        self.base_id = base_id
        self.torso_id = torso_id

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 100
        qp.init(self.H, self.g, None, None, C, self.l, self.u)
        self.qp = qp

        self.model = model
    
    def computeMatrice(self, data, v, q_diff, dq_diff, LF_diff, RF_diff, dLF_diff, dRF_diff, dH, base_diff, dbase_diff, torso_diff, dtorso_diff):
        Jc_left = pin.getFrameJacobian(self.model, data, self.sole_ids[0], pin.LOCAL) 
        Jc_right = pin.getFrameJacobian(self.model, data, self.sole_ids[1], pin.LOCAL) 
        dJ_left = pin.getFrameJacobianTimeVariation(self.model, data, self.sole_ids[0], pin.LOCAL)
        dJ_right = pin.getFrameJacobianTimeVariation(self.model, data, self.sole_ids[1], pin.LOCAL)
        Jc_base = pin.getFrameJacobian(self.model, data, self.base_id, pin.LOCAL)[3:]
        Jc_torso = pin.getFrameJacobian(self.model, data, self.torso_id, pin.LOCAL)[3:]
        dJ_base = pin.getFrameJacobianTimeVariation(self.model, data, self.base_id, pin.LOCAL)[3:]
        dJ_torso = pin.getFrameJacobianTimeVariation(self.model, data, self.torso_id, pin.LOCAL)[3:]
        
        self.H = self.weights[0] * np.eye(self.model.nv)
        self.H += self.weights[1] * Jc_left.transpose() @ Jc_left
        self.H += self.weights[1] * Jc_right.transpose() @ Jc_right
        self.H += self.weights[2] * data.Ag.transpose() @ data.Ag
        self.H += self.weights[3] * Jc_base.transpose() @ Jc_base
        self.H += self.weights[3] * Jc_torso.transpose() @ Jc_torso

        self.g = self.weights[0] * (- self.K_gains[0][0] @ q_diff - self.K_gains[0][1] @ dq_diff )
        self.g += self.weights[1] * (dJ_left @ v - self.K_gains[1][0] @ LF_diff - self.K_gains[1][1] @ dLF_diff).transpose() @ Jc_left
        self.g += self.weights[1] * (dJ_right @ v - self.K_gains[1][0] @ RF_diff - self.K_gains[1][1] @ dRF_diff).transpose() @ Jc_right
        self.g -= self.weights[2] * (dH - data.dAg @ v).transpose() @ data.Ag
        self.g += self.weights[3] * (dJ_base @ v - self.K_gains[3][0] @ base_diff - self.K_gains[3][1] @ dbase_diff).transpose() @ Jc_base
        self.g += self.weights[3] * (dJ_torso @ v - self.K_gains[3][0] @ torso_diff - self.K_gains[3][1] @ dtorso_diff).transpose() @ Jc_torso


    def solve(self, data, v, q_diff, dq_diff, LF_diff, RF_diff, dLF_diff, dRF_diff, dH, base_diff, dbase_diff, torso_diff, dtorso_diff):
        self.computeMatrice(data, v, q_diff, dq_diff, LF_diff, RF_diff, dLF_diff, dRF_diff, dH, base_diff, dbase_diff, torso_diff, dtorso_diff)

        self.qp.update(
            H=self.H,
            g=self.g,
            update_preconditioner=False,
        )
        self.qp.solve()
        qddot = self.qp.results.x[:]

        return qddot 

class IDSolver:
    def __init__(
        self, model, weights, nk, mu, contact_ids, force_size, verbose: bool
    ):
        kd = 2 * np.sqrt(100)
        baum_Kd = np.array([kd,kd,kd])
        self.baum_Kd = np.diag(baum_Kd)
        self.nk = nk
        self.contact_ids = contact_ids
        self.mu = mu
        self.force_size = force_size

        n = 2 * model.nv - 6 + force_size * nk 
        neq = model.nv + force_size * nk
        nin = 5 * nk

        self.A = np.zeros((model.nv + force_size * nk, 2 * model.nv - 6 + force_size * nk))
        self.b = np.zeros(model.nv + force_size * nk)
        self.l = np.zeros(5 * nk)
        self.C = np.zeros((5 * nk, 2 * model.nv - 6 + force_size * nk))

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((5, force_size))
        if force_size == 3:
            self.Cmin = np.array([
                [-1, 0, mu],
                [1, 0, mu],
                [-1, 0, mu],
                [1, 0, mu],
                [0, 0, 1]
            ])
        else:
            self.Cmin = np.array([
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])

        u = np.ones(5 * nk) * 100000
        g = np.zeros(n)
        H = np.zeros((n,n))
        H[:model.nv, :model.nv] = np.eye(model.nv) * weights[0]
        H[model.nv:model.nv + force_size * nk,model.nv:model.nv + force_size * nk] = np.eye(force_size * nk) * weights[1]

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 10
        qp.settings.max_iter_in = 10
        qp.init(H, g, self.A, self.b, self.C, self.l, u)
        self.qp = qp

        self.model = model
    
    def computeMatrice(self, data, cs, v, a, forces, M):
        nle = data.nle
        Jc = np.zeros((self.nk * self.force_size, self.model.nv))
        gamma = np.zeros(self.force_size * self.nk)
        for i in range(self.nk):
            if cs[i]:
                fJf = pin.getFrameJacobian(self.model, data, self.contact_ids[i], pin.LOCAL)[:self.force_size]
                Jdot = pin.getFrameJacobianTimeVariation(self.model, data, self.contact_ids[i], pin.LOCAL)[:self.force_size]
                Jc[i * self.force_size:(i+1) * self.force_size,:] = fJf
                gamma[i * self.force_size:(i+1) * self.force_size] = Jdot @ v
                #gamma[i * self.force_size:i * self.force_size + 3] += self.baum_Kp @ data.oMf[contact_ids[i]].rotation.T @ (data.oMf[contact_ids[i]].translation - feet_refs[i])
                gamma[i * self.force_size:i * self.force_size + 3] += self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).linear \
                  + self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).angular

        JcT = Jc.T
        self.A[:self.model.nv,:self.model.nv] = M 
        self.A[:self.model.nv,self.model.nv:self.model.nv + self.nk * self.force_size] = -JcT
        self.A[:self.model.nv,self.model.nv + self.nk * self.force_size:] = -self.S
        self.A[self.model.nv:,:self.model.nv] = Jc 

        self.b[:self.model.nv] = -nle - M @ a + JcT @ forces
        self.b[self.model.nv:] = -gamma - Jc @ a
        
        self.l = np.zeros(5 * self.nk)
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 5:(i+1)*5] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2]
                ]) 

        self.C = np.zeros((5 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for j in range(self.nk):
            if cs[i]:
                self.C[j * 5:(j + 1) * 5, self.model.nv + j * self.force_size:self.model.nv + (j + 1) * self.force_size] = self.Cmin


    def solve(self, data, cs, v, a, forces, M):
        self.computeMatrice(data, cs, v, a, forces, M)

        self.qp.update(
            A=self.A,
            b=self.b,
            C=self.C,
            l=self.l,
            update_preconditioner=False,
        )
        
        self.qp.solve()

        da = self.qp.results.x[:self.model.nv]
        anew = a + da
        dforces = self.qp.results.x[self.model.nv:self.model.nv + self.force_size * self.nk]
        new_forces = forces + dforces
        torque = self.qp.results.x[self.model.nv + self.force_size * self.nk:]

        return anew, new_forces, torque 

class IKIDSolver_f6:
    def __init__(
        self, model, weights, K_gains, nk, mu, contact_ids, base_id, torso_id, force_size, verbose: bool
    ):
        
        self.K_gains = K_gains

        kd = 2 * np.sqrt(10)
        baum_Kd = np.array([kd,kd,kd])
        self.baum_Kd = np.diag(baum_Kd)
        self.nk = nk
        self.contact_ids = contact_ids
        self.base_id = base_id
        self.torso_id = torso_id
        self.mu = mu
        self.force_size = force_size
        self.weights = weights

        n = 2 * model.nv - 6 + force_size * nk 
        neq = model.nv + force_size * nk
        nin = 5 * nk
        
        self.A = np.zeros((model.nv + force_size * nk, 2 * model.nv - 6 + force_size * nk))
        self.b = np.zeros(model.nv + force_size * nk) #
        self.l = np.zeros(5 * nk)
        self.C = np.zeros((5 * nk, 2 * model.nv - 6 + force_size * nk))

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((5, force_size))
        if force_size == 3:
            self.Cmin = np.array([
                [-1, 0, mu],
                [1, 0, mu],
                [-1, 0, mu],
                [1, 0, mu],
                [0, 0, 1]
            ])
        else:
            self.Cmin = np.array([
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])

        u = np.ones(5 * nk) * 100000
        self.g = np.zeros(n)
        self.H = np.zeros((n,n))
        self.H[model.nv:model.nv + force_size * nk,model.nv:model.nv + force_size * nk] = np.eye(force_size * nk) * weights[4]

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 100
        qp.settings.max_iter_in = 100
        qp.init(self.H, self.g, self.A, self.b, self.C, self.l, u)
        self.qp = qp

        self.model = model
    
    def computeMatrice(self, data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M):
        nle = data.nle
        Jc_left = pin.getFrameJacobian(self.model, data, self.contact_ids[0], pin.LOCAL) 
        Jc_right = pin.getFrameJacobian(self.model, data, self.contact_ids[1], pin.LOCAL) 
        dJ_left = pin.getFrameJacobianTimeVariation(self.model, data, self.contact_ids[0], pin.LOCAL)
        dJ_right = pin.getFrameJacobianTimeVariation(self.model, data, self.contact_ids[1], pin.LOCAL)
        Jc_base = pin.getFrameJacobian(self.model, data, self.base_id, pin.LOCAL)[3:]
        Jc_torso = pin.getFrameJacobian(self.model, data, self.torso_id, pin.LOCAL)[3:]
        dJ_base = pin.getFrameJacobianTimeVariation(self.model, data, self.base_id, pin.LOCAL)[3:]
        dJ_torso = pin.getFrameJacobianTimeVariation(self.model, data, self.torso_id, pin.LOCAL)[3:]
        
        self.H[:self.model.nv,:self.model.nv] = self.weights[0] * np.eye(self.model.nv)
        self.H[:self.model.nv,:self.model.nv] += self.weights[1] * Jc_left.transpose() @ Jc_left
        self.H[:self.model.nv,:self.model.nv] += self.weights[1] * Jc_right.transpose() @ Jc_right
        self.H[:self.model.nv,:self.model.nv] += self.weights[2] * data.Ag.transpose() @ data.Ag
        self.H[:self.model.nv,:self.model.nv] += self.weights[3] * Jc_base.transpose() @ Jc_base
        self.H[:self.model.nv,:self.model.nv] += self.weights[3] * Jc_torso.transpose() @ Jc_torso
       
        self.g[:self.model.nv] = self.weights[0] * (- self.K_gains[0][0] @ q_diff - self.K_gains[0][1] @ dq_diff )
        self.g[:self.model.nv] += self.weights[1] * (dJ_left @ v - self.K_gains[1][0] @ LF_diff - self.K_gains[1][1] @ dLF_diff).transpose() @ Jc_left
        self.g[:self.model.nv] += self.weights[1] * (dJ_right @ v - self.K_gains[1][0] @ RF_diff - self.K_gains[1][1] @ dRF_diff).transpose() @ Jc_right
        self.g[:self.model.nv] -= self.weights[2] * (dH - data.dAg @ v).transpose() @ data.Ag
        self.g[:self.model.nv] += self.weights[3] * (dJ_base @ v - self.K_gains[3][0] @ base_diff - self.K_gains[3][1] @ dbase_diff).transpose() @ Jc_base
        self.g[:self.model.nv] += self.weights[3] * (dJ_torso @ v - self.K_gains[3][0] @ torso_diff - self.K_gains[3][1] @ dtorso_diff).transpose() @ Jc_torso
        
        self.A[:self.model.nv,:self.model.nv] = M
        if cs[0]:
            self.A[:self.model.nv,self.model.nv:self.model.nv + 6] = -Jc_left.transpose()
        else:
            self.A[:self.model.nv,self.model.nv:self.model.nv + 6] = np.zeros((self.model.nv, 6))
        if cs[1]:
            self.A[:self.model.nv,self.model.nv + 6:self.model.nv + 12] = -Jc_right.transpose()
        else:
            self.A[:self.model.nv,self.model.nv + 6:self.model.nv + 12] = np.zeros((self.model.nv, 6))

        self.A[:self.model.nv,self.model.nv + 12:] = -self.S

        if cs[0]:
            self.A[self.model.nv:self.model.nv + 6,:self.model.nv] = Jc_left 
        else:
            self.A[self.model.nv:self.model.nv + 6,:self.model.nv] = np.zeros((6, self.model.nv))
        
        if cs[1]:
            self.A[self.model.nv + 6:self.model.nv + 12,:self.model.nv] = Jc_right
        else:
            self.A[self.model.nv + 6:self.model.nv + 12,:self.model.nv] = np.zeros((6, self.model.nv))

        self.b[:self.model.nv] = -nle
        self.b[self.model.nv:] = np.zeros(12)
        if cs[0]:
            self.b[:self.model.nv] += Jc_left.transpose() @ forces[:6]
            self.b[self.model.nv:self.model.nv + 6] = -dJ_left @ v
            #self.b[self.model.nv:self.model.nv + 3] -= self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[0]).linear
        if cs[1]:
            self.b[:self.model.nv] += Jc_right.transpose() @ forces[6:]
            self.b[self.model.nv + 6:self.model.nv + 12] = -dJ_right @ v
            #self.b[self.model.nv + 6:self.model.nv + 9] -= self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[1]).linear
        
        self.l = np.zeros(5 * self.nk)
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 5:(i+1)*5] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2]
                ]) 

        self.C = np.zeros((5 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for j in range(self.nk):
            if cs[i]:
                self.C[j * 5:(j + 1) * 5, self.model.nv + j * self.force_size:self.model.nv + (j + 1) * self.force_size] = self.Cmin


    def solve(self, data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M):
        self.computeMatrice(data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M)

        self.qp.update(
            H=self.H,
            g=self.g,
            A=self.A,
            b=self.b,
            C=self.C,
            l=self.l,
            update_preconditioner=False,
        )
        
        self.qp.solve()

        anew = self.qp.results.x[:self.model.nv]
        dforces = self.qp.results.x[self.model.nv:self.model.nv + self.force_size * self.nk]
        new_forces = forces + dforces
        torque = self.qp.results.x[self.model.nv + self.force_size * self.nk:]

        return anew, new_forces, torque 

class IKIDSolver_f3:
    def __init__(
        self, model, weights, K_gains, nk, mu, sole_ids, contact_ids, base_id, torso_id, verbose: bool
    ):
        kd = 2 * np.sqrt(10)
        baum_Kd = np.array([kd,kd,kd])
        self.baum_Kd = np.diag(baum_Kd)
        self.nk = nk
        self.contact_ids = contact_ids
        self.sole_ids = sole_ids
        self.mu = mu
        self.base_id = base_id
        self.torso_id = torso_id
        self.weights = weights
        self.alpha = 0.1
        self.K_gains = K_gains

        n = 2 * model.nv - 6 + 3 * nk 
        neq = model.nv + 3 * nk
        nin = 5 * nk
        
        self.H = np.zeros((n,n))
        self.g = np.zeros(n)
        self.A = np.zeros((model.nv + 3 * nk, 2 * model.nv - 6 + 3 * nk))
        self.b = np.zeros(model.nv + 3 * nk)
        self.l = np.zeros(5 * nk)
        self.C = np.zeros((5 * nk, 2 * model.nv - 6 + 3 * nk))

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((5, 3))
        self.Cmin = np.array([
            [-1, 0, mu],
            [1, 0, mu],
            [-1, 0, mu],
            [1, 0, mu],
            [0, 0, 1]
        ])

        u = np.ones(5 * nk) * 100000
        g = np.zeros(n)
        H = np.zeros((n,n))
        self.H[model.nv:model.nv + 3 * nk,model.nv:model.nv + 3 * nk] = np.eye(3 * nk) * weights[4]

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 100
        qp.settings.max_iter_in = 100
        qp.init(H, g, self.A, self.b, self.C, self.l, u)
        self.qp = qp

        self.model = model
    
    def computeMatrice(self, data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M):
        nle = data.nle
        
        Jc_left = pin.getFrameJacobian(self.model, data, self.sole_ids[0], pin.LOCAL) 
        Jc_right = pin.getFrameJacobian(self.model, data, self.sole_ids[1], pin.LOCAL) 
        dJ_left = pin.getFrameJacobianTimeVariation(self.model, data, self.sole_ids[0], pin.LOCAL)
        dJ_right = pin.getFrameJacobianTimeVariation(self.model, data, self.sole_ids[1], pin.LOCAL)
        Jc_base = pin.getFrameJacobian(self.model, data, self.base_id, pin.LOCAL)[3:]
        Jc_torso = pin.getFrameJacobian(self.model, data, self.torso_id, pin.LOCAL)[3:]
        dJ_base = pin.getFrameJacobianTimeVariation(self.model, data, self.base_id, pin.LOCAL)[3:]
        dJ_torso = pin.getFrameJacobianTimeVariation(self.model, data, self.torso_id, pin.LOCAL)[3:]

        Jc = np.zeros((self.nk * 3, self.model.nv))
        gamma = np.zeros(3 * self.nk)
        for i in range(self.nk):
            if cs[i]:
                fJf = pin.getFrameJacobian(self.model, data, self.contact_ids[i], pin.LOCAL)[:3]
                Jdot = pin.getFrameJacobianTimeVariation(self.model, data, self.contact_ids[i], pin.LOCAL)[:3]
                Jc[i * 3:(i+1) * 3,:] = fJf
                gamma[i * 3:(i+1) * 3] = Jdot @ v
                gamma[i * 3:i * 3 + 3] += self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).linear \
                  + self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).angular
        
        self.H[:self.model.nv,:self.model.nv] = self.weights[0] * np.eye(self.model.nv)
        self.H[:self.model.nv,:self.model.nv] += self.weights[1] * Jc_left.transpose() @ Jc_left
        self.H[:self.model.nv,:self.model.nv] += self.weights[1] * Jc_right.transpose() @ Jc_right
        self.H[:self.model.nv,:self.model.nv] += self.weights[2] * data.Ag.transpose() @ data.Ag
        self.H[:self.model.nv,:self.model.nv] += self.weights[3] * Jc_base.transpose() @ Jc_base
        self.H[:self.model.nv,:self.model.nv] += self.weights[3] * Jc_torso.transpose() @ Jc_torso
       
        self.g[:self.model.nv] = self.weights[0] * (- self.K_gains[0][0] @ q_diff - self.K_gains[0][1] @ dq_diff )
        self.g[:self.model.nv] += self.weights[1] * (dJ_left @ v - self.K_gains[1][0] @ LF_diff - self.K_gains[1][1] @ dLF_diff).transpose() @ Jc_left
        self.g[:self.model.nv] += self.weights[1] * (dJ_right @ v - self.K_gains[1][0] @ RF_diff - self.K_gains[1][1] @ dRF_diff).transpose() @ Jc_right
        self.g[:self.model.nv] -= self.weights[2] * (dH - data.dAg @ v).transpose() @ data.Ag
        self.g[:self.model.nv] += self.weights[3] * (dJ_base @ v - self.K_gains[3][0] @ base_diff - self.K_gains[3][1] @ dbase_diff).transpose() @ Jc_base
        self.g[:self.model.nv] += self.weights[3] * (dJ_torso @ v - self.K_gains[3][0] @ torso_diff - self.K_gains[3][1] @ dtorso_diff).transpose() @ Jc_torso

        JcT = Jc.T
        self.A[:self.model.nv,:self.model.nv] = M 
        self.A[:self.model.nv,self.model.nv:self.model.nv + self.nk * 3] = -JcT
        self.A[:self.model.nv,self.model.nv + self.nk * 3:] = -self.S
        self.A[self.model.nv:,:self.model.nv] = Jc 

        self.b[:self.model.nv] = -nle + JcT @ forces
        self.b[self.model.nv:] = -gamma 
        
        self.l = np.zeros(5 * self.nk)
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 5:(i+1)*5] = np.array([
                    forces[i * 3] - forces[i * 3 + 2] * self.mu,
                    -forces[i * 3] - forces[i * 3 + 2] * self.mu,
                    forces[i * 3 + 1] - forces[i * 3 + 2] * self.mu,
                    -forces[i * 3 + 1] - forces[i * 3 + 2] * self.mu,
                    - forces[i * 3 + 2]
                ])  

        self.C = np.zeros((5 * self.nk, 2 * self.model.nv - 6 + 3 * self.nk))
        for j in range(self.nk):
            if cs[i]:
                self.C[j * 5:(j + 1) * 5, self.model.nv + j * 3:self.model.nv + (j + 1) * 3] = self.Cmin


    def solve(self, data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M):
        self.computeMatrice(data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M)

        self.qp.update(
            H=self.H,
            g=self.g,
            A=self.A,
            b=self.b,
            C=self.C,
            l=self.l,
            update_preconditioner=True,
        )
        
        self.qp.solve()

        anew = self.qp.results.x[:self.model.nv]
        dforces = self.qp.results.x[self.model.nv:self.model.nv + 3 * self.nk]
        new_forces = forces + dforces
        torque = self.qp.results.x[self.model.nv + 3 * self.nk:]

        return anew, new_forces, torque 

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
    LF_diff[3:] = -pin.log3(rdata.oMf[LF_id].rotation)
    RF_diff = np.zeros(6)
    RF_diff[:3] = RF_refs[0].translation - rdata.oMf[RF_id].translation
    RF_diff[3:] = -pin.log3(rdata.oMf[RF_id].rotation)

    dLF_diff = np.zeros(6)
    dLF_diff[:3] = (LF_refs[1].translation - LF_refs[0].translation) / dt - LF_vel_lin
    dLF_diff[3:] = -LF_vel_ang
    dRF_diff = np.zeros(6)
    dRF_diff[:3] = (RF_refs[1].translation - RF_refs[0].translation) / dt - RF_vel_lin
    dRF_diff[3:] = -RF_vel_ang

    base_diff = -pin.log3(rdata.oMf[base_id].rotation)
    torso_diff = -pin.log3(rdata.oMf[torso_id].rotation)
    dbase_diff = -pin.getFrameVelocity(rmodel,rdata,base_id,pin.LOCAL).angular
    dtorso_diff = -pin.getFrameVelocity(rmodel,rdata,torso_id,pin.LOCAL).angular

    return q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff