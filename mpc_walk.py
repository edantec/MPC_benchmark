import numpy as np
import aligator
import pinocchio as pin
import matplotlib.pyplot as plt
from bullet_robot import BulletRobot
import example_robot_data
import time

from talos_utils import (
    shapeState,
    footTrajectory,
    update_timings,
    save_trajectory
)

from aligator import (manifolds, 
                    dynamics, 
                    constraints,)
from utils import get_endpoint_traj, compute_quasistatic, ArgsBase

DEFAULT_SAVE_DIR = "/home/edantec/Documents/MPC/examples/tmp"

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

robotComplete = example_robot_data.load("talos")
qComplete = robotComplete.model.referenceConfigurations["half_sitting"]

locked_joints = [20,21,22,23,28,29,30,31]
locked_joints += [32, 33]
robot = robotComplete.buildReducedRobot(locked_joints, qComplete)
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data
nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6
gravity = -9.81
print("nq:", nq)
print("nv:", nv)

FOOT_FRAME_IDS = {
    fname: rmodel.getFrameId(fname)
    for fname in ["left_sole_link", "right_sole_link"]
}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}

controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [robotComplete.model.getJointId(name_joint) for name_joint in controlled_joints[1:]]
q0 = rmodel.referenceConfigurations["half_sitting"]

pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)

device = BulletRobot(controlled_joints,
                        modelPath,
                        URDF_FILENAME,
                        1e-3,
                        robotComplete.model)
device.initializeJoints(qComplete)
q_current, v_current = device.measureState()

space = manifolds.MultibodyPhaseSpace(rmodel)

x0 = np.concatenate((q0, np.zeros(nv)))
act_matrix = np.eye(nv, nu, -6)
u0 = np.zeros(rmodel.nv - 6)
com0 = pin.centerOfMass(rmodel,rdata,x0[:nq])

w_x = np.array([
    0, 0, 0, 100, 100, 100, # Base pos/ori
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Left leg
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Right leg
    10, 10, # Torso
    0.1, 0.1, 0.1, 0.1, # Left arm
    0.1, 0.1, 0.1, 0.1, # Right arm
    1, 1, 1, 1, 1, 1, # Base pos/ori vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01, # Left leg vel
    0.1, 0.1, 0.1, 0.1, 0.01, 0.01, # Right leg vel
    10, 10, # Torso vel
    0.1, 0.1, 0.1, 0.1, # Left arm vel
    0.1, 0.1, 0.1, 0.1, # Right arm vel
]) 
w_x = np.diag(w_x) 
w_u = np.eye(nu) * 1e-5
w_LFRF = 100 * np.eye(6)
w_com = 10 * np.ones(3)
w_com = np.diag(w_com) 

act_matrix = np.eye(nv, nu, -6)
prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
constraint_models = []
constraint_datas = []
for fname, fid in FOOT_FRAME_IDS.items():
    joint_id = FOOT_JOINT_IDS[fname]
    pl1 = rmodel.frames[fid].placement
    pl2 = rdata.oMf[fid]
    cm = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        rmodel,
        joint_id,
        pl1,
        0,
        pl2,
        pin.LOCAL,
    )
    cm.corrector.Kp[:] = (0.1, 0.1, 1, 0.1, 0.1, 0.1)
    cm.corrector.Kd[:] = (5, 5, 5, 5, 5, 5)
    cm.name = fname
    constraint_models.append(cm)
    constraint_datas.append(cm.createData())


def create_dynamics(stage_space, cs):
    dyn_model = None
    if cs[0] and not(cs[1]):
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, [constraint_models[0]], prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    elif cs[1] and not(cs[0]):
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, [constraint_models[1]], prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    else:
        ode = dynamics.MultibodyConstraintFwdDynamics(stage_space, act_matrix, constraint_models, prox_settings)
        dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)
    return dyn_model 

LF_id = rmodel.getFrameId("left_sole_link")
RF_id = rmodel.getFrameId("right_sole_link")
root_id = rmodel.getFrameId("root_joint")
LF_placement = rdata.oMf[LF_id]
RF_placement = rdata.oMf[RF_id]
LF_init = rdata.oMf[LF_id].copy()
RF_init = rdata.oMf[RF_id].copy()

frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)
v_ref = pin.Motion()
v_ref.np[:] = 0.0

def createStage(cs, cs_previous, LF_target, RF_target):
    stage_rmodel = robot.model.copy()
    stage_space = manifolds.MultibodyPhaseSpace(stage_rmodel)

    frame_vel_LF = aligator.FrameVelocityResidual(stage_space.ndx, nu, rmodel, v_ref, LF_id, pin.LOCAL)
    frame_vel_RF = aligator.FrameVelocityResidual(stage_space.ndx, nu, rmodel, v_ref, RF_id, pin.LOCAL)

    frame_fn_LF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, rmodel, LF_target, LF_id)
    frame_fn_RF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, rmodel, RF_target, RF_id)
    frame_cs_RF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, rmodel, RF_target.translation, RF_id)[2]
    frame_cs_LF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, rmodel, LF_target.translation, LF_id)[2]
    
    #frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)

    rcost = aligator.CostStack(stage_space, nu)
    rcost.addCost(aligator.QuadraticStateCost(stage_space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(stage_space, u0, w_u))
    w_LF = np.zeros((6,6))
    w_RF = np.zeros((6,6))
    if cs[0] and not(cs[1]):
        w_RF = 1000 * np.eye(6)
    if cs[1] and not(cs[0]):
        w_LF = 1000 * np.eye(6)
    """ elif support == "DOUBLE":
        w_RF = 1000 * np.eye(6)
        w_LF = 1000 * np.eye(6) """
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_LF, w_LF))
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_RF, w_RF))
    #rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_com, w_com))


    stm = aligator.StageModel(rcost, create_dynamics(stage_space, cs))
    umax = rmodel.effortLimit[6:]
    umin = -umax
    """
    ctrl_fn = aligator.ControlErrorResidual(stage_space.ndx, np.zeros(nu))
    stm.addConstraint(ctrl_fn, constraints.BoxConstraint(umin, umax)) """

    if cs[1] and not(cs_previous[1]):
        stm.addConstraint(frame_cs_RF, constraints.EqualityConstraintSet())
        stm.addConstraint(frame_vel_RF, constraints.EqualityConstraintSet())
    if cs[0] and not(cs_previous[0]):
        stm.addConstraint(frame_cs_LF, constraints.EqualityConstraintSet()) 
        stm.addConstraint(frame_vel_LF, constraints.EqualityConstraintSet())
    return stm

term_cost = aligator.CostStack(space, nu)
#term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 100 * w_x))

""" Define gait and time parameters"""
T_ds = 10
T_ss = 60
dt = 0.01
nsteps = 100
Nsimu = int(dt / 0.001)

""" Define contact sequence throughout horizon"""
total_steps = 4
contact_phases = [[True,True]] * T_ds
for s in range(total_steps):
    contact_phases += [[True,False]] * T_ss + \
                      [[True,True]] * T_ds + \
                      [[False,True]] * T_ss + \
                      [[True,True]] * T_ds 

contact_phases += [[True,True]] * nsteps

takeoff_RFs = []
takeoff_LFs = []
land_RFs = []
land_LFs = []
for i in range(1, len(contact_phases)):
    if contact_phases[i] == [True, False] and contact_phases[i - 1] == [True, True]:
        takeoff_RFs.append(i + nsteps)
    elif contact_phases[i] == [False, True] and contact_phases[i - 1] == [True, True]:
        takeoff_LFs.append(i + nsteps)
    elif contact_phases[i] == [True, True] and contact_phases[i - 1] == [True, False]:
        land_RFs.append(i + nsteps)
    elif contact_phases[i] == [True, True] and contact_phases[i - 1] == [False, True]:
        land_LFs.append(i + nsteps)

Tmpc = len(contact_phases)

""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0.2
y_gap = 0.18
x_depth = 0.0

foottraj = footTrajectory(
    rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), T_ss, T_ds, nsteps, swing_apex, x_forward, y_gap, x_depth
)

stages_full = [createStage(contact_phases[0],contact_phases[0], LF_placement.copy(), RF_placement.copy())]
for i in range(1,Tmpc):
    stages_full.append(createStage(contact_phases[i],contact_phases[i-1], LF_placement.copy(), RF_placement.copy()))

stages = [createStage(contact_phases[0],contact_phases[0], LF_placement.copy(), RF_placement.copy())] * nsteps
problem = aligator.TrajOptProblem(x0, stages, term_cost)

TOL = 1e-5
mu_init = 1e-8 

rho_init = 0.0
max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init) #, verbose=verbose)
#solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
#print("LDLT algo choice:", solver.ldlt_algo_choice)
#solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL #LQ_SOLVER_SERIAL
solver.force_initial_condition = True
#solver.setNumThreads(8)
solver.max_iters = max_iters

solver.setup(problem)

us_init = [np.zeros(nu)] * nsteps
xs_init = [x0] * (nsteps + 1) #aligator.rollout(dyn_model, x0, us_init).tolist()

solver.run(
    problem,
    xs_init,
    us_init,
)

workspace = solver.workspace
results = solver.results
print(results)

xs = results.xs.tolist().copy()
us = results.us.tolist().copy()
K0 = results.controlFeedbacks()[0]

solver.max_iters = 1

x_measured = shapeState(q_current, 
                        v_current, 
                        nq, 
                        nq + nv, 
                        controlled_ids)

force_left = []
force_right = []
torque_left = []
torque_right = []
LF_measured = []
RF_measured = []
LF_references = []
RF_references = []
x_multibody = []
u_multibody = []
com_measured = []
solve_time = []
L_measured = []

device.showTargetToTrack(LF_placement, RF_placement)
for t in range(Tmpc):
    #print("Time " + str(t))

    pin.forwardKinematics(rmodel, rdata, x_measured[:nq])
    pin.updateFramePlacements(rmodel, rdata)

    takeoff_RF, takeoff_LF, land_RF, land_LF = update_timings(
        land_LFs, land_RFs, takeoff_LFs, takeoff_RFs
    )
    
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    LF_refs, RF_refs = foottraj.updateTrajectory(
        takeoff_RF, takeoff_LF, land_RF, land_LF, rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()
    )

    for j in range(nsteps):
        problem.stages[j].cost.components[2].residual.setReference(LF_refs[j])
        problem.stages[j].cost.components[3].residual.setReference(RF_refs[j])

    if problem.stages[0].dyn_model.differential_dynamics.constraint_models.__len__() == 1:
        if problem.stages[0].dyn_model.differential_dynamics.constraint_models[0].name == 'left_sole_link':
            force_left.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.linear)
            force_right.append(np.zeros(3))
            torque_left.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.angular)
            torque_right.append(np.zeros(3))
        else:
            force_right.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.linear)
            force_left.append(np.zeros(3))
            torque_right.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.angular)
            torque_left.append(np.zeros(3))
    elif problem.stages[0].dyn_model.differential_dynamics.constraint_models.__len__() == 2:
        force_left.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.linear)
        force_right.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[1].contact_force.linear)
        torque_left.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[0].contact_force.angular)
        torque_right.append(solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.constraint_datas[1].contact_force.angular)
    else:
        force_right.append(np.zeros(3))
        force_left.append(np.zeros(3))
        torque_left.append(np.zeros(3))
        torque_right.append(np.zeros(3))
    
    u_multibody.append(us[0])
    x_multibody.append(xs[0])
    LF_measured.append(rdata.oMf[LF_id].copy())
    RF_measured.append(rdata.oMf[RF_id].copy())
    LF_references.append(LF_refs[0].copy())
    RF_references.append(RF_refs[0].copy())
    com_measured.append(pin.centerOfMass(rmodel, rdata, x_measured[:nq]))
    pin.computeCentroidalMomentum(rmodel,rdata, x_measured[:nq], x_measured[nq:])
    L_measured.append(rdata.hg.angular.copy())

    """ if t == 300:
        for s in range(nsteps):
            device.resetState(xs[s][:rmodel.nq])
            time.sleep(0.1)
            print("s = " + str(s))
            LF_ref = problem.stages[s].cost.components[2].residual.getReference().translation
            RF_ref = problem.stages[s].cost.components[3].residual.getReference().translation
            device.moveMarkers(LF_ref, RF_ref)
        exit() """

    device.moveMarkers(LF_refs[0].translation, RF_refs[0].translation)
    problem.replaceStageCircular(stages_full[t])

    com_final = com0.copy()
    com_final[:2] = (LF_refs[-1].translation[:2] + RF_refs[-1].translation[:2]) / 2
    com_cstr = aligator.CentroidalCoMResidual(space.ndx, nu, com_final)
    term_constraint_com = aligator.StageConstraint(
        com_cstr, constraints.EqualityConstraintSet()
    )
    problem.removeTerminalConstraint()
    problem.addTerminalConstraint(term_constraint_com)

    for j in range(Nsimu):
        time.sleep(0.001)
        q_current, v_current = device.measureState()
        
        x_measured = shapeState(q_current, 
                                v_current, 
                                nq, 
                                nq + nv, 
                                controlled_ids)  

        current_torque = us[0] - K0 @ space.difference(x_measured, xs[0])
        device.execute(current_torque)

    xs = xs[1:] + [xs[-1]]
    us = us[1:] + [us[-1]]
    xs[0] = x_measured

    problem.x0_init = x_measured
    """ problem.addTerminalConstraint(term_cstr) """
    solver.setup(problem)
    start = time.time()
    solver.run(problem, xs, us)
    end = time.time()
    solve_time.append(end - start)
    print("solver.run = " + str(end - start))

    xs = solver.results.xs.tolist().copy()
    us = solver.results.us.tolist().copy()
    K0 = solver.results.controlFeedbacks()[0]

print("Elapsed time:")
print(np.mean(np.array(solve_time)))


force_left = np.array(force_left)
force_right = np.array(force_right)
torque_left = np.array(torque_left)
torque_right = np.array(torque_right)
solve_time = np.array(solve_time)
LF_measured = np.array(LF_measured)
RF_measured = np.array(RF_measured)
LF_references = np.array(LF_references)
RF_references = np.array(RF_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

save_trajectory(x_multibody, u_multibody, com_measured, force_left, force_right, torque_left, torque_right, solve_time, 
                LF_measured, RF_measured, LF_references, RF_references, L_measured, "fulldynamics")

Tn = force_left.shape[0]
ttlin = np.linspace(0,Tn * dt, Tn)

BINS = 50
plt.figure(1)
plt.title("Time consumption for FDDP")
plt.hist(solve_time, bins=BINS)
plt.xlabel("Time (s)")
plt.ylabel("Occurences")
plt.grid(True)
#plt.savefig("plots/fddp.png")


plt.show()