"""
This script launches a locomotion MPC scheme which solves repeatedly an 
optimal control problem based on the centroidal model of the humanoid robot Talos. 
The contacts forces are modeled as 6D wrenches. 
"""

import numpy as np
import aligator
import pinocchio as pin
from bullet_robot import BulletRobot
import time
from scipy.spatial.transform import Rotation as R

import copy
from talos_utils import (
    loadTalos, 
    URDF_FILENAME,
    modelPath,
    shapeState,
    footTrajectory,
    save_trajectory,
    update_timings,
    compute_ID_references,
)

from QP_utils import (
   IKIDSolver_f6
)

from aligator import (manifolds, 
                    dynamics, 
                    constraints,)

rmodelComplete, rmodel, qComplete, q0 = loadTalos()
rdata = rmodel.createData()

low_limits = rmodel.lowerPositionLimit[7:]
up_limits = rmodel.upperPositionLimit[7:]

nq = rmodel.nq
nv = rmodel.nv
nk = 2
force_size = 6
nu = nk * force_size
nx = 9
space = manifolds.VectorSpace(nx)
space_multibody = manifolds.MultibodyPhaseSpace(rmodel)
x0 = space.neutral()
u_min = -rmodel.effortLimit[6:]
u_max = rmodel.effortLimit[6:]

def yawRotation(yaw):
    Ro = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Ro

r=R.from_matrix(yawRotation(1.5))
#q0[3:7] = r.as_quat()
x0_multibody = np.concatenate((q0, np.zeros(rmodel.nv)))
u0 = np.zeros(nu)
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
com0 = pin.centerOfMass(rmodel, rdata, q0)
x0[:3] = com0.copy()

gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
Lfoot = 0.1
Wfoot = 0.075
mass = pin.computeTotalMass(rmodel)
f_ref = np.array([0, 0, -mass * gravity[2] / nk])
for i in range(nk):
    u0[force_size * i + 2] = -gravity[2] * mass / nk

controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [rmodelComplete.getJointId(name_joint) for name_joint in controlled_joints[1:]]
umax = rmodel.effortLimit[6:]

LF_id = rmodel.getFrameId("left_sole_link")
RF_id = rmodel.getFrameId("right_sole_link")
sole_ids = [LF_id, RF_id]
base_id = rmodel.getFrameId("base_link")
torso_id = rmodel.getFrameId("torso_2_link")
name_sols = ["left_sole_link", "right_sole_link"]

""" Initialize simulation """
device = BulletRobot(controlled_joints,
                        modelPath,
                        URDF_FILENAME,
                        1e-3,
                        rmodelComplete,
                        q0[:3])
device.initializeJoints(qComplete)
#device.changeCamera(1., 50, -15, [1.7, -0.5, 1.2])
device.changeCamera(1., 90, -5, [1, 0, 1])
q_current, v_current = device.measureState()

""" Define gait and time parameters"""
T_ds = 20 # Double support time
T_ss = 80 # Singel support time

dt = 0.01
nsteps = 100
Nsimu = int(dt / 0.001)

""" Define contact sequence throughout horizon"""
total_steps = 1
contact_phases = [[True,True]] * T_ds
for s in range(total_steps):
    contact_phases += [[True,False]] * T_ss + \
                      [[True,True]] * T_ds + \
                      [[False,True]] * T_ss + \
                      [[True,True]] * T_ds

contact_phases += [[True,True]] * nsteps * 2

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

f_full = -mass * gravity[2]
f_half = -mass * gravity[2] / 2.
urefs = []
for i in range(total_steps):
    for j in range(T_ds):
        un = np.zeros(nu)
        if i == 0:
            un[2] = f_full * j / T_ds + f_half * (T_ds - j) / T_ds
            un[8] = f_half * (T_ds - j) / T_ds 
        else:
            un[2] = f_full * (j + 1) / T_ds
            un[8] = f_full * (T_ds - j) / T_ds 
        urefs.append(un)
    for j in range(T_ss):
        un = np.zeros(nu)
        un[2] = f_full
        urefs.append(un)
    for j in range(T_ds):
        un = np.zeros(nu)
        un[2] = f_full * (T_ds - j) / T_ds
        un[8] = f_full * (j + 1) / T_ds 
        urefs.append(un)
    for j in range(T_ss):
        un = np.zeros(nu)
        un[8] = f_full
        urefs.append(un)

for j in range(T_ds):
    un = np.zeros(nu)
    un[8] = f_half * (j + 1) / float(T_ds)
    un[2] = f_full * (T_ds - j) / float(T_ds) + f_half * j / float(T_ds)
    urefs.append(un)

for j in range(nsteps * 2):
    un = np.zeros(nu)
    un[2] = f_half 
    un[8] = f_half 
    urefs.append(un)

T_mpc = len(contact_phases)  # Size of the problem

""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0.2
y_forward = 0.0
foot_yaw = 0
y_gap = 0.18
x_depth = 0

foottraj = footTrajectory(
    rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), T_ss, T_ds, nsteps, swing_apex, x_forward, y_forward, foot_yaw, y_gap, x_depth
)

""" Create dynamics and costs """

w_centroidal_com = np.diag(np.array([0,0,0]))
w_linear_mom = np.diag(np.array([0.01,0.01,100]))
w_linear_acc = 0.01 * np.eye(3)
w_angular_mom = np.diag(np.array([0.1,0.1,1000]))
w_angular_acc = 0.01 * np.eye(3)

w_control_linear = np.ones(3) * 0.001
w_control_angular = np.ones(3) * 0.1
w_control = np.diag(np.concatenate((
    w_control_linear,
    w_control_angular,
    w_control_linear,
    w_control_angular
)))

def create_dynamics(contact_map):
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(contact_state, LF_pose, RF_pose, ur):
    contact_pose = [LF_pose.translation, RF_pose.translation]
    contact_map = aligator.ContactMap(name_sols, contact_state, contact_pose)

    rcost = aligator.CostStack(space, nu)

    linear_acc = aligator.CentroidalAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    angular_acc = aligator.AngularAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.zeros(3))
    angular_mom = aligator.AngularMomentumResidual(nx, nu, np.zeros(3))
    centroidal_com = aligator.CentroidalCoMResidual(nx, nu, com0)

    rcost.addCost("state_cost",
        aligator.QuadraticControlCost(space, ur, w_control))
    rcost.addCost("com_cost",
        aligator.QuadraticResidualCost(space, centroidal_com, w_centroidal_com)
    )
    rcost.addCost("linear_mom_cost",
        aligator.QuadraticResidualCost(space, linear_mom, w_linear_mom)
    )
    rcost.addCost("angular_mom_cost",
        aligator.QuadraticResidualCost(space, angular_mom, w_angular_mom)
    )
    rcost.addCost("angular_acc_cost",
        aligator.QuadraticResidualCost(space, angular_acc, w_angular_acc)
    )
    rcost.addCost("linear_acc_cost",
        aligator.QuadraticResidualCost(space, linear_acc, w_linear_acc)
    )
    stm = aligator.StageModel(rcost, create_dynamics(contact_map))
    for i in range(len(contact_state)):
        if contact_state[i]:
            cone_cstr = aligator.CentroidalWrenchConeResidual(space.ndx, nu, i, mu, Lfoot, Wfoot)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())

    return stm

term_cost = aligator.CostStack(space, nu)

""" Create the optimal problem and the full horizon """
stages = []
for i in range(nsteps):
    stages.append(createStage(contact_phases[0], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), urefs[0]))

stages_full = [createStage(contact_phases[i], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), urefs[i]) for i in range(T_mpc)]
stages_full_data = []
for i in range(T_mpc):
    stages_full_data.append(stages_full[i].createData())

problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Parametrize the solver"""

TOL = 1e-5
mu_init = 1e-8 

rho_init = 0.0
max_iters = 100
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(TOL, mu_init, rho_init)
#solver = aligator.SolverFDDP(TOL, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_LINEAR
#print("LDLT algo choice:", solver.ldlt_algo_choice)
solver.linear_solver_choice = aligator.LQ_SOLVER_PARALLEL #LQ_SOLVER_SERIAL
solver.force_initial_condition = True
solver.setNumThreads(2)
solver.max_iters = max_iters

solver.setup(problem)

us_init = [u0 for _ in range(nsteps)]
xs_init = [x0] * (nsteps + 1)

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

x_measured = shapeState(
    q_current, 
    v_current, 
    nq, 
    nq + nv, 
    controlled_ids
)

g_p = 400
g_h = 10
g_b = 10
K_gains = []
g_q = np.diag(np.array([
    0, 0, 0, 10, 10, 10,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    1, 1,
    10, 10, 10, 10,
    10, 10, 10, 10
]))
K_gains.append([g_q * 10, 2 * np.sqrt(g_q * 10)])
K_gains.append([np.eye(6) * g_p, np.eye(6) * 2 * np.sqrt(g_p)])
K_gains.append([np.eye(6) * g_h, np.eye(6) * 2 * np.sqrt(g_h)])
K_gains.append([np.eye(3) * g_b, np.eye(3) * 2 * np.sqrt(g_b)])

weights_IKID = [500, 50000, 10, 1000, 100]  # qref, foot_pose, centroidal, base_rot, force
IKID_solver = IKIDSolver_f6(rmodel, weights_IKID, K_gains, nk, mu, Lfoot, Wfoot, sole_ids, base_id, torso_id, force_size, False)

qdot = np.zeros(rmodel.nv)
qddot = np.zeros(rmodel.nv)
LF_vel_ref = np.zeros(6)
RF_vel_ref = np.zeros(6)

device.showTargetToTrack(rdata.oMf[LF_id], rdata.oMf[RF_id])

""" Launch the MPC loop"""
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

fd = 100
theta = 6 * np.pi / 4
f_disturbance = [np.cos(theta)* fd, np.sin(theta) * fd, 0]

for t in range(T_mpc):
    print("Time " + str(t))
    
    takeoff_RF, takeoff_LF, land_RF, land_LF = update_timings(
        land_LFs, land_RFs, takeoff_LFs, takeoff_RFs
    )
    
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )
    if land_RF == -1: 
        foottraj.updateForward(0, 0, y_gap, y_forward, -0.01, 0, swing_apex)

    LF_refs, RF_refs = foottraj.updateTrajectory(
        takeoff_RF, takeoff_LF, land_RF, land_LF, rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()
    )
    device.moveMarkers(LF_refs[0].translation, RF_refs[0].translation)

    for n in range(nsteps):
        contact_state = problem.stages[n].dynamics.differential_dynamics.contact_map.contact_states
        if contact_state[0]:
            problem.stages[n].dynamics.differential_dynamics.contact_map.contact_poses[0] = LF_refs[n].translation
            problem.stages[n].cost.getComponent("angular_acc_cost").residual.contact_map.contact_poses[0] = LF_refs[n].translation
            problem.stages[n].cost.getComponent("linear_acc_cost").residual.contact_map.contact_poses[0] = LF_refs[n].translation 
        
        if contact_state[1]:
            problem.stages[n].dynamics.differential_dynamics.contact_map.contact_poses[1] = RF_refs[n].translation
            problem.stages[n].cost.getComponent("angular_acc_cost").residual.contact_map.contact_poses[1] = RF_refs[n].translation
            problem.stages[n].cost.getComponent("linear_acc_cost").residual.contact_map.contact_poses[1] = RF_refs[n].translation 
    
    contact_state = problem.stages[0].dynamics.differential_dynamics.contact_map.contact_states
    
    if problem.stages[0].dynamics.differential_dynamics.contact_map.contact_states[0]:
        force_left.append(us[0][:3])
        torque_left.append(us[0][3:6])
    else:
        force_left.append(np.zeros(3))
        torque_left.append(np.zeros(3))
    if problem.stages[0].dynamics.differential_dynamics.contact_map.contact_states[1]:
        force_right.append(us[0][6:9])
        torque_right.append(us[0][9:])
    else:
        force_right.append(np.zeros(3))
        torque_right.append(np.zeros(3))

    LF_measured.append(rdata.oMf[LF_id].copy())
    RF_measured.append(rdata.oMf[RF_id].copy())
    LF_references.append(LF_refs[0])
    RF_references.append(RF_refs[0])

    """ Compute various references for ID """
    q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff = \
      compute_ID_references(space_multibody, rmodel, rdata, LF_id, RF_id, base_id, torso_id, x0_multibody, x_measured, LF_refs, RF_refs, dt)
    dH = solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.xdot[3:9]

    for j in range(Nsimu):
        q_current, v_current = device.measureState()
        
        x_measured = shapeState(q_current, 
                                v_current, 
                                nq, 
                                nq + nv, 
                                controlled_ids)

        new_x = np.zeros(9)
        new_x[:3] = pin.centerOfMass(rmodel, rdata, x_measured[:nq])
        pin.computeCentroidalMomentum(rmodel,rdata, x_measured[:nq], x_measured[nq:])
        new_x[3:6] = rdata.hg.linear
        new_x[6:] = rdata.hg.angular

        pin.forwardKinematics(rmodel, rdata, x_measured[:nq])
        pin.updateFramePlacements(rmodel, rdata)
        pin.computeJointJacobians(rmodel,rdata)
        pin.computeJointJacobiansTimeVariation(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        M = pin.crba(rmodel, rdata, x_measured[:nq])
        pin.nonLinearEffects(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        pin.dccrba(rmodel,rdata, x_measured[:nq], x_measured[nq:])
        
        forces = us[0] - solver.results.controlFeedbacks()[0] @ space.difference(new_x, xs[0])
        new_acc, new_forces, torque = IKID_solver.solve(
            rdata,
            contact_state, 
            x_measured[nq:], 
            q_diff, dq_diff, 
            LF_diff, dLF_diff, 
            RF_diff, dRF_diff, 
            base_diff, dbase_diff, 
            torso_diff, dtorso_diff, 
            forces, dH, M
        )

        device.execute(torque)
        x_multibody.append(x_measured)
        u_multibody.append(torque)
        """ if t >= 160 and t < 171:
            print("Force applied")
            device.apply_force(f_disturbance, [0, 0, 0])  """

    xs = xs[1:] + [xs[-1]]
    us = us[1:] + [us[-1]]
    xs[0] = new_x

    problem.x0_init = new_x
    problem.replaceStageCircular(stages_full[t])
    solver.cycleProblem(problem, stages_full_data[t])
    solver.run(problem, xs, us)

    xs = solver.results.xs.tolist().copy()
    us = solver.results.us.tolist().copy()
    K0 = solver.results.controlFeedbacks()[0]


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

""" save_trajectory(x_multibody, u_multibody, com_measured, force_left, force_right, torque_left, torque_right, solve_time, 
                LF_measured, RF_measured, LF_references, RF_references, L_measured, "centroidal_f6") """