import numpy as np
import aligator
import pinocchio as pin
from bullet_robot import BulletRobot
import time

import copy
from talos_utils import (
    loadTalos, 
    URDF_FILENAME,
    modelPath,
    IKIDSolver_f3,
    shapeState,
    footTrajectory,
    save_trajectory,
    update_timings,
    compute_ID_references,
    addCoinFrames
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
nk = 8
force_size = 3
nu = nk * force_size
nx = 9
space = manifolds.VectorSpace(nx)
space_multibody = manifolds.MultibodyPhaseSpace(rmodel)
x0 = space.neutral()

q0 = rmodel.referenceConfigurations["half_sitting"]
u0 = np.zeros(nu)
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
com0 = pin.centerOfMass(rmodel, rdata, q0)
x0[:3] = com0.copy()

gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
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

""" Initialize simulation """
device = BulletRobot(controlled_joints,
                        modelPath,
                        URDF_FILENAME,
                        1e-3,
                        rmodelComplete,
                        q0[:3])
device.initializeJoints(qComplete)
q_current, v_current = device.measureState()

""" Add coin frames in contact, 4 for each foot"""
rmodel, contact_ids, frame_contact_placement = addCoinFrames(rmodel, LF_id, RF_id)

""" Define gait and time parameters"""
T_ds = 20  # Double support time
T_ss = 80  # Singel support time

dt = 0.01
nsteps = 100
Nsimu = 10  #int(0.1 / dt)

""" Define contact sequence throughout horizon"""
total_steps = 6
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

T_mpc = len(contact_phases)  # Size of the problem

""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0
y_forward = 0
foot_yaw = 0
y_gap = 0.18
x_depth = 0.0

foottraj = footTrajectory(
    rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), T_ss, T_ds, nsteps, swing_apex, x_forward, y_forward, y_gap, foot_yaw, x_depth
)

""" Create dynamics and costs """

w_angular_acc = 10 * np.eye(3)
w_linear_mom = np.diag(np.array([0.001,0.001,100]))
w_centroidal_com = np.diag(np.array([0,0,1000]))
w_linear_acc = 0.1 * np.eye(3)
w_angular_mom = np.diag(np.array([10,10,100]))

w_control = np.ones(nu) * 0.005
w_control = np.diag(w_control)

def create_dynamics(contact_map):
    ode = dynamics.CentroidalFwdDynamics(space, mass, gravity, contact_map, force_size)
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model


def createStage(contact_state, LF_pose, RF_pose):
    coins_boolean = [contact_state[0] for _ in range(4)] + \
                    [contact_state[1] for _ in range(4)]
    
    coins_pose = []
    for i in range(4):
        coins_pose.append(frame_contact_placement[i].act(LF_pose).translation)
    for i in range(4):
        coins_pose.append(frame_contact_placement[i].act(RF_pose).translation)  

    contact_map = aligator.ContactMap(coins_boolean, coins_pose)

    rcost = aligator.CostStack(space, nu)

    linear_acc = aligator.CentroidalAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    angular_acc = aligator.AngularAccelerationResidual(
        nx, nu, mass, gravity, contact_map, force_size
    )
    u00 = np.zeros(nu)
    linear_mom = aligator.LinearMomentumResidual(nx, nu, np.zeros(3))
    angular_mom = aligator.AngularMomentumResidual(nx, nu, np.zeros(3))
    centroidal_com = aligator.CentroidalCoMResidual(nx, nu, com0)
    
    rcost.addCost(aligator.QuadraticControlCost(space, u00, w_control))
    rcost.addCost(
        aligator.QuadraticResidualCost(space, centroidal_com, w_centroidal_com)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, linear_mom, w_linear_mom)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, angular_mom, w_angular_mom)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, angular_acc, w_angular_acc)
    )
    rcost.addCost(
        aligator.QuadraticResidualCost(space, linear_acc, w_linear_acc)
    )
    stm = aligator.StageModel(rcost, create_dynamics(contact_map))
    for i in range(len(coins_boolean)):
        if coins_boolean[i]:
            cone_cstr = aligator.FrictionConeResidual(space.ndx, nu, i, mu, 0)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())

    return stm

term_cost = aligator.CostStack(space, nu)
#term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 100 * w_x))

""" Create the optimal problem and the full horizon """
stages = []
for i in range(nsteps):
    stages.append(createStage(contact_phases[0], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()))

stages_full = [createStage(contact_phases[i], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()) for i in range(T_mpc)]

problem = aligator.TrajOptProblem(x0, stages, term_cost)

""" Parametrize the solver"""

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

weights_IK = [100, 10000, 10, 500] # Posture, feet tracking, centroidal tracking, base angle
weights_ID = [10000, 100]

g_p = 200
g_h = 10
g_b = 10
K_gains = []
g_q = np.diag(np.array([
    0, 0, 0, 0, 0, 0,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.1, 0.1,
    10, 10, 10, 10,
    10, 10, 10, 10
]))
K_gains.append([g_q, 2 * np.sqrt(g_q)])
K_gains.append([np.eye(6) * g_p, np.eye(6) * 2 * np.sqrt(g_p)])
K_gains.append([np.eye(6) * g_h, np.eye(6) * 2 * np.sqrt(g_h)])
K_gains.append([np.eye(3) * g_b, np.eye(3) * 2 * np.sqrt(g_b)])

weights_IKID = [100, 10000, 10, 500, 100]
IKID_solver = IKIDSolver_f3(rmodel, weights_IKID, K_gains, nk, mu, sole_ids, contact_ids, base_id, torso_id, False)

x0_multibody = np.concatenate((q0, np.zeros(rmodel.nv)))
qdot_prev = np.zeros(rmodel.nv)
qdot = np.zeros(rmodel.nv)
qddot = np.zeros(rmodel.nv)

previous_contact_state = [True, True]
device.showTargetToTrack(rdata.oMf[LF_id], rdata.oMf[RF_id])

""" Launch the MPC loop"""

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

    LF_refs, RF_refs = foottraj.updateTrajectory(
        takeoff_RF, takeoff_LF, land_RF, land_LF, rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy()
    )
    device.moveMarkers(LF_refs[0].translation, RF_refs[0].translation)
    
    contact_state = problem.stages[0].dyn_model.differential_dynamics.contact_map.contact_states
    
    for n in range(nsteps):
        contact_state = problem.stages[n].dyn_model.differential_dynamics.contact_map.contact_states
        if contact_state[0]:
            for k in range(4):
                coin_pose = frame_contact_placement[k].act(LF_refs[n]).translation
                problem.stages[n].dyn_model.differential_dynamics.contact_map.contact_poses[k] = coin_pose
                problem.stages[n].cost.components[4].residual.contact_map.contact_poses[k] = coin_pose
                problem.stages[n].cost.components[5].residual.contact_map.contact_poses[k] = coin_pose 
        
        if contact_state[4]:
            for k in range(4):
                coin_pose = frame_contact_placement[k].act(RF_refs[n]).translation
                problem.stages[n].dyn_model.differential_dynamics.contact_map.contact_poses[k + 4] = coin_pose
                problem.stages[n].cost.components[4].residual.contact_map.contact_poses[k + 4] = coin_pose
                problem.stages[n].cost.components[5].residual.contact_map.contact_poses[k + 4] = coin_pose 
    
    contact_state = problem.stages[0].dyn_model.differential_dynamics.contact_map.contact_states
    qdot_prev = qdot.copy()

    """ if t == 700:
        play_trajectory(x0_multibody, x_measured)
        exit() """
    
    if problem.stages[0].dyn_model.differential_dynamics.contact_map.contact_states[0]:
        fleft = np.zeros(3)
        tleft = np.zeros(3)
        for f in range(4):
            fleft += us[0][f * 3:(f+1) * 3]
            tleft += np.cross(us[0][f * 3:(f+1) * 3], frame_contact_placement[f].translation)
        force_left.append(fleft)
        torque_left.append(tleft)
    else:
        force_left.append(np.zeros(3))
        torque_left.append(np.zeros(3))
    if problem.stages[0].dyn_model.differential_dynamics.contact_map.contact_states[4]:
        fright = np.zeros(3)
        tright = np.zeros(3)
        for f in range(4):
            fright += us[0][12 + f * 3:12 + (f+1) * 3]
            tright += np.cross(us[0][12 + f * 3:12 + (f+1) * 3], frame_contact_placement[f].translation)
        force_right.append(fright)
        torque_right.append(tright)
    else:
        force_right.append(np.zeros(3))
        torque_right.append(np.zeros(3))

    LF_measured.append(rdata.oMf[LF_id].copy())
    RF_measured.append(rdata.oMf[RF_id].copy())
    LF_references.append(LF_refs[0])
    RF_references.append(RF_refs[0])

    problem.replaceStageCircular(stages_full[t])
    com_final = com0.copy()
    com_final[:2] = (LF_refs[-1].translation[:2] + RF_refs[-1].translation[:2]) / 2
    com_cstr = aligator.CentroidalCoMResidual(space.ndx, nu, com_final)
    term_constraint_com = aligator.StageConstraint(
        com_cstr, constraints.EqualityConstraintSet()
    )
    problem.removeTerminalConstraint()
    problem.addTerminalConstraint(term_constraint_com)

    """ Compute various references for IK """
    q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff = \
      compute_ID_references(space_multibody, rmodel, rdata, LF_id, RF_id, base_id, torso_id, x0_multibody, x_measured, LF_refs, RF_refs, dt)
    dH = solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.xdot[3:9]
    
    for j in range(Nsimu):
        time.sleep(0.001)
        q_current, v_current = device.measureState()
        
        x_measured = shapeState(q_current, 
                                v_current, 
                                nq, 
                                nq + nv, 
                                controlled_ids)
        
        x_multibody.append(x_measured)

        #state_diff = space_multibody.difference(xs[0], x_measured)
        start = time.time()
        pin.forwardKinematics(rmodel, rdata, x_measured[:nq])
        pin.updateFramePlacements(rmodel, rdata)
        pin.computeJointJacobians(rmodel,rdata)
        pin.computeJointJacobiansTimeVariation(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        M = pin.crba(rmodel, rdata, x_measured[:nq])
        pin.nonLinearEffects(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        pin.dccrba(rmodel,rdata, x_measured[:nq], x_measured[nq:])

        new_x = np.zeros(9)
        new_x[:3] = pin.centerOfMass(rmodel, rdata, x_measured[:nq])
        pin.computeCentroidalMomentum(rmodel,rdata, x_measured[:nq], x_measured[nq:])
        new_x[3:6] = rdata.hg.linear
        new_x[6:] = rdata.hg.angular

        #qddot = (qdot - x_measured[nq:]) / dt

        """ forces = us[0] #+ K0 @ (new_x - xs[0])
        new_acc, new_forces, torque =ID_solver.solve(
            rdata,
            contact_state, 
            x_measured[nq:], 
            qddot,
            forces,
            M
        )  """
        new_acc, new_forces, torque = IKID_solver.solve(
            rdata,
            contact_state, 
            x_measured[nq:], 
            q_diff, dq_diff, 
            LF_diff, dLF_diff, 
            RF_diff, dRF_diff, 
            base_diff, dbase_diff, 
            torso_diff, dtorso_diff, 
            us[0], dH, M
        )
        end = time.time() 
        #print(end - start) 
        #qdot += new_acc * 0.001

        u_multibody.append(torque)
        device.execute(torque)
    
    previous_contact_state = copy.deepcopy(contact_state)
    com_measured.append(new_x[:3])
    L_measured.append(rdata.hg.angular.copy())
    xs = xs[1:] + [xs[-1]]
    us = us[1:] + [us[-1]]
    xs[0] = new_x

    problem.x0_init = new_x
    solver.setup(problem)
    start = time.time()
    solver.run(problem, xs, us)
    end = time.time()
    solve_time.append(end - start)
    #print(end - start)

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
                LF_measured, RF_measured, LF_references, RF_references, L_measured, "centroidal")
