print("Start script")
import numpy as np
import pinocchio as pin
import aligator
print("Import aligator")
from bullet_robot import BulletRobot
import time
import copy
from talos_utils import (
    loadTalos,
    URDF_FILENAME,
    modelPath,
    IDSolver,
    shapeState,
    footTrajectory,
    update_timings,
    save_trajectory,
)

from aligator import (manifolds, 
                    dynamics, 
                    constraints,)

print("Load robot model")
rmodelComplete, rmodel, qComplete, q0 = loadTalos()
rdata = rmodel.createData()
space = manifolds.MultibodyPhaseSpace(rmodel)

nq = rmodel.nq
nv = rmodel.nv
nk = 2
force_size = 6
nu = nv - 6 + nk * force_size
gravity = np.array([0, 0, -9.81])
mu = 0.8  # Friction coefficient
mass = pin.computeTotalMass(rmodel)
f_ref = np.array([0, 0, -mass * gravity[2] / nk, 0, 0, 0])
Lfoot = 0.1
Wfoot = 0.075

x0 = np.concatenate((q0, np.zeros(nv)))
u0 = np.zeros(nu)
for i in range(nk):
    u0[force_size * i + 2] = -gravity[2] * mass / nk

u00 = np.zeros(nu)
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
com0 = pin.centerOfMass(rmodel,rdata,q0)

LF_id = rmodel.getFrameId("left_sole_link")
RF_id = rmodel.getFrameId("right_sole_link")

controlled_joints = rmodel.names[1:].tolist()
controlled_ids = [rmodelComplete.getJointId(name_joint) for name_joint in controlled_joints[1:]]

""" Initialize simulation"""
print("Initialize simu")
device = BulletRobot(controlled_joints,
                     modelPath,
                     URDF_FILENAME,
                     1e-3,
                     rmodelComplete)
device.initializeJoints(qComplete)
q_current, v_current = device.measureState()

sole_ids = [LF_id, RF_id]

""" Create costs and dynamics"""
w_x = np.array([
    0, 0, 0, 100, 100, 100, # Base pos/ori
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Left leg
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # Right leg
    100, 100, # Torso
    1, 10, 10, 1, # Left arm
    1, 10, 10, 1, # Right arm
    0.1, 0.1, 0.1, 100, 100, 100, # Base pos/ori vel
    1, 1, 1, 1, 10, 10, # Left leg vel
    1, 1, 1, 1, 10, 10, # Right leg vel
    100, 100, # Torso vel
    1, 10, 10, 1, # Left arm vel
    1, 10, 10, 1, # Right arm vel
]) 
w_x = np.diag(w_x)
w_linforce = np.ones(3) * 0.00001
w_angforce = np.ones(3) * 0.001
w_u = np.concatenate((
    w_linforce, 
    w_angforce,
    w_linforce, 
    w_angforce,
    np.ones(rmodel.nv - 6) * 1e-3
))
w_u = np.diag(w_u) 
w_LFRF = 50000
w_cent_lin = np.ones(3) * 1e-4
w_cent_ang = np.ones(3) * 1e-4
w_cent = np.diag(np.concatenate((w_cent_lin,w_cent_ang)))
w_com = np.diag(np.array([1000]))

def create_dynamics(stage_space, cont_states):
    ode = dynamics.KinodynamicsFwdDynamics(
        stage_space, rmodel, gravity, cont_states, sole_ids, force_size
    )
    dyn_model = dynamics.IntegratorEuler(ode, dt)
    return dyn_model

v_ref = pin.Motion()
v_ref.np[:] = 0.0

def createStage(contact_state, contact_state_previous, LF_pose, RF_pose, uforce):
    stage_rmodel = rmodel.copy()
    stage_space = manifolds.MultibodyPhaseSpace(stage_rmodel)

    cent_mom = aligator.CentroidalMomentumDerivativeResidual(
        stage_space.ndx, stage_rmodel, gravity, contact_state, sole_ids, force_size
    )

    frame_vel_LF = aligator.FrameVelocityResidual(stage_space.ndx, nu, stage_rmodel, v_ref, LF_id, pin.LOCAL)
    frame_vel_RF = aligator.FrameVelocityResidual(stage_space.ndx, nu, stage_rmodel, v_ref, RF_id, pin.LOCAL)

    frame_fn_LF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, stage_rmodel, LF_pose, LF_id)
    frame_fn_RF = aligator.FramePlacementResidual(
        stage_space.ndx, nu, stage_rmodel, RF_pose, RF_id)
    frame_cs_RF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, stage_rmodel, RF_pose.translation, RF_id)[2]
    frame_cs_LF = aligator.FrameTranslationResidual(
        stage_space.ndx, nu, stage_rmodel, LF_pose.translation, LF_id)[2]
    frame_com = aligator.CenterOfMassTranslationResidual(space.ndx, nu, rmodel, com0)[2]
    
    rcost = aligator.CostStack(stage_space, nu)

    rcost.addCost(aligator.QuadraticStateCost(stage_space, nu, x0, w_x))
    rcost.addCost(aligator.QuadraticControlCost(stage_space, uforce, w_u))
    w_LF = np.zeros((6,6))
    w_RF = np.zeros((6,6))
    if contact_state[0]: #and not(contact_state[1]):
        w_RF = w_LFRF * np.eye(6)
    if contact_state[1]: #and not(contact_state[0]):
        w_LF = w_LFRF * np.eye(6)

    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_LF, w_LF))
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_fn_RF, w_RF))
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, cent_mom, w_cent))
    rcost.addCost(aligator.QuadraticResidualCost(stage_space, frame_com, w_com))

    stm = aligator.StageModel(rcost, create_dynamics(stage_space, contact_state))
    
    for i in range(len(contact_state)):
        if contact_state[i]:
            cone_cstr = aligator.WrenchConeResidual(space.ndx, nu, i, mu, 3/5*Lfoot, 3/5*Wfoot) #np.eye(3)
            stm.addConstraint(cone_cstr, constraints.NegativeOrthant())
    
    if contact_state[0]:
        stm.addConstraint(frame_vel_LF, constraints.EqualityConstraintSet())
    if contact_state[1]:
        stm.addConstraint(frame_vel_RF, constraints.EqualityConstraintSet()) 
    """
    if contact_state[1] and not(contact_state_previous[1]):
        stm.addConstraint(frame_cs_RF, constraints.EqualityConstraintSet())
    if contact_state[0] and not(contact_state_previous[0]):
        stm.addConstraint(frame_cs_LF, constraints.EqualityConstraintSet()) """
    return stm

term_cost = aligator.CostStack(space, nu)
#term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 100 * w_x))

""" Define gait and time parameters"""
T_ds = 20
T_ss = 80
dt = 0.01
nsteps = 100
Nsimu = int(dt / 0.001)

""" Define contact sequence throughout horizon"""
total_steps = 6
contact_phases = [[True,True]] * T_ds
for s in range(total_steps):
    contact_phases += [[True,False]] * T_ss + \
                      [[True,True]] * T_ds + \
                      [[False,True]] * T_ss + \
                      [[True,True]] * T_ds

contact_phases += [[True,True]] * nsteps

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
    un[2] = f_half * (j + 1) / float(T_ds)
    un[8] = f_full * (T_ds - j) / float(T_ds) + f_half * j / float(T_ds)
    urefs.append(un)

for j in range(nsteps):
    un = np.zeros(nu)
    un[2] = f_half 
    un[8] = f_half 
    urefs.append(un)

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
y_forward = 0.
foot_yaw = 0
y_gap = 0.18
x_depth = 0

foottraj = footTrajectory(
    rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), T_ss, T_ds, nsteps, swing_apex, x_forward, y_forward, foot_yaw, y_gap, x_depth
)

""" Create the optimal problem and the full horizon """
print("Create stages")
stages_full = [createStage(contact_phases[0], contact_phases[0], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), urefs[0])]
for i in range(1,Tmpc):
    stages_full.append(createStage(contact_phases[i],contact_phases[i-1], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), urefs[i]))

stages = [createStage(contact_phases[0], contact_phases[0], rdata.oMf[LF_id].copy(), rdata.oMf[RF_id].copy(), u0)] * nsteps
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
solver.sa_strategy = aligator.SA_FILTER
solver.filter.beta = 1e-5
#solver.setNumThreads(8)
solver.max_iters = max_iters

solver.setup(problem)

u_ref = np.concatenate((f_ref, f_ref, np.zeros(rmodel.nv - 6)))
us_init = [u_ref for _ in range(nsteps)]
xs_init = [x0] * (nsteps + 1)
print("solver run")
solver.run(
    problem,
    xs_init,
    us_init,
)
print("solved")
workspace = solver.workspace
results = solver.results
print(results)


xs = results.xs.tolist().copy()
us = results.us.tolist().copy()
K0 = results.controlFeedbacks()[0]

solver.max_iters = 1

q_current, v_current = device.measureState()
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

device.showTargetToTrack(rdata.oMf[LF_id], rdata.oMf[RF_id])
solve_time = []
time_computation = 0
weights_ID = [100, 100000] # Acceleration, forces
ID_solver = IDSolver(rmodel, weights_ID, nk, mu, Lfoot, Wfoot, sole_ids, force_size, False)

lowlevel_time = 0
time_computation = 0.01

for t in range(Tmpc):
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
    
    for j in range(nsteps):
        problem.stages[j].cost.components[2].residual.setReference(LF_refs[j])
        problem.stages[j].cost.components[3].residual.setReference(RF_refs[j]) 
    
    if problem.stages[0].dyn_model.differential_dynamics.contact_states[0]:
        force_left.append(us[0][:3])
        torque_left.append(us[0][3:6])
    else:
        force_left.append(np.zeros(3))
        torque_left.append(np.zeros(3))
    if problem.stages[0].dyn_model.differential_dynamics.contact_states[1]:
        force_right.append(us[0][6:9])
        torque_right.append(us[0][9:12])
    else:
        force_right.append(np.zeros(3))
        torque_right.append(np.zeros(3))

    LF_measured.append(rdata.oMf[LF_id].copy())
    RF_measured.append(rdata.oMf[RF_id].copy())
    LF_references.append(LF_refs[0].copy())
    RF_references.append(RF_refs[0].copy())
    com_measured.append(pin.centerOfMass(rmodel, rdata, x_measured[:nq]))
    L_measured.append(rdata.hg.angular.copy())

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

    """ for n in range(nsteps):
        contact_state = problem.stages[n].dyn_model.differential_dynamics.contact_states
        if contact_state[0]:
            problem.stages[n].constraints[1].func.updateWrenchCone(LF_refs[n].rotation)
        
        if contact_state[1]:
            if contact_state[0]:
                problem.stages[n].constraints[2].func.updateWrenchCone(RF_refs[n].rotation)
            else:
                problem.stages[n].constraints[1].func.updateWrenchCone(RF_refs[n].rotation)
    contact_state = problem.stages[0].dyn_model.differential_dynamics.contact_states """
    """ if t == 150:
        print("Force applied")
        device.apply_force([6000, 0,0], [0, 0, 0]) """
    """ if t == 350:
        for s in range(nsteps):
            device.resetState(xs[s][:rmodel.nq])
            time.sleep(0.5)
            print("s = " + str(s))
            device.moveMarkers(LF_refs[s].translation, RF_refs[s].translation)
        exit() """

    a0 = solver.workspace.problem_data.stage_data[0].constraint_data[0].continuous_data.xdot[rmodel.nv:]
    #a0 = solver.workspace.problem_data.stage_data[0].dynamics_data.continuous_data.xdot[rmodel.nv:]
    

    #while lowlevel_time < time_computation:
    for j in range(Nsimu):
        #time.sleep(0.001)
        lowlevel_time += 0.001
        q_current, v_current = device.measureState()
        
        x_measured = shapeState(q_current, 
                                v_current, 
                                nq, 
                                nq + nv, 
                                controlled_ids)  
        
        state_diff = space.difference(xs[0], x_measured)
        contact_state = problem.stages[0].dyn_model.differential_dynamics.contact_states

        pin.forwardKinematics(rmodel, rdata, x_measured[:nq])
        pin.updateFramePlacements(rmodel, rdata)
        pin.computeJointJacobians(rmodel,rdata)
        M = pin.crba(rmodel, rdata, x_measured[:nq])
        pin.nonLinearEffects(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        pin.computeJointJacobiansTimeVariation(rmodel, rdata, x_measured[:nq], x_measured[nq:])
        pin.dccrba(rmodel,rdata, x_measured[:nq], x_measured[nq:])

        a0[6:] = us[0][nk * force_size:] #- K0[nk * force_size:] @ state_diff
        forces = us[0][:nk * force_size] #- K0[:nk * force_size] @ state_diff

        new_acc, new_forces, torque_qp =ID_solver.solve(
            rdata,
            contact_state, 
            x_measured[nq:], 
            a0,
            forces,
            M
        )

        device.execute(torque_qp)

        u_multibody.append(torque_qp)
        x_multibody.append(x_measured)
    
    lowlevel_time = 0

    xs = xs[1:] + [xs[-1]]
    us = us[1:] + [us[-1]]
    xs[0] = x_measured

    problem.x0_init = x_measured
    solver.setup(problem)
    start = time.time()
    solver.run(problem, xs, us)
    end = time.time()
    time_computation = end - start
    solve_time.append(end - start)
    print("MPC solve " + str(end - start))

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
                LF_measured, RF_measured, LF_references, RF_references, L_measured, "kinodynamics_f6")