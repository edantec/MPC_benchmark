import numpy as np
import matplotlib.pyplot as plt
import os 
import pinocchio as pin

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rcParams['figure.dpi'] = 150
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_data(npz_file):
    """
    Loads a npz archive of sim_data into a dict
    """
    d = np.load(npz_file, allow_pickle=True, encoding="latin1")
    return d["data"][()]

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

data_kino = load_data("tmp/kinodynamics.npz") # kinodynamics_f6 fulldynamics centroidal_f6
xs_kino = data_kino["xs"]
us_kino = data_kino["us"]
com_kino = data_kino["com"]
LF_force_kino = data_kino["LF_force"]
RF_force_kino = data_kino["RF_force"]
LF_torque_kino = data_kino["LF_torque"]
RF_torque_kino= data_kino["RF_torque"]
LF_pose_kino = data_kino["LF_pose"]
RF_pose_kino = data_kino["RF_pose"]
LF_pose_ref_kino = data_kino["LF_pose_ref"]
RF_pose_ref_kino = data_kino["RF_pose_ref"]
L_measured_kino = data_kino["L_measured"]
solve_time_kino = data_kino["time"]

data_full = load_data("tmp/fulldynamics.npz") # kinodynamics_f6 fulldynamics centroidal_f6
xs_full = data_full["xs"]
us_full = data_full["us"]
com_full = data_full["com"]
LF_force_full = data_full["LF_force"]
RF_force_full = data_full["RF_force"]
LF_torque_full = data_full["LF_torque"]
RF_torque_full = data_full["RF_torque"]
LF_pose_full = data_full["LF_pose"]
RF_pose_full = data_full["RF_pose"]
LF_pose_ref_full = data_full["LF_pose_ref"]
RF_pose_ref_full = data_full["RF_pose_ref"]
L_measured_full = data_full["L_measured"]
solve_time_full = data_full["time"]

data = load_data("tmp/centroidal_f6.npz") # kinodynamics_f6 fulldynamics centroidal_f6
xs = data["xs"]
us = data["us"]
com = data["com"]
LF_force = data["LF_force"]
RF_force = data["RF_force"]
LF_torque = data["LF_torque"]
RF_torque = data["RF_torque"]
LF_pose = data["LF_pose"]
RF_pose = data["RF_pose"]
LF_pose_ref = data["LF_pose_ref"]
RF_pose_ref = data["RF_pose_ref"]
L_measured = data["L_measured"]
solve_time = data["time"]

Tx_cent = len(xs)
Tl_cent = len(LF_pose)
ttx = np.linspace(0,Tx_cent * 0.001, Tx_cent)
ttl = np.linspace(0,Tl_cent * 0.01, Tl_cent)

T_start = 0
Tx_kino = len(xs_kino)
Tl_kino = len(LF_pose_kino) - T_start
ttx_kino = np.linspace(0,Tx_kino * 0.001, Tx_kino)
ttl_kino = np.linspace(0,Tl_kino * 0.01, Tl_kino)

Tx_full = len(xs_full)
Tl_full = len(LF_pose_full)
ttx_full = np.linspace(0,Tx_full * 0.001, Tx_full)
ttl_full = np.linspace(0,Tl_full * 0.01, Tl_full)

cop_total_cent = []
cop_total_kino = []
cop_total_full = []
footylimup_full, footylimdown_full = [], []
footxlimup_full, footxlimdown_full = [], []
footylimup_cent, footylimdown_cent = [], []
footxlimup_cent, footxlimdown_cent = [], []
footylimup_kino, footylimdown_kino = [], []
footxlimup_kino, footxlimdown_kino = [], []
FOOT_LENGTH = 0.1
FOOT_WIDTH = 0.05
LF_trans = []
RF_trans = []
LF_trans_full = []
RF_trans_full = []
LF_trans_kino = []
RF_trans_kino = []
LF_trans_ref = []
RF_trans_ref = []
LF_trans_ref_full = []
RF_trans_ref_full = []
LF_trans_ref_kino = []
RF_trans_ref_kino = []
for i in range(Tl_cent):
    LF_se3 = pin.SE3(LF_pose[i])
    RF_se3 = pin.SE3(RF_pose[i])
    LF_trans.append(LF_se3.translation)
    RF_trans.append(RF_se3.translation)
    LF_trans_ref.append(pin.SE3(LF_pose_ref[i]).translation)
    RF_trans_ref.append(pin.SE3(RF_pose_ref[i]).translation)
    cop_total_cent.append(computeCoP(LF_se3, RF_se3, LF_force[i], LF_torque[i], RF_force[i], RF_torque[i]))
    if LF_force[i][2] > 1 and RF_force[i][2] > 1:
        footylimup_cent.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown_cent.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup_cent.append(max(LF_se3.translation[0],RF_se3.translation[0]) + FOOT_LENGTH) 
        footxlimdown_cent.append(min(LF_se3.translation[0],RF_se3.translation[0]) - FOOT_LENGTH)
    elif LF_force[i][2] > 1:
        footylimup_cent.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown_cent.append(LF_se3.translation[1] - FOOT_WIDTH)
        footxlimup_cent.append(LF_se3.translation[0] + FOOT_LENGTH) 
        footxlimdown_cent.append(LF_se3.translation[0] - FOOT_LENGTH)
    elif RF_force[i][2] > 1:
        footylimup_cent.append(RF_se3.translation[1] + FOOT_WIDTH)
        footylimdown_cent.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup_cent.append(RF_se3.translation[0] + FOOT_LENGTH) 
        footxlimdown_cent.append(RF_se3.translation[0] - FOOT_LENGTH)
    else:
        footylimup_cent.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown_cent.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup_cent.append(max(LF_se3.translation[0],RF_se3.translation[0]) + FOOT_LENGTH) 
        footxlimdown_cent.append(min(LF_se3.translation[0],RF_se3.translation[0]) - FOOT_LENGTH)

for i in range(Tl_full):
    LF_se3_full = pin.SE3(LF_pose_full[i])
    RF_se3_full = pin.SE3(RF_pose_full[i])
    LF_trans_full.append(LF_se3_full.translation)
    RF_trans_full.append(RF_se3_full.translation)
    LF_trans_ref_full.append(pin.SE3(LF_pose_ref_full[i]).translation)
    RF_trans_ref_full.append(pin.SE3(RF_pose_ref_full[i]).translation)
    cop_total_full.append(computeCoP(LF_se3_full, RF_se3_full, LF_force_full[i], LF_torque_full[i], RF_force_full[i], RF_torque_full[i]))

    if LF_force_full[i][2] > 1 and RF_force_full[i][2] > 1:
        footylimup_full.append(LF_se3_full.translation[1] + FOOT_WIDTH)
        footylimdown_full.append(RF_se3_full.translation[1] - FOOT_WIDTH)
        footxlimup_full.append(max(LF_se3_full.translation[0],RF_se3_full.translation[0]) + FOOT_LENGTH) 
        footxlimdown_full.append(min(LF_se3_full.translation[0],RF_se3_full.translation[0]) - FOOT_LENGTH)
    elif LF_force_full[i][2] > 1:
        footylimup_full.append(LF_se3_full.translation[1] + FOOT_WIDTH)
        footylimdown_full.append(LF_se3_full.translation[1] - FOOT_WIDTH)
        footxlimup_full.append(LF_se3_full.translation[0] + FOOT_LENGTH) 
        footxlimdown_full.append(LF_se3_full.translation[0] - FOOT_LENGTH)
    elif RF_force_full[i][2] > 1:
        footylimup_full.append(RF_se3_full.translation[1] + FOOT_WIDTH)
        footylimdown_full.append(RF_se3_full.translation[1] - FOOT_WIDTH)
        footxlimup_full.append(RF_se3_full.translation[0] + FOOT_LENGTH) 
        footxlimdown_full.append(RF_se3_full.translation[0] - FOOT_LENGTH)
    else:
        footylimup_full.append(LF_se3_full.translation[1] + FOOT_WIDTH)
        footylimdown_full.append(RF_se3_full.translation[1] - FOOT_WIDTH)
        footxlimup_full.append(max(LF_se3_full.translation[0],RF_se3_full.translation[0]) + FOOT_LENGTH) 
        footxlimdown_full.append(min(LF_se3_full.translation[0],RF_se3_full.translation[0]) - FOOT_LENGTH)

for i in range(T_start, Tl_kino + T_start):
    LF_se3_kino = pin.SE3(LF_pose_kino[i])
    RF_se3_kino = pin.SE3(RF_pose_kino[i])
    LF_trans_kino.append(LF_se3_kino.translation)
    RF_trans_kino.append(RF_se3_kino.translation)
    LF_trans_ref_kino.append(pin.SE3(LF_pose_ref_kino[i]).translation)
    RF_trans_ref_kino.append(pin.SE3(RF_pose_ref_kino[i]).translation)
    cop_total_kino.append(computeCoP(pin.SE3(LF_pose_kino[i]), pin.SE3(RF_pose_kino[i]), LF_force_kino[i], LF_torque_kino[i], RF_force_kino[i], RF_torque_kino[i]))
    
    if LF_force_kino[i][2] > 1 and RF_force_kino[i][2] > 1:
        footylimup_kino.append(LF_se3_kino.translation[1] + FOOT_WIDTH)
        footylimdown_kino.append(RF_se3_kino.translation[1] - FOOT_WIDTH)
        footxlimup_kino.append(max(LF_se3_kino.translation[0],RF_se3_kino.translation[0]) + FOOT_LENGTH) 
        footxlimdown_kino.append(min(LF_se3_kino.translation[0],RF_se3_kino.translation[0]) - FOOT_LENGTH)
    elif LF_force_kino[i][2] > 1:
        footylimup_kino.append(LF_se3_kino.translation[1] + FOOT_WIDTH)
        footylimdown_kino.append(LF_se3_kino.translation[1] - FOOT_WIDTH)
        footxlimup_kino.append(LF_se3_kino.translation[0] + FOOT_LENGTH) 
        footxlimdown_kino.append(LF_se3_kino.translation[0] - FOOT_LENGTH)
    elif RF_force_kino[i][2] > 1:
        footylimup_kino.append(RF_se3_kino.translation[1] + FOOT_WIDTH)
        footylimdown_kino.append(RF_se3_kino.translation[1] - FOOT_WIDTH)
        footxlimup_kino.append(RF_se3_kino.translation[0] + FOOT_LENGTH) 
        footxlimdown_kino.append(RF_se3_kino.translation[0] - FOOT_LENGTH)
    else:
        footylimup_kino.append(LF_se3_kino.translation[1] + FOOT_WIDTH)
        footylimdown_kino.append(RF_se3_kino.translation[1] - FOOT_WIDTH)
        footxlimup_kino.append(max(LF_se3_kino.translation[0],RF_se3_kino.translation[0]) + FOOT_LENGTH) 
        footxlimdown_kino.append(min(LF_se3_kino.translation[0],RF_se3_kino.translation[0]) - FOOT_LENGTH)

cop_total_cent = np.array(cop_total_cent)
cop_total_kino = np.array(cop_total_kino)
cop_total_full = np.array(cop_total_full)
LF_trans = np.array(LF_trans)
RF_trans = np.array(RF_trans)
LF_trans_full = np.array(LF_trans_full)
RF_trans_full = np.array(RF_trans_full)
LF_trans_kino = np.array(LF_trans_kino)
RF_trans_kino = np.array(RF_trans_kino)
LF_trans_ref = np.array(LF_trans_ref)
RF_trans_ref = np.array(RF_trans_ref)
LF_trans_ref_full = np.array(LF_trans_ref_full)
RF_trans_ref_full = np.array(RF_trans_ref_full)
LF_trans_ref_kino = np.array(LF_trans_ref_kino)
RF_trans_ref_kino = np.array(RF_trans_ref_kino)

# CoP 
fig, axs = plt.subplots(2, 3)
axs[0, 0].set_title('Full dynamics')
axs[0, 0].set_ylabel('X-axis (m)')
axs[0, 0].plot(ttl_full, cop_total_full[:,0], label = 'CoP traj')
axs[0, 0].plot(ttl_full, com_full[:,0], 'g', label = 'CoM traj')
axs[0, 0].plot(ttl_full, footxlimup_full, 'r', label = 'Support limits')
axs[0, 0].plot(ttl_full, footxlimdown_full, 'r')
axs[0, 0].grid(True)
axs[0, 0].legend(loc="upper left")
axs[1, 0].set_ylabel('Y-axis (m)')
axs[1, 0].plot(ttl_full, cop_total_full[:,1], label = 'CoP in y')
axs[1, 0].plot(ttl_full, com_full[:,1], 'g', label = 'CoM in y')
axs[1, 0].plot(ttl_full, footylimup_full, 'r', label = 'Support limits')
axs[1, 0].plot(ttl_full, footylimdown_full, 'r')
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].grid(True)
#axs[1, 0].legend(loc="upper left")
axs[0, 1].set_title('Centroidal')
axs[0, 1].plot(ttl, cop_total_cent[:,0], label = 'CoP in x')
axs[0, 1].plot(ttl, com[:,0], 'g', label = 'CoM in x')
axs[0, 1].plot(ttl, footxlimup_cent, 'r', label = 'Support limits')
axs[0, 1].plot(ttl, footxlimdown_cent, 'r')
axs[0, 1].grid(True)
#axs[0, 1].legend(loc="upper left")
axs[1, 1].plot(ttl, cop_total_cent[:,1], label = 'CoP in y')
axs[1, 1].plot(ttl, com[:,1], 'g', label = 'CoM in y')
axs[1, 1].plot(ttl, footylimup_cent, 'r', label = 'Support limits')
axs[1, 1].plot(ttl, footylimdown_cent, 'r')
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].grid(True)
#axs[1, 1].legend(loc="upper left")
axs[0, 2].set_title('Kinodynamics')
axs[0, 2].plot(ttl_kino, cop_total_kino[:,0], label = 'CoP in x')
axs[0, 2].plot(ttl_kino, com_kino[T_start:,0], 'g', label = 'CoM in x')
axs[0, 2].plot(ttl_kino, footxlimup_kino, 'r', label = 'Support limits')
axs[0, 2].plot(ttl_kino, footxlimdown_kino, 'r')
axs[0, 2].grid(True)
#axs[0, 2].legend(loc="upper left")
axs[1, 2].plot(ttl_kino, cop_total_kino[:,1], label = 'CoP in y')
axs[1, 2].plot(ttl_kino, com_kino[T_start:,1], 'g', label = 'CoM in y')
axs[1, 2].plot(ttl_kino, footylimup_kino, 'r', label = 'Support limits')
axs[1, 2].plot(ttl_kino, footylimdown_kino, 'r')
axs[1, 2].set_xlabel("Time (s)")
axs[1, 2].grid(True)
#axs[1, 2].legend(loc="upper left")

# CoM 
plt.figure('CoM')
plt.subplot(311)
plt.ylabel('x')
plt.plot(ttl, com[:,0], label = 'CoM_x cent')
plt.plot(ttl_kino, com_kino[T_start:,0], label = 'CoM_x kino')
plt.plot(ttl_full, com_full[:,0], label = 'CoM_x full')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel('y')
plt.plot(ttl, com[:,1], label = 'CoM_y cent')
plt.plot(ttl_kino, com_kino[T_start:,1], label = 'CoM_y kino')
plt.plot(ttl_full, com_full[:,1], label = 'CoM_y full')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(313)
plt.ylabel('z')
plt.plot(ttl, com[:,2], label = 'CoM_z cent')
plt.plot(ttl_kino, com_kino[T_start:,2], label = 'CoM_z kino')
plt.plot(ttl_full, com_full[:,2], label = 'CoM_z full')
plt.grid(True)
plt.legend(loc="upper left")

# L 
""" plt.figure('Angular momentum')
plt.subplot(311)
plt.title('Angular momentum along x')
plt.ylabel('x')
plt.plot(ttl, L_measured[:,0], label = 'L_x cent')
plt.plot(ttl_kino, L_measured_kino[T_start:,0], label = 'L_x kino')
plt.plot(ttl_full, L_measured_full[:,0], label = 'L_x full')
plt.ylabel("$kg.m^2.s^{-1}$")
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel('y')
plt.plot(ttl, L_measured[:,1], label = 'L_y cent')
plt.plot(ttl_kino, L_measured_kino[T_start:,1], label = 'L_y kino')
plt.plot(ttl_full, L_measured_full[:,1], label = 'L_y full')
plt.ylabel("$kg.m^2.s^{-1}$")
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(313)
plt.ylabel('z')
plt.plot(ttl, L_measured[:,2], label = 'L_z')
plt.plot(ttl_kino, L_measured_kino[T_start:,2], label = 'L_z kino')
plt.plot(ttl_full, L_measured_full[:,2], label = 'L_z full')
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("$kg.m^2.s^{-1}$")
plt.legend(loc="upper left") """

plt.figure('Angular momentum')
plt.plot(ttl, L_measured[:,2], label = 'Centroidal')
plt.plot(ttl_kino, L_measured_kino[T_start:,2], label = 'Kinodynamics')
plt.plot(ttl_full, L_measured_full[:,2], label = 'Full dynamics')
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Angular momentum along z $(kg.m^2.s^{-1})$")
plt.legend(loc="upper left")
# Foot force

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0,0].plot(ttl, LF_force[:,0])
axs[0,0].plot(ttl_kino, LF_force_kino[T_start:,0])
axs[0,0].plot(ttl_full, LF_force_full[:,0])
axs[0,0].set_title('Left foot')
axs[0,0].set_ylabel('Force along x (N)')
axs[0,0].grid(True)
axs[1,0].plot(ttl, LF_force[:,1])
axs[1,0].plot(ttl_kino, LF_force_kino[T_start:,1])
axs[1,0].plot(ttl_full, LF_force_full[:,1])
axs[1,0].set_ylabel('Force along y (N)')
axs[1,0].grid(True)
#axs[1,0].set_title('Left foot')
axs[2,0].plot(ttl, LF_force[:,2])
axs[2,0].plot(ttl_kino, LF_force_kino[T_start:,2])
axs[2,0].plot(ttl_full, LF_force_full[:,2])
axs[2,0].set_ylabel('Force along z (N)')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].grid(True)
#axs[2,0].set_title('Fz left')
axs[0,1].plot(ttl, RF_force[:,0], label = 'Centroidal')
axs[0,1].plot(ttl_kino, RF_force_kino[T_start:,0], label = 'Kinodynamics')
axs[0,1].plot(ttl_full, RF_force_full[:,0], label = 'Full dynamics')
axs[0,1].legend()
axs[0,1].grid(True)
axs[0,1].set_title('Right foot')
axs[1,1].plot(ttl, RF_force[:,1])
axs[1,1].plot(ttl_kino, RF_force_kino[T_start:,1])
axs[1,1].plot(ttl_full, RF_force_full[:,1])
axs[1,1].grid(True)
#axs[1,1].set_title('Fy right')
axs[2,1].plot(ttl, RF_force[:,2])
axs[2,1].plot(ttl_kino, RF_force_kino[T_start:,2])
axs[2,1].plot(ttl_full, RF_force_full[:,2])
axs[2,1].grid(True)
axs[2,1].set_xlabel('Time (s)')
#axs[2,1].set_title('Fz right') 

# Foot translation

""" fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0,0].plot(ttl, LF_trans[:,0])
axs[0,0].plot(ttl, LF_trans_ref[:,0], 'r')
axs[0,0].set_title('LF x')
axs[0,0].grid(True)
axs[1,0].plot(ttl, LF_trans[:,1])
axs[1,0].plot(ttl, LF_trans_ref[:,1], 'r')
axs[1,0].grid(True)
axs[1,0].set_title('LF y')
axs[2,0].plot(ttl, LF_trans[:,2])
axs[2,0].plot(ttl, LF_trans_ref[:,2], 'r')
axs[2,0].grid(True)
axs[2,0].set_title('LF z')
axs[0,1].plot(ttl, RF_trans[:,0])
axs[0,1].plot(ttl, RF_trans_ref[:,0], 'r')
axs[0,1].grid(True)
axs[0,1].set_title('RF x')
axs[1,1].plot(ttl, RF_trans[:,1])
axs[1,1].plot(ttl, RF_trans_ref[:,1], 'r')
axs[1,1].grid(True)
axs[1,1].set_title('RF y')
axs[2,1].plot(ttl, RF_trans[:,2])
axs[2,1].plot(ttl, RF_trans_ref[:,2], 'r')
axs[2,1].grid(True)
axs[2,1].set_title('RF z') """

kk=2
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0].plot(ttl_full, RF_trans_full[:,kk])
axs[0].plot(ttl_full, RF_trans_ref_full[:,kk], 'r')
axs[0].set_title('LF z full')
axs[0].set_xlim(0,5.45)
axs[0].grid(True)
axs[1].plot(ttl, RF_trans[:,kk])
axs[1].plot(ttl, RF_trans_ref[:,kk], 'r')
axs[1].set_xlim(0,5.45)
axs[1].grid(True)
axs[1].set_title('LF z cent')
axs[2].plot(ttl_kino, RF_trans_kino[:,kk])
axs[2].plot(ttl_kino, RF_trans_ref_kino[:,kk], 'r')
axs[2].set_xlim(0,5.45)
axs[2].grid(True)
axs[2].set_title('LF z kino')

# Joint power
nq = 29
nv = 28
nu = 22
power = []
power_kino = []
power_full = []
for i in range(Tx_cent):
    pp = np.abs(us[i] * xs[i][nq + 6:])
    power.append(np.sum(pp))

for i in range(Tx_kino):
    pp2 = np.abs(us_kino[i] * xs_kino[i][nq + 6:])
    power_kino.append(np.sum(pp2))

for i in range(Tx_full):
    pp3 = np.abs(us_full[i] * xs_full[i][nq + 6:])
    power_full.append(np.sum(pp3))

plt.figure()
#plt.title('Total joint power')
plt.grid(True)
plt.ylabel('Dissipated power $(kg.m^2.s^{-3})$')
plt.xlabel('Time (s)')
plt.plot(ttx, power, label = "Centroidal")
plt.plot(ttx_kino, power_kino,label = "Kinodynamics")
plt.plot(ttx_full, power_full,label = "Full dynamics")
plt.legend(loc = "upper left")
print("Mean power for centroidal is " + str(np.mean(power)))
print("Mean power for kinodynamics is " + str(np.mean(power_kino)))
print("Mean power for fulldynamics is " + str(np.mean(power_full)))
plt.show()
""" BINS = 50
plt.figure("Computational load for 1 MPC iteration")
plt.hist(solve_time * 1000, bins=BINS)
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.grid(True)

plt.show() """

""" 
BINS = 50
plt.figure(1)
plt.title("Computational load for 1 MPC iteration")
plt.hist(time2, bins=BINS, label='ProxDDP 2 threads')
plt.hist(time4, bins=BINS, label='ProxDDP 4 threads')
plt.hist(time8, bins=BINS, label='ProxDDP 8 threads')
plt.hist(time10, bins=BINS, label='ProxDDP 10 threads')
plt.hist(time_serial, bins=BINS, label='ProxDDP serial')
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.grid(True)
plt.savefig("plots/allhist_gr.png") """