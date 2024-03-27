import numpy as np
import matplotlib.pyplot as plt
import os 
import pinocchio as pin

CURRENT_DIRECTORY = os.getcwd()
DEFAULT_SAVE_DIR = CURRENT_DIRECTORY + '/tmp'

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

data = load_data("tmp/centroidal_f6.npz") # kinodynamics_f3 fulldynamics centroidal
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

Tx = len(xs)
Tl = len(LF_force)
ttx = np.linspace(0,Tx * 0.001, Tx)
ttl = np.linspace(0,Tl * 0.01, Tl)

cop_total = []
footylimup, footylimdown = [], []
footxlimup, footxlimdown = [], []
FOOT_LENGTH = 0.1
FOOT_WIDTH = 0.05
LF_trans = []
RF_trans = []
LF_trans_ref = []
RF_trans_ref = []
for i in range(Tl):
    LF_se3 = pin.SE3(LF_pose[i])
    RF_se3 = pin.SE3(RF_pose[i])
    LF_trans.append(LF_se3.translation)
    RF_trans.append(RF_se3.translation)
    LF_trans_ref.append(pin.SE3(LF_pose_ref[i]).translation)
    RF_trans_ref.append(pin.SE3(RF_pose_ref[i]).translation)
    cop_total.append(computeCoP(LF_se3, RF_se3, LF_force[i], LF_torque[i], RF_force[i], RF_torque[i]))
    if LF_force[i][2] > 0.1 and RF_force[i][2] > 0.1:
        footylimup.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup.append(max(LF_se3.translation[0],RF_se3.translation[0]) + FOOT_LENGTH) 
        footxlimdown.append(min(LF_se3.translation[0],RF_se3.translation[0]) - FOOT_LENGTH)
    elif LF_force[i][2] > 0.1:
        footylimup.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown.append(LF_se3.translation[1] - FOOT_WIDTH)
        footxlimup.append(LF_se3.translation[0] + FOOT_LENGTH) 
        footxlimdown.append(LF_se3.translation[0] - FOOT_LENGTH)
    elif RF_force[i][2] > 0.1:
        footylimup.append(RF_se3.translation[1] + FOOT_WIDTH)
        footylimdown.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup.append(RF_se3.translation[0] + FOOT_LENGTH) 
        footxlimdown.append(RF_se3.translation[0] - FOOT_LENGTH)
    else:
        footylimup.append(LF_se3.translation[1] + FOOT_WIDTH)
        footylimdown.append(RF_se3.translation[1] - FOOT_WIDTH)
        footxlimup.append(max(LF_se3.translation[0],RF_se3.translation[0]) + FOOT_LENGTH) 
        footxlimdown.append(min(LF_se3.translation[0],RF_se3.translation[0]) - FOOT_LENGTH)

cop_total = np.array(cop_total)
LF_trans = np.array(LF_trans)
RF_trans = np.array(RF_trans)
LF_trans_ref = np.array(LF_trans_ref)
RF_trans_ref = np.array(RF_trans_ref)

# CoP 
plt.figure('Total CoP in limits')
plt.subplot(211)
plt.ylabel('x')
plt.plot(ttl, cop_total[:,0], label = 'CoP_x')
plt.plot(ttl, com[:,0], label = 'CoM_x')
plt.plot(ttl, footxlimup, label = 'limits')
plt.plot(ttl, footxlimdown, label = 'limits')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(212)
plt.ylabel('y')
plt.plot(ttl, cop_total[:,1], label = 'CoP_y')
plt.plot(ttl, com[:,1], label = 'CoM_y')
plt.plot(ttl, footylimup, label = 'limits')
plt.plot(ttl, footylimdown, label = 'limits')
plt.grid(True)
plt.legend(loc="upper left")

# CoM 
plt.figure('CoM')
plt.subplot(311)
plt.ylabel('x')
plt.plot(ttl, com[:,0], label = 'CoM_x')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel('y')
plt.plot(ttl, com[:,1], label = 'CoM_y')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(313)
plt.ylabel('z')
plt.plot(ttl, com[:,2], label = 'CoM_z')
plt.grid(True)
plt.legend(loc="upper left")

# L 
plt.figure('Angular momentum')
plt.subplot(311)
plt.ylabel('x')
plt.plot(ttl, L_measured[:,0], label = 'L_x')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(312)
plt.ylabel('y')
plt.plot(ttl, L_measured[:,1], label = 'L_y')
plt.grid(True)
plt.legend(loc="upper left")
plt.subplot(313)
plt.ylabel('z')
plt.plot(ttl, L_measured[:,2], label = 'L_z')
plt.grid(True)
plt.legend(loc="upper left")

# Foot force

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(3.5, 2.5),
                        layout="constrained")
axs[0,0].plot(ttl, LF_force[:,0])
axs[0,0].set_title('Fx left')
axs[0,0].grid(True)
axs[1,0].plot(ttl, LF_force[:,1])
axs[1,0].grid(True)
axs[1,0].set_title('Fy left')
axs[2,0].plot(ttl, LF_force[:,2])
axs[2,0].grid(True)
axs[2,0].set_title('Fz left')
axs[0,1].plot(ttl, RF_force[:,0])
axs[0,1].grid(True)
axs[0,1].set_title('Fx right')
axs[1,1].plot(ttl, RF_force[:,1])
axs[1,1].grid(True)
axs[1,1].set_title('Fy right')
axs[2,1].plot(ttl, RF_force[:,2])
axs[2,1].grid(True)
axs[2,1].set_title('Fz right') 

# Foot translation

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(3.5, 2.5),
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
axs[2,1].set_title('RF z')

BINS = 50
plt.figure("Computational load for 1 MPC iteration")
plt.hist(solve_time * 1000, bins=BINS)
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Occurences")
plt.tight_layout()
plt.grid(True)

plt.show()

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