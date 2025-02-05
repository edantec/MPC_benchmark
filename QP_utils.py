import numpy as np
import pinocchio as pin
import proxsuite

"""
First order inverse kinematics QP solver which outputs an optimal joint velocity 
that tracks desired end effector and centroidal trajectories.
"""

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

"""
Second order inverse kinematics QP solver which outputs an optimal joint acceleration 
that tracks desired end effector and centroidal trajectories.
"""

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

"""
Inverse dynamics QP solver which outputs optimal joint torque and contact forces 
so that the rigid contact dynamics of the multibody system is respected.
Take as input desired joint velocity and desired contact forces.
"""

class IDSolver_velocity:
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
    
    def computeMatrice(self, data, cs, v, forces, M):
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

        self.b[:self.model.nv] = -nle + JcT @ forces
        self.b[self.model.nv:] = -gamma
        
        self.l = np.zeros(5 * self.nk)
        self.C = np.zeros((5 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 5:(i+1)*5] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2]
                ]) 
                self.C[i * 5:(i + 1) * 5, self.model.nv + i * self.force_size:self.model.nv + (i + 1) * self.force_size] = self.Cmin


    def solve(self, data, cs, v, forces, M):
        self.computeMatrice(data, cs, v, forces, M)

        self.qp.update(
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

"""
Inverse dynamics QP solver which outputs optimal joint torque and contact forces 
so that the rigid contact dynamics of the multibody system is respected.
Take as input desired joint velocity, desired joint acceleration and desired contact forces.
"""

class IDSolver:
    def __init__(
        self, model, weights, nk, mu, L, W, contact_ids, force_size, verbose: bool
    ):
        kd = 2 * np.sqrt(0)
        baum_Kd = np.array([kd,kd,kd])
        self.baum_Kd = np.diag(baum_Kd)
        self.nk = nk
        self.contact_ids = contact_ids
        self.mu = mu
        self.L = L
        self.W = W
        self.force_size = force_size

        n = 2 * model.nv - 6 + force_size * nk 
        neq = model.nv + force_size * nk
        nin = 9 * nk

        self.A = np.zeros((model.nv + force_size * nk, 2 * model.nv - 6 + force_size * nk))
        self.b = np.zeros(model.nv + force_size * nk)
        self.l = np.zeros(9 * nk)
        self.C = np.zeros((9 * nk, 2 * model.nv - 6 + force_size * nk))

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((9, force_size))
        if force_size == 3:
            self.Cmin = np.array([
                [-1, 0, mu],
                [1, 0, mu],
                [-1, 0, mu],
                [1, 0, mu],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]
            ])
        else:
            self.Cmin = np.array([
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, W, -1, 0, 0],
                [0, 0, W, 1, 0, 0],
                [0, 0, L, 0,-1, 0],
                [0, 0, L, 0, 1, 0]
            ])

        u = np.ones(9 * nk) * 100000
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
                fJf = pin.getFrameJacobian(self.model, data, self.contact_ids[i], pin.LOCAL_WORLD_ALIGNED)[:self.force_size]
                Jdot = pin.getFrameJacobianTimeVariation(self.model, data, self.contact_ids[i], pin.LOCAL_WORLD_ALIGNED)[:self.force_size]
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
        
        self.l = np.zeros(9 * self.nk)
        self.C = np.zeros((9 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 9:(i+1)*9] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2],
                    forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    -forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L,
                    -forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L
                ]) 
                self.C[i * 9:(i + 1) * 9, self.model.nv + i * self.force_size:self.model.nv + (i + 1) * self.force_size] = self.Cmin

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

"""
Inverse dynamics QP solver which outputs optimal joint torque and contact forces 
so that the rigid contact dynamics of the multibody system is respected.
Take as input desired joint velocity and desired contact forces.
Enforce torque limits.
"""

class IDSolver_ulim:
    def __init__(
        self, model, weights, nk, mu, L, W, contact_ids, force_size, verbose: bool
    ):
        kd = 1
        baum_Kd = np.array([kd,kd,kd])
        self.baum_Kd = np.diag(baum_Kd)
        self.nk = nk
        self.contact_ids = contact_ids
        self.mu = mu
        self.L = L
        self.W = W
        self.force_size = force_size

        n = 2 * model.nv - 6 + force_size * nk 
        neq = model.nv + force_size * nk
        nin = 9 * nk

        self.A = np.zeros((model.nv + force_size * nk, 2 * model.nv - 6 + force_size * nk))
        self.b = np.zeros(model.nv + force_size * nk)
        self.l = np.zeros(9 * nk)
        self.C = np.zeros((9 * nk, 2 * model.nv - 6 + force_size * nk))

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((9, force_size))
        if force_size == 3:
            self.Cmin = np.array([
                [-1, 0, mu],
                [1, 0, mu],
                [-1, 0, mu],
                [1, 0, mu],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]
            ])
        else:
            self.Cmin = np.array([
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, W, -1, 0, 0],
                [0, 0, W, 1, 0, 0],
                [0, 0, L, 0,-1, 0],
                [0, 0, L, 0, 1, 0]
            ])

        u = np.ones(9 * nk) * 100000
        
        self.l_box = -np.ones(n) * 100000
        self.l_box[model.nv + force_size * nk:] = -model.effortLimit[6:]
        self.u_box = np.ones(n) * 100000
        self.u_box[model.nv + force_size * nk:] = model.effortLimit[6:]
        g = np.zeros(n)
        H = np.zeros((n,n))
        H[:model.nv, :model.nv] = np.eye(model.nv) * weights[0]
        H[model.nv:model.nv + force_size * nk,model.nv:model.nv + force_size * nk] = np.eye(force_size * nk) * weights[1]

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, False, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 10
        qp.settings.max_iter_in = 10
        qp.init(H, g, self.A, self.b, self.C, self.l, u)#, self.l_box, self.u_box)
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
                #gamma[i * self.force_size:i * self.force_size + 3] += self.baum_Kp @ data.oMf[self.contact_ids[i]].rotation.T @ (data.oMf[self.contact_ids[i]].translation - feet_refs[i])
                gamma[i * self.force_size:i * self.force_size + 3] += self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).linear \
                  + self.baum_Kd @ pin.getFrameVelocity(self.model, data, self.contact_ids[i]).angular

        JcT = Jc.T
        self.A[:self.model.nv,:self.model.nv] = M 
        self.A[:self.model.nv,self.model.nv:self.model.nv + self.nk * self.force_size] = -JcT
        self.A[:self.model.nv,self.model.nv + self.nk * self.force_size:] = -self.S
        self.A[self.model.nv:,:self.model.nv] = Jc 

        self.b[:self.model.nv] = -nle - M @ a + JcT @ forces
        self.b[self.model.nv:] = -gamma - Jc @ a
        
        self.l = np.zeros(9 * self.nk)
        self.C = np.zeros((9 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 9:(i+1)*9] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2],
                    forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    -forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L,
                    -forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L
                ]) 
                self.C[i * 9:(i + 1) * 9, self.model.nv + i * self.force_size:self.model.nv + (i + 1) * self.force_size] = self.Cmin

    def solve(self, data, cs, v, a, forces, M):
        self.computeMatrice(data, cs, v, a, forces, M)

        self.qp.update(
            A=self.A,
            b=self.b,
            C=self.C,
            l=self.l,
            #l_box=self.l_box,
            #u_box=self.u_box,
            update_preconditioner=False,
        )
        
        self.qp.solve()

        da = self.qp.results.x[:self.model.nv]
        anew = a + da
        dforces = self.qp.results.x[self.model.nv:self.model.nv + self.force_size * self.nk]
        new_forces = forces + dforces
        torque = self.qp.results.x[self.model.nv + self.force_size * self.nk:]

        return anew, new_forces, torque 

"""
Inverse kinematics + dynamics QP solver which outputs optimal joint torque and contact forces 
so that the rigid contact dynamics of the multibody system is respected.
Take as input desired joint velocity,  desired contact forces and desired variations of end effectors.
Use 6D wrench representation for contacts.
"""

class IKIDSolver_f6:
    def __init__(
        self, model, weights, K_gains, nk, mu, L, W, contact_ids, base_id, torso_id, force_size, verbose: bool
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
        self.W = W
        self.L = L
        self.force_size = force_size
        self.weights = weights

        n = 2 * model.nv - 6 + force_size * nk 
        neq = model.nv + force_size * nk
        nin = 9 * nk
        
        self.A = np.zeros((model.nv + force_size * nk, 2 * model.nv - 6 + force_size * nk))
        self.b = np.zeros(model.nv + force_size * nk) #
        self.l = np.zeros(9 * nk)
        self.C = np.zeros((9 * nk, 2 * model.nv - 6 + force_size * nk))
        self.l_box = -np.ones(n) * 100000
        self.l_box[model.nv + force_size * nk:] = -model.effortLimit[6:]
        self.u_box = np.ones(n) * 100000
        self.u_box[model.nv + force_size * nk:] = model.effortLimit[6:]

        self.S = np.zeros((model.nv,model.nv - 6))
        self.S[6:,:] = np.eye(model.nv - 6)
        
        self.Cmin = np.zeros((9, force_size))
        if force_size == 3:
            self.Cmin = np.array([
                [-1, 0, mu],
                [1, 0, mu],
                [-1, 0, mu],
                [1, 0, mu],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]
            ])
        else:
            self.Cmin = np.array([
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [-1, 0, mu, 0, 0, 0],
                [1, 0, mu, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, W, -1, 0, 0],
                [0, 0, W, 1, 0, 0],
                [0, 0, L, 0,-1, 0],
                [0, 0, L, 0, 1, 0]
            ])

        u = np.ones(9 * nk) * 100000
        self.g = np.zeros(n)
        self.H = np.zeros((n,n))
        self.H[model.nv:model.nv + force_size * nk,model.nv:model.nv + force_size * nk] = np.eye(force_size * nk) * weights[4]

        qp = proxsuite.proxqp.dense.QP(
            n, neq, nin, True, dense_backend=proxsuite.proxqp.dense.DenseBackend.PrimalDualLDLT)
        qp.settings.eps_abs = 1e-3
        qp.settings.eps_rel = 0.0
        qp.settings.primal_infeasibility_solving = True
        qp.settings.check_duality_gap = True
        qp.settings.verbose = verbose
        qp.settings.max_iter = 100
        qp.settings.max_iter_in = 100
        qp.init(self.H, self.g, self.A, self.b, self.C, self.l, u, self.l_box, self.u_box)
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
        
        self.l = np.zeros(9 * self.nk)
        self.C = np.zeros((9 * self.nk, 2 * self.model.nv - 6 + self.force_size * self.nk))
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 9:(i+1)*9] = np.array([
                    forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size] - forces[i * self.force_size + 2] * self.mu,
                    forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    -forces[i * self.force_size + 1] - forces[i * self.force_size + 2] * self.mu,
                    - forces[i * self.force_size + 2],
                    forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    -forces[i * self.force_size + 3] - forces[i * self.force_size + 2] * self.W,
                    forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L,
                    -forces[i * self.force_size + 4] - forces[i * self.force_size + 2] * self.L
                ]) 
                self.C[i * 9:(i + 1) * 9, self.model.nv + i * self.force_size:self.model.nv + (i + 1) * self.force_size] = self.Cmin

    def solve(self, data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M):
        self.computeMatrice(data, cs, v, q_diff, dq_diff, LF_diff, dLF_diff, RF_diff, dRF_diff, base_diff, dbase_diff, torso_diff, dtorso_diff, forces, dH, M)

        self.qp.update(
            H=self.H,
            g=self.g,
            A=self.A,
            b=self.b,
            C=self.C,
            l=self.l,
            l_box=self.l_box,
            u_box=self.u_box,
            update_preconditioner=False,
        )
        
        self.qp.solve()

        anew = self.qp.results.x[:self.model.nv]
        dforces = self.qp.results.x[self.model.nv:self.model.nv + self.force_size * self.nk]
        new_forces = forces + dforces
        torque = self.qp.results.x[self.model.nv + self.force_size * self.nk:]

        return anew, new_forces, torque 

"""
Inverse kinematics + dynamics QP solver which outputs optimal joint torque and contact forces 
so that the rigid contact dynamics of the multibody system is respected.
Take as input desired joint velocity,  desired contact forces and desired variations of end effectors.
Use 3D forces representation for contacts.
"""

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
        self.C = np.zeros((5 * self.nk, 2 * self.model.nv - 6 + 3 * self.nk))
        for i in range(self.nk):
            if cs[i]:
                self.l[i * 5:(i+1)*5] = np.array([
                    forces[i * 3] - forces[i * 3 + 2] * self.mu,
                    -forces[i * 3] - forces[i * 3 + 2] * self.mu,
                    forces[i * 3 + 1] - forces[i * 3 + 2] * self.mu,
                    -forces[i * 3 + 1] - forces[i * 3 + 2] * self.mu,
                    - forces[i * 3 + 2]
                ])  
                self.C[i * 5:(i + 1) * 5, self.model.nv + i * 3:self.model.nv + (i + 1) * 3] = self.Cmin


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