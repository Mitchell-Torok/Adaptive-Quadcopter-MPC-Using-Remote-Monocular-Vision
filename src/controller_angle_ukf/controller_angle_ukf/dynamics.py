import os
import sys
import shutil
import casadi as cs
import numpy as np
from copy import copy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

class QuadDynamics:
    def __init__(self):
        # --- states ---
        self.p = cs.MX.sym('p', 3)   # position
        self.q = cs.MX.sym('a', 4)   # quaternion [qw, qx, qy, qz]
        self.v = cs.MX.sym('v', 3)   # linear velocity (world frame)
        self.r = cs.MX.sym('r', 3)   # body rates [p, q, r] (only r[2] used now)
        self.u = cs.MX.sym('u', 4)   # sticks: [roll, pitch, thrust, yaw]
        self.x = cs.vertcat(self.p, self.q, self.v, self.r, self.u)
        self.state_dim = 17

        # control is u_dot (stick rate)
        self.u_dot = cs.MX.sym('u_dot', 4)

        # --- parameters ---
        self.thrust_ratio   = cs.MX.sym('kT', 1)
        self.drag_coeff_z   = cs.MX.sym('drag_coeff_z', 1)
        self.tau_rate       = cs.MX.sym('tau_rate', 1)          # first-order yaw-rate time constant
        self.centre_rate_deg= cs.MX.sym('centre_rate_deg', 1)   # BF rates (for yaw)
        self.max_rate_deg   = cs.MX.sym('max_rate_deg', 1)
        self.rate_expo      = cs.MX.sym('rate_expo', 1)

        # Angle-mode (roll/pitch)
        self.angle_max_deg  = cs.MX.sym('angle_max_deg', 1)     # stick→angle map
        self.tau_angle      = cs.MX.sym('tau_angle', 1)         # first-order angle loop time constant

        # FC mounting angle offsets (disturbances to be estimated)
        self.fc_roll_offset_deg  = cs.MX.sym('fc_roll_offset_deg', 1)
        self.fc_pitch_offset_deg = cs.MX.sym('fc_pitch_offset_deg', 1)

        # Parameter vector
        self.p_param = cs.vertcat(
            self.thrust_ratio,
            self.drag_coeff_z,
            self.tau_rate,
            self.centre_rate_deg,
            self.max_rate_deg,
            self.rate_expo,
            self.angle_max_deg,
            self.tau_angle,
            self.fc_roll_offset_deg,
            self.fc_pitch_offset_deg
        )

        self.g = 9.81

    # ---------------- rotation helpers ----------------
    def q_to_rot_mat(self, q):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        if isinstance(q, np.ndarray):
            rot_mat = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
        else:
            rot_mat = cs.vertcat(
                cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
                cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
                cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))
        return rot_mat

    def v_dot_q(self, v, q):
        rot_mat = self.q_to_rot_mat(q)
        if isinstance(q, np.ndarray):
            return rot_mat.dot(v)
        return cs.mtimes(rot_mat, v)

    def skew_symmetric(self, v):
        if isinstance(v, np.ndarray):
            return np.array([[0, -v[0], -v[1], -v[2]],
                             [v[0], 0, v[2], -v[1]],
                             [v[1], -v[2], 0, v[0]],
                             [v[2], v[1], -v[0], 0]])

        return cs.vertcat(
            cs.horzcat(0, -v[0], -v[1], -v[2]),
            cs.horzcat(v[0], 0, v[2], -v[1]),
            cs.horzcat(v[1], -v[2], 0, v[0]),
            cs.horzcat(v[2], v[1], -v[0], 0))

    # ---------------- helpers ----------------
    def quat_to_euler(self, q):
        # q = [qw, qx, qy, qz]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        # roll (x-axis)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = cs.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = cs.fmax(-1.0, cs.fmin(1.0, sinp))
        pitch = cs.asin(sinp)

        # yaw (z-axis)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = cs.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def betaflight_rates(self, x):
        """
        Betaflight-style RC rate curve (deg/s) – used for yaw.
        """
        ax  = cs.sqrt(x * x + 1e-6)
        sgn = x / ax
        h_abs = ax * (cs.power(ax, 5) * self.rate_expo + ax * (1.0 - self.rate_expo))
        j_abs = self.centre_rate_deg * ax + (self.max_rate_deg - self.centre_rate_deg) * h_abs
        return sgn * j_abs

    # ---------------- dynamics wrapper ----------------
    def quad_dynamics(self):
        x_dot = cs.vertcat(
            self.p_dynamics(),
            self.q_dynamics(),   # modified: angle-mode logic lives here
            self.v_dynamics(),
            self.w_dynamics(),   # modified: yaw dynamics only
            self.u_dynamics()
        )
        return cs.Function('x_dot',
                           [self.x, self.u_dot, self.p_param],
                           [x_dot],
                           ['x', 'u', 'p'],
                           ['x_dot'])

    # ---------------- individual dynamics ----------------
    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        """
        Quaternion dynamics driven by *Euler angle first-order laws* for roll/pitch
        (angle mode) + yaw rate from state r[2].

        1. Stick → roll/pitch angle setpoints (with FC offsets).
        2. First-order angle dynamics towards those setpoints.
        3. Yaw Euler rate = yaw body rate r_z (state).
        4. Convert Euler rates → body rates via kinematic map.
        5. Quaternion update: q̇ = 0.5 * Ω(r_body) * q.
        """
        # Extract current Euler angles from quaternion
        roll, pitch, yaw = self.quat_to_euler(self.q)

        # Stick → angle setpoints (rad)
        angle_max_rad = (self.angle_max_deg * cs.pi) / 180.0
        phi_sp_base   = angle_max_rad * self.u[0]  # roll stick
        theta_sp_base = angle_max_rad * self.u[1]  # pitch stick

        # FC mounting offsets (deg → rad), sign to *compensate* mounting error
        fc_roll_offset_rad  = -(self.fc_roll_offset_deg  * cs.pi) / 180.0
        fc_pitch_offset_rad = -(self.fc_pitch_offset_deg * cs.pi) / 180.0

        phi_sp   = phi_sp_base   + fc_roll_offset_rad
        theta_sp = theta_sp_base + fc_pitch_offset_rad

        # First-order angle dynamics (angle mode)
        phi_dot   = (phi_sp   - roll)  / self.tau_angle
        theta_dot = (theta_sp - pitch) / self.tau_angle

        # Yaw: Euler yaw rate comes from yaw body-rate state r_z
        yaw_rate_body = self.r[2]              # r_z (rad/s)
        # Map body rates → Euler rates: euler_dot = T(φ,θ) * r_body
        # We want euler_dot[2] = ψ̇ consistent with r_z and current φ,θ.
        # Instead of inverting, we simply set ψ̇ to whatever r_z implies.
        # For small angles, ψ̇ ≈ r_z; we can just take:
        psi_dot = yaw_rate_body

        euler_dot = cs.vertcat(phi_dot, theta_dot, psi_dot)

        # Kinematic map: euler_dot = T(φ,θ) * r_body  →  r_body = solve(T, euler_dot)
        sphi, cphi = cs.sin(roll), cs.cos(roll)
        ttheta     = cs.tan(pitch)
        ctheta     = cs.cos(pitch)
        eps        = 1e-6

        T = cs.vertcat(
            cs.horzcat(1,            sphi * ttheta,           cphi * ttheta),
            cs.horzcat(0,            cphi,                   -sphi),
            cs.horzcat(0,            sphi / (ctheta + eps),   cphi / (ctheta + eps))
        )
        r_body = cs.solve(T, euler_dot)   # [p, q, r] (rad/s) algebraic, not state

        # Quaternion kinematics
        q_dot = 0.5 * cs.mtimes(self.skew_symmetric(r_body), self.q)
        return q_dot

    def v_dynamics(self):
        a_thrust = cs.vertcat(0.0, 0.0, self.thrust_ratio * self.u[2])
        drag_force = cs.vertcat(0.0, 0.0, -self.drag_coeff_z * self.v[2])
        g_vec = cs.vertcat(0.0, 0.0, self.g)
        return self.v_dot_q(a_thrust, self.q) - g_vec #+ drag_force

    def w_dynamics(self):
        """
        Yaw-only body-rate dynamics.

        - Roll/pitch rates are no longer dynamic states (effectively r[0], r[1] are unused).
        - Yaw stick maps to desired yaw rate via Betaflight curve (deg/s → rad/s).
        - First-order actuator on yaw rate:
            ṙ_z = (1/τ_rate) * (r_z_des - r_z)
        """
        # Desired yaw rate from Betaflight curve (deg/s → rad/s)
        yaw_rate_des = (self.betaflight_rates(-self.u[3]) * cs.pi) / 180.0

        # Current yaw rate state
        yaw_rate = self.r[2]

        # First-order dynamics on yaw rate
        yaw_dot = (1.0 / self.tau_rate) * (yaw_rate_des - yaw_rate)

        # Roll/pitch rate dynamics set to zero (no model)
        return cs.vertcat(0.0, 0.0, yaw_dot)

    def u_dynamics(self):
        # control is directly the stick derivative
        return self.u_dot
