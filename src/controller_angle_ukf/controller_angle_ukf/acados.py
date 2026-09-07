import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from .dynamics import QuadDynamics
import casadi as ca
from acados_template import AcadosSim, AcadosSimSolver


# ---------------------------
# Quaternion helper functions
# ---------------------------
def quat_conj(q):
    """Conjugate of quaternion q = [w, x, y, z]."""
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])


def quat_mul(q1, q2):
    """Hamilton product q1 ⊗ q2, both [w, x, y, z]"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )


# --------------------------------
# Public API: build OCP & simulator
# --------------------------------
def generate_ocp_controller(dt, N_horizon, skip_steps, dynamics=None):
    if dynamics is None:
        quad_dynamics = QuadDynamics()
    else:
        quad_dynamics = dynamics

    dynamics_expr = quad_dynamics.quad_dynamics()

    model = AcadosModel()
    model.name = 'quad_dynamics'
    model.x = quad_dynamics.x                 # 17 x 1
    model.u = quad_dynamics.u_dot            # control is u_dot (4 x 1)

    # ---- parameters: 10 dyn + 4 q_ref (total 14) ----
    # dyn order must match QuadDynamics.p_param:
    # [kT, drag_coeff_z, tau_rate, centre_rate_deg, max_rate_deg, rate_expo, 
    #  angle_max_deg, tau_angle, fc_roll_offset_deg, fc_pitch_offset_deg]
    p_dyn = quad_dynamics.p_param             # length 10
    p_qref = ca.MX.sym('p_qref', 4)           # [qw, qx, qy, qz]
    model.p = ca.vertcat(p_dyn, p_qref)       # length 14

    # feed first 10 params to dynamics
    model.f_expl_expr = dynamics_expr(model.x, model.u, model.p[:10])

    xdot = ca.MX.sym('xdot', model.x.size()[0])
    f_impl_expr = model.f_expl_expr - xdot
    model.xdot = xdot
    model.f_impl_expr = f_impl_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon * skip_steps * dt

    nu = 4
    nx = 17

    # ---------- COST: NONLINEAR_LS with sign-invariant quaternion error ----------
    q_idx = 3
    x = model.x
    u = model.u
    q = x[q_idx:q_idx+4]
    q_ref = model.p[10:14]   # <-- shifted: after 10 dyn params

    q_err = quat_mul(q_ref, quat_conj(q))
    e_att = 2 * q_err[1:4]  # vector part; sign-invariant attitude error

    # Layout: [pos(3), vel(3), omega(3), u_state(4), u(4), e_att(3)] -> total ny = 20
    y_expr   = ca.vertcat(x[0:3], x[7:10], x[10:13], x[13:17], u, e_att)
    y_expr_e = ca.vertcat(x[0:3], x[7:10], x[10:13], x[13:17], e_att)

    ocp.model.cost_y_expr = y_expr
    ocp.model.cost_y_expr_e = y_expr_e

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ny = 3 + 3 + 3 + 4 + 4 + 3
    ny_e = 3 + 3 + 3 + 4 + 3

    W = np.diag([
        2.0, 2.0, 5.0,         # pos
        0.1, 0.1, 1.0,         # vel
        1.0, 1.0, 1.0,         # omega
        10.0, 10.0, 2e-4, 10.0,# u_state (integrator smoothness)
        10.0, 10.0, 10.0, 10.0,    # u_dot penalty
        5.0, 5.0, 5.0          # attitude error
    ])
    W_e = np.diag([
        2.0, 2.0, 5.0,         # pos
        0.1, 0.1, 1.0,         # vel
        1.0, 1.0, 1.0,         # omega
        10.0, 10.0, 2e-4, 10.0,   # u_state
        5.0, 5.0, 5.0          # attitude error
    ])
    ocp.cost.W = W
    ocp.cost.W_e = W_e

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # Initial condition
    x0 = np.zeros(nx)
    ocp.constraints.x0 = x0

    # -------- Default parameters: 10 dyn + 4 q_ref --------
    # kT, dragZ, tau_rate, centre_deg, max_deg, expo, angle_max_deg, tau_angle, 
    # fc_roll_offset_deg, fc_pitch_offset_deg, q_ref(4)
    ocp.parameter_values = np.array([
        38.0,   # kT
        0.0,    # drag_coeff_z (fixed, disabled)
        0.07,   # tau_rate
        100.0,  # centre_rate_deg (yaw BF curve)
        100.0,  # max_rate_deg (yaw BF curve)
        0.5,    # rate_expo (yaw BF curve)
        55.0,   # angle_max_deg (Angle mode)
        0.15,   # tau_angle (Angle outer loop)
        0.0,    # fc_roll_offset_deg (FC mounting error - to be estimated)
        0.0,    # fc_pitch_offset_deg (FC mounting error - to be estimated)
        1.0, 0.0, 0.0, 0.0  # q_ref
    ])

    # ---------- Solver options ----------
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.qp_solver_iter_max = 600
    ocp.solver_options.qp_solver_tol_stat = 1e-3
    ocp.solver_options.qp_solver_tol_eq = 1e-3
    ocp.solver_options.qp_solver_tol_ineq = 1e-3
    ocp.solver_options.qp_solver_tol_comp = 1e-3
    ocp.solver_options.nlp_solver_tol_stat = 1e-3
    ocp.solver_options.nlp_solver_tol_eq = 1e-3
    ocp.solver_options.nlp_solver_tol_ineq = 1e-3
    ocp.solver_options.nlp_solver_tol_comp = 1e-3
    ocp.solver_options.levenberg_marquardt = 1e-2

    # ---------- State constraints (unchanged) ----------
    max_rate = 0.25
    max_vz = 0.5
    ocp.constraints.lbx = np.array([0.05, -max_rate, -max_rate, -max_rate])
    ocp.constraints.ubx = np.array([0.6,   max_rate,  max_rate,  max_rate])
    ocp.constraints.idxbx = np.array([15, 13, 14, 16])  # throttle state & p,q,r (check ordering if you changed mixer)

    # ---------- Input rate bounds (u = u_dot) ----------
    ocp.constraints.lbu = np.array([-0.2, -0.2, -0.5, -0.2])
    ocp.constraints.ubu = np.array([ 0.2,  0.2,  0.5,  0.2])
    ocp.constraints.idxbu = np.arange(nu)

    # Create OCP solver
    ocp_solver = AcadosOcpSolver(ocp)

    # --- Simulator config (1/30 s step) ---
    sim = AcadosSim()
    sim.model = ocp.model
    sim.solver_options.T = dt
    # same parameter vector as ocp.parameter_values but with potentially different taus for quick sim testing
    sim.parameter_values = np.array([
        38.0, 0.0, 0.12, 100.0, 100.0, 0.5, 55.0, 0.15, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ])
    sim_solver = AcadosSimSolver(sim)

    return ocp_solver, sim_solver


# ------------------------
# Warm start convenience
# ------------------------
def set_initial_guess(ocp_solver, N_horizon=20):
    u_init = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    for i in range(N_horizon):
        ocp_solver.set(i, "u", u_init)


def warm_start_from_previous_solution(ocp_solver, N_horizon=20):
    """Shift control warm start by one stage."""
    for i in range(N_horizon - 1):
        u_prev = ocp_solver.get(i + 1, "u")
        ocp_solver.set(i, "u", u_prev)
    u_last = ocp_solver.get(N_horizon - 1, "u")
    ocp_solver.set(N_horizon - 1, "u", u_last)


# -----------------------------------------------
# Reference handling: sign-continuous quaternion
# -----------------------------------------------
def _normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q if n == 0.0 else (q / n)


def _align_quat_to(prev_q: np.ndarray, q: np.ndarray) -> np.ndarray:
    qn = _normalize(q)
    return qn if float(np.dot(prev_q, qn)) >= 0.0 else -qn


def _make_quat_sequence_continuous(q_list):
    out = []
    if not q_list:
        return out
    out.append(_normalize(q_list[0]))
    for k in range(1, len(q_list)):
        out.append(_align_quat_to(out[-1], q_list[k]))
    return np.array(out)


def update_ocp_parameters(ocp_solver, est_params, N_horizon):
    """
    Update the dynamic parameters for all stages in the OCP solver.

    est_params must be length-10 in the order:
    [kT, dragZ, tau_rate, centre_rate_deg, max_rate_deg, rate_expo, 
     angle_max_deg, tau_angle, fc_roll_offset_deg, fc_pitch_offset_deg]
    """
    dyn_par = np.array(est_params, dtype=float)  # len 10
    default_qref = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    for j in range(N_horizon):
        full_params = np.concatenate([dyn_par, default_qref])  # len 14
        ocp_solver.set(j, "p", full_params)

    ocp_solver.set(N_horizon, "p", np.concatenate([dyn_par, default_qref]))


def set_trajectory_reference_aligned(ocp_solver, traj_states: np.ndarray, N_horizon: int, step_counter: int, skip_steps: int, est_params=None):
    """
    Sets yref and parameters per stage, with sign-continuous quaternion references.
    est_params is the length-10 dynamic parameter vector (same order as above).
    """
    horizon_indices = [step_counter + j * skip_steps for j in range(N_horizon)]
    terminal_index = step_counter + N_horizon * skip_steps
    all_indices = horizon_indices + [terminal_index]

    qs_raw = [traj_states[3:7, idx].copy() for idx in all_indices]
    qs_cont = _make_quat_sequence_continuous(qs_raw)

    dyn_par = np.array(est_params, dtype=float)

    for j, sc in enumerate(horizon_indices):
        # 1) yref
        yref = np.zeros((20,), dtype=float)
        yref[0:3]   = traj_states[0:3, sc]      # pos
        yref[3:6]   = traj_states[7:10, sc]     # vel
        yref[6:9]   = traj_states[10:13, sc]    # omega
        yref[13:17] = [0.0, 0.0, 0.0, 0.0]      # u_state refs (keep near zero)
        ocp_solver.set(j, "yref", yref)

        # 2) parameters: [dyn(10), q_ref(4)]
        qref = qs_cont[j]
        ocp_solver.set(j, "p", np.concatenate([dyn_par, qref]))

    yref_N = np.zeros((16,), dtype=float)
    yref_N[0:3] = traj_states[0:3, terminal_index]
    ocp_solver.set(N_horizon, "yref", yref_N)
    ocp_solver.set(N_horizon, "p", np.concatenate([dyn_par, qs_cont[-1]]))