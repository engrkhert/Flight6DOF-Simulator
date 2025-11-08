# flight6dof.py
# 6-DOF aircraft flight dynamics simulator with Streamlit UI
# Save as flight6dof.py and run: streamlit run flight6dof.py

import streamlit as st
import numpy as np
from math import sin, cos, atan2, sqrt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

st.set_page_config(layout="wide", page_title="6-DOF Flight Simulator")

# -------------------------
# Utility / quaternion math
# -------------------------
def quat_to_rot(q):
    # q = [q0, q1, q2, q3] scalar-first (w, x, y, z)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def rot_to_quat(mat):
    r = R.from_matrix(mat)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])

def quat_mul(q, rq):
    # quaternion multiply (w,x,y,z)
    w0,x0,y0,z0 = q
    w1,x1,y1,z1 = rq
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def omega_to_quat_dot(q, omega):
    # omega is body rates [p q r]
    p, qy, r = omega
    Omega = np.array([0.0, p, qy, r])
    return 0.5 * quat_mul(q, Omega)

# -------------------------
# Aircraft model
# -------------------------
class Aircraft:
    def __init__(self):
        # mass properties (example light GA aircraft)
        self.m = 1200.0            # kg
        self.I = np.diag([1400., 1800., 2600.])  # kg.m^2 (Ixx, Iyy, Izz)
        # geometry
        self.S = 16.2              # wing area m^2
        self.c = 1.5               # mean chord m
        self.b = 10.0              # wingspan m
        # aerodynamic coefficients (baseline)
        self.CL0 = 0.2
        self.CL_alpha = 5.5        # per rad
        self.CD0 = 0.02
        self.k = 0.045             # induced drag factor
        # pitching moment coefficients
        self.Cm0 = 0.0
        self.Cm_alpha = -1.2       # per rad
        self.Cm_q = -8.0           # per rad
        # control derivative (elevator)
        self.CL_de = 0.8           # per rad (lift change)
        self.Cm_de = -1.1          # per rad (pitching moment)
        # roll/yaw control (ailerons, rudder)
        self.Cl_da = 0.08          # roll moment per rad
        self.Cn_dr = -0.06         # yaw moment per rad
        # damping derivatives approximations
        self.Cl_p = -0.5
        self.Cn_r = -0.2
        self.Cl_r = 0.1
        # engine / thrust
        self.T_max = 2000.0        # N max
        # atmosphere
        self.rho0 = 1.225

    def rho(self, alt):
        # ISA-like simple (up to small altitudes)
        return max(0.1, self.rho0 * np.exp(-alt/8500.0))

    def aerodynamic_forces(self, vel_body, omega, alpha, beta, control, speed, alt):
        """
        vel_body: [u v w]
        omega: [p q r]
        alpha, beta: angles (rad)
        control: dict with 'de', 'da', 'dr' (radians)
        speed: airspeed magnitude
        alt: altitude (m)
        returns (F_body, M_body)
        """
        rho = self.rho(alt)
        qbar = 0.5 * rho * speed**2
        # Lift and Drag (approx)
        CL = self.CL0 + self.CL_alpha * alpha + self.CL_de * control['de']
        L = qbar * self.S * CL
        CD = self.CD0 + self.k * (CL**2)
        D = qbar * self.S * CD

        # Convert lift/drag in body axes:
        # assume small sideslip => D along -u, L approx along -w/apple: compute using alpha
        # Aerodynamic force in body frame:
        Fx = -D * cos(alpha) + -L * sin(alpha)  # approx
        Fz = -D * sin(alpha) + L * cos(alpha)
        # Side force from rudder/sideslip (linear)
        Cy_beta = -0.5  # per rad
        Fy = qbar * self.S * (Cy_beta * beta)  # simple

        F_body = np.array([Fx, Fy, Fz])

        # Moments: roll, pitch, yaw
        Cl = qbar * self.S * self.b * ( self.Cl_da * control['da'] + self.Cl_p * (omega[0]*self.b/(2*speed+1e-6)) + self.Cl_r * (omega[2]*self.b/(2*speed+1e-6)) )
        Cm = qbar * self.S * self.c * ( self.Cm0 + self.Cm_alpha * alpha + self.Cm_q * (omega[1]*self.c/(2*speed+1e-6)) + self.Cm_de * control['de'] )
        Cn = qbar * self.S * self.b * ( self.Cn_dr * control['dr'] + self.Cn_r * (omega[2]*self.b/(2*speed+1e-6)) )

        M_body = np.array([Cl, Cm, Cn])
        return F_body, M_body

    def thrust_force(self, throttle):
        # simple linear thrust: throttle 0-1
        return np.array([throttle * self.T_max, 0.0, 0.0])

# -------------------------
# Dynamics / Integrator
# -------------------------
class Simulator:
    def __init__(self, aircraft):
        self.ac = aircraft
        self.reset()

    def reset(self):
        # state: pos_inertial (3), vel_body (3), quat (4 w,x,y,z), omega_body (3)
        self.pos = np.array([0.0, 0.0, -1000.0])  # NED: negative z is altitude -1000m => alt=1000m
        self.vel_body = np.array([60.0, 0.0, 0.0])  # initial u ~ 60 m/s
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.time = 0.0

    def state_vector(self):
        return np.concatenate([self.pos, self.vel_body, self.quat, self.omega])

    def step(self, dt, controls):
        # RK4 integrate state
        y0 = self.state_vector()
        k1 = self._derivatives(y0, controls)
        k2 = self._derivatives(y0 + 0.5*dt*k1, controls)
        k3 = self._derivatives(y0 + 0.5*dt*k2, controls)
        k4 = self._derivatives(y0 + dt*k3, controls)
        yf = y0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # unpack
        self._unpack_state(yf)
        self.time += dt

    def _unpack_state(self, y):
        self.pos = y[0:3]
        self.vel_body = y[3:6]
        self.quat = y[6:10]
        self.quat = self.quat / np.linalg.norm(self.quat)  # normalize
        self.omega = y[10:13]

    def _derivatives(self, y, controls):
        pos = y[0:3]
        vel_body = y[3:6]
        quat = y[6:10]
        omega = y[10:13]

        # compute rotation matrix body->inertial
        R_bi = quat_to_rot(quat)
        # compute inertial gravity vector expressed in body frame
        g_inertial = np.array([0.0, 0.0, -9.80665])
        g_body = R_bi.T @ g_inertial  # gravity in body frame

        # airspeed and angles
        u, v, w = vel_body
        V = max(1e-3, np.linalg.norm(vel_body))
        alpha = atan2(w, u)  # angle of attack
        beta = atan2(v, sqrt(u*u + w*w))

        # aerodynamic forces and moments
        F_aero, M_aero = self.ac.aerodynamic_forces(vel_body, omega, alpha, beta, controls, V, -pos[2])  # altitude is -pos[2]

        # thrust
        F_thrust = self.ac.thrust_force(controls['throttle'])

        # total forces in body frame
        F_total = F_aero + F_thrust + self.ac.m * g_body

        # translational acceleration (body frame)
        v_dot = (1.0/self.ac.m) * F_total - np.cross(omega, vel_body)

        # rotational dynamics
        I = self.ac.I
        omega_vec = omega
        M_total = M_aero  # no prop moments for now

        omega_dot = np.linalg.inv(I) @ (M_total - np.cross(omega_vec, I @ omega_vec))

        # quaternion derivative
        quat_dot = omega_to_quat_dot(quat, omega_vec)

        # inertial pos rate = R_bi @ vel_body
        pos_dot = R_bi @ vel_body

        return np.concatenate([pos_dot, v_dot, quat_dot, omega_dot])

# -------------------------
# Streamlit UI + main loop
# -------------------------
st.title("6-DOF Flight Simulator — Created by Khert")

# Left control panel
col1, col2 = st.columns([1,2])

with col1:
    st.header("Controls")
    # initial condition controls
    if 'sim' not in st.session_state:
        st.session_state['sim'] = Simulator(Aircraft())
    sim = st.session_state['sim']

    dt = st.number_input("Time step (s)", value=0.02, min_value=0.001, max_value=0.5, step=0.01, format="%.3f")
    run = st.button("Run / Resume")
    pause = st.button("Pause")
    reset = st.button("Reset sim")
    single_step = st.button("Single step")

    st.subheader("Control surfaces")
    throttle = st.slider("Throttle (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    de_deg = st.slider("Elevator (deg)", -20.0, 20.0, 0.0, 0.5)
    da_deg = st.slider("Aileron (deg)", -20.0, 20.0, 0.0, 0.5)
    dr_deg = st.slider("Rudder (deg)", -20.0, 20.0, 0.0, 0.5)

    controls = {
        'throttle': float(throttle),
        'de': np.deg2rad(float(de_deg)),
        'da': np.deg2rad(float(da_deg)),
        'dr': np.deg2rad(float(dr_deg))
    }

    st.subheader("Initial conditions")
    if st.button("Apply trimmed-ish initial speed"):
        sim.reset()
        sim.vel_body = np.array([70.0, 0.0, 0.0])
        sim.pos = np.array([0.0, 0.0, -1000.0])
        sim.quat = np.array([1.0, 0.0, 0.0, 0.0])
    if reset:
        sim.reset()

with col2:
    st.header("Live visualization")
    plot_area = st.empty()
    three_d_area = st.empty()
    metrics = st.columns(4)
    t_disp = metrics[0]
    alt_disp = metrics[1]
    speed_disp = metrics[2]
    attitude_disp = metrics[3]

# Data buffers
if 'buf' not in st.session_state:
    st.session_state['buf'] = {
        't': [],
        'pos': [],
        'V': [],
        'alt': [],
        'euler': []
    }

buf = st.session_state['buf']

# Loop runner
running = run and not pause
# if pause pressed, stop running
if pause:
    running = False

# If single step pressed or running -> step once or many steps
steps_per_frame = int(max(1, round(0.1 / dt)))  # integrate a few steps per page update
max_steps = 10000

if single_step:
    running = False
    for _ in range(1):
        sim.step(dt, controls)
        # collect
        V = np.linalg.norm(sim.vel_body)
        alt = -sim.pos[2]
        eul = R.from_quat([sim.quat[1], sim.quat[2], sim.quat[3], sim.quat[0]]).as_euler('xyz', degrees=True)
        buf['t'].append(sim.time); buf['pos'].append(sim.pos.copy()); buf['V'].append(V); buf['alt'].append(alt); buf['euler'].append(eul)

if running:
    # run for a short duration to keep UI responsive
    steps = 0
    t0 = time.time()
    while steps < max_steps and (time.time() - t0) < 1.2:  # limit CPU per cycle
        sim.step(dt, controls)
        steps += 1
        V = np.linalg.norm(sim.vel_body)
        alt = -sim.pos[2]
        eul = R.from_quat([sim.quat[1], sim.quat[2], sim.quat[3], sim.quat[0]]).as_euler('xyz', degrees=True)
        buf['t'].append(sim.time); buf['pos'].append(sim.pos.copy()); buf['V'].append(V); buf['alt'].append(alt); buf['euler'].append(eul)

# trim buffer to reasonable length
max_len = 2000
if len(buf['t']) > max_len:
    for k in buf: buf[k] = buf[k][-max_len:]

# Update metrics
if len(buf['t'])>0:
    t_disp.metric("t (s)", f"{buf['t'][-1]:.1f}")
    alt_disp.metric("Altitude (m)", f"{buf['alt'][-1]:.1f}")
    speed_disp.metric("Speed (m/s)", f"{buf['V'][-1]:.1f}")
    e = buf['euler'][-1]; attitude_disp.metric("Euler (deg)", f"ϕ={e[0]:.1f} θ={e[1]:.1f} ψ={e[2]:.1f}")

# Plotting
fig, axs = plt.subplots(2,2, figsize=(10,6))
axs = axs.ravel()

if len(buf['t'])>1:
    t = np.array(buf['t'])
    alt = np.array(buf['alt'])
    V = np.array(buf['V'])
    eulers = np.array(buf['euler'])
    axs[0].plot(t, alt); axs[0].set_title("Altitude (m)"); axs[0].set_xlabel("t (s)")
    axs[1].plot(t, V); axs[1].set_title("Airspeed (m/s)"); axs[1].set_xlabel("t (s)")
    axs[2].plot(t, eulers[:,0]); axs[2].set_title("Roll (deg)"); axs[3].plot(t, eulers[:,1]); axs[3].set_title("Pitch (deg)")
else:
    axs[0].text(0.5,0.5,"No data yet", ha='center')

plt.tight_layout()
plot_area.pyplot(fig)

# 3D trajectory + simple aircraft orientation
fig3 = plt.figure(figsize=(6,5))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_title("Trajectory and Attitude (inertial coords, NED)")
if len(buf['pos'])>1:
    pts = np.array(buf['pos'])
    ax3.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=1)
    # show current aircraft axes
    pos_curr = pts[-1]
    quat = sim.quat
    Rb = quat_to_rot(quat)
    # body axes length
    L = 10.0
    x_axis = Rb @ np.array([L,0,0])
    y_axis = Rb @ np.array([0,L,0])
    z_axis = Rb @ np.array([0,0,L])
    ax3.quiver(pos_curr[0], pos_curr[1], pos_curr[2], x_axis[0], x_axis[1], x_axis[2], length=1.0, normalize=False)
    ax3.quiver(pos_curr[0], pos_curr[1], pos_curr[2], y_axis[0], y_axis[1], y_axis[2], length=1.0, normalize=False)
    ax3.quiver(pos_curr[0], pos_curr[1], pos_curr[2], z_axis[0], z_axis[1], z_axis[2], length=1.0, normalize=False)
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    # auto scale
    ax3.set_box_aspect([1,1,0.5])
else:
    ax3.text(0.5,0.5,0.5,"No trajectory yet", ha='center')

three_d_area.pyplot(fig3)

st.write("Notes: This is a flexible codebase — modify aerodynamic coefficients and moments for other aircraft. "
         "The aerodynamic model is simplified but suited for control design, simulation, and prototyping.")

st.write("Export/Inspect internal state by reading `st.session_state['sim']` and `st.session_state['buf']` in the console.")
