import mujoco, mujoco.viewer
import numpy as np
import math, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  OFFICIAL PARAMETERS  (get_params.m)
# ═══════════════════════════════════════════════════════════
LH   = 0.096;    LK = 0.1545;   LB = 0.6585
DB   = 0.052;    HB = 0.1965
M1   = 0.268;    M2 = 2.365;    M3 = 0.332;   M4 = 0.149
# CoM positions (MATLAB link frame convention)
rx3  = 0.04826;  rz3 = 0.07709   # thigh CoM
rx4  = 0.00207;  rz4 = 0.14512   # shank CoM

Nh   = 26.9;     Nk = 28.8;     Ir = 7e-6
kT   = 0.0135;   kv = 0.0186;   Rw = 1.3;    Vmax = 12.0

# CORRECT torque polytope (from get_params.m p.uHip / p.uKnee)
# Constraint: A*[w; tau] <= b
# Hip:  [0 1; 0 -1; 0.1391 1; -0.1391 -1] * [w;tau] <= [4.3578;4.3578;3.3356;3.3356]
# Knee: [0 1; 0 -1; 0.1594 1; -0.1594 -1] * [w;tau] <= [4.6656;4.6656;3.5712;3.5712]
def clamp_hip(tau, w):
    hi = min(4.3578,  3.3356 - 0.1391*w)
    lo = max(-4.3578, -(3.3356 + 0.1391*w))
    return float(np.clip(tau, lo, hi))

def clamp_knee(tau, w):
    hi = min(4.6656,  3.5712 - 0.1594*w)
    lo = max(-4.6656, -(3.5712 + 0.1594*w))
    return float(np.clip(tau, lo, hi))

# Yaw
YAW_TARGET = 1.2;  KP_YAW = 20;  KI_YAW = 5;  YAW_MAX = 30

# Controller gains (get_params.m)
Tst   = 0.35
Kp_sw = 150.0;   Kd_sw = 5.0;   Krh = 0.1
Kp_st = 0.03;    Kd_st = 0.08
q_d   = np.array([math.pi/3, -math.pi/2])

# GRF Bezier (get_params.m)
Fx_bz = np.array([0, 0, -25, 0, 0])
Fz_bz = np.array([0, 20, 100, 0, 0])

g    = 9.81
DT   = 0.001
LAM  = 10.0    # velocity filter bandwidth


def polyval_bz(c, s):
    """Bezier polynomial evaluation (MATLAB polyval_bz)"""
    n = len(c) - 1
    s = float(np.clip(s, 0, 1))
    return sum(c[i] * math.comb(n,i) * s**i * (1-s)**(n-i) for i in range(n+1))


# ═══════════════════════════════════════════════════════════
#  LEG KINEMATICS  (gen/fcn_J_toe_HIP.m, fcn_p_toe_HIP.m)
#  p(5)=LH, p(6)=DB/2, p(7)=0  →  sqrt(p6²+p7²)=DB/2=0.026
# ═══════════════════════════════════════════════════════════
DK2 = DB / 2   # = 0.026 m

def p_toe(q3, q4):
    """Foot position in hip frame (MATLAB fcn_p_toe_HIP, 2×1)"""
    return np.array([
        LH*math.sin(q3) + math.cos(q3)*math.sin(q4)*DK2 + math.cos(q4)*math.sin(q3)*DK2,
        math.sin(q3)*math.sin(q4)*DK2 - math.cos(q3)*math.cos(q4)*DK2 - LH*math.cos(q3)
    ])

def J_toe(q3, q4):
    """Leg Jacobian in hip frame (MATLAB fcn_J_toe_HIP, 2×2)"""
    return np.array([
        [LH*math.cos(q3) + math.cos(q3)*math.cos(q4)*DK2 - math.sin(q3)*math.sin(q4)*DK2,
         math.cos(q3)*math.cos(q4)*DK2 - math.sin(q3)*math.sin(q4)*DK2],
        [LH*math.sin(q3) + math.cos(q3)*math.sin(q4)*DK2 + math.cos(q4)*math.sin(q3)*DK2,
         math.cos(q3)*math.sin(q4)*DK2 + math.cos(q4)*math.sin(q3)*DK2]
    ])


# ═══════════════════════════════════════════════════════════
#  GRAVITY COMPENSATION  (from fcn_Ge.m)
#  Simplified 2-DOF gravity for joint3 and joint4
#  (uses MATLAB CoM positions and masses)
# ═══════════════════════════════════════════════════════════
def gravity_comp(q2, q3, q4):
    """Feedforward gravity torques for hip (Ge3) and knee (Ge4)"""
    cq2 = math.cos(q2)
    # Ge(3): gravity torque about hip joint
    Ge3 = g * cq2 * (
        M3*(LH*math.sin(q3)) +
        M4*(LH*math.sin(q3) + LK*math.sin(q3+q4))
    )
    # Ge(4): gravity torque about knee joint
    Ge4 = g * cq2 * M4 * LK * math.sin(q3+q4)
    return Ge3, Ge4


# ═══════════════════════════════════════════════════════════
#  IMPACT MAP  (fcn_impactMap.m)
#  At touchdown: redistribute velocities via J'*impulse
#  so foot velocity = 0 post-impact (hard contact)
# ═══════════════════════════════════════════════════════════
def impact_map(q3, q4, dq3, dq4, q2):
    """
    Apply impact map (fcn_impactMap.m) for 2-DOF leg.
    Returns (dq3_post, dq4_post) after foot contact.
    """
    J = J_toe(q3, q4)   # 2×2
    # Simplified 2×2 inertia (leg only, approximate)
    I33 = M3*LH**2 + M4*(LH**2 + LK**2 + 2*LH*LK*math.cos(q4))
    I34 = M4*(LK**2 + LH*LK*math.cos(q4))
    I44 = M4*LK**2
    De = np.array([[I33, I34], [I34, I44]])
    # Solve: [De, -J'; J, 0] * [dq_post; F_imp] = [De*dq_prev; 0]
    dq_prev = np.array([dq3, dq4])
    A = np.block([[De, -J.T], [J, np.zeros((2,2))]])
    b = np.concatenate([De @ dq_prev, np.zeros(2)])
    try:
        sol = np.linalg.solve(A, b)
        return sol[0], sol[1]
    except np.linalg.LinAlgError:
        return dq3, dq4   # fallback: unchanged


# ═══════════════════════════════════════════════════════════
#  VELOCITY FILTER
# ═══════════════════════════════════════════════════════════
class VelFilter:
    def __init__(self): self._p=0.0; self._f=0.0
    def reset(self, p0=0.0): self._p=p0; self._f=0.0
    def update(self, q):
        self._f = (1-LAM*DT)*self._f + LAM*DT*(q-self._p)/DT
        self._p = q
        return self._f


# ═══════════════════════════════════════════════════════════
#  MAIN CONTROLLER
# ═══════════════════════════════════════════════════════════
class HoppyController:
    def __init__(self):
        self.state    = "FLIGHT"
        self.t_td     = 0.0
        self._yaw_ei  = 0.0
        self._fh      = VelFilter()
        self._fk      = VelFilter()
        self._prev_ic = False
        self._ptd     = np.zeros(2)   # touchdown foot pos (for stance Jhc)
        self.log = {k: [] for k in [
            't','th','tk','dth','dtk',
            'tau_h','tau_k','state','foot_z','touch','yaw_vel']}

    def step(self, model, data):
        t   = data.time
        q2  = data.qpos[1]   # arm angle (pitch)
        q3  = data.qpos[2]   # hip
        q4  = data.qpos[3]   # knee
        dq3 = self._fh.update(q3)
        dq4 = self._fk.update(q4)

        fz    = float(data.sensor('foot_pos').data[2])
        touch = float(data.sensor('foot_touch').data[0])
        ic    = touch > 1.0
        just_landed = ic and not self._prev_ic
        self._prev_ic = ic

        # ── IMPACT MAP at touchdown ──────────────────────────────
        if just_landed:
            dq3_new, dq4_new = impact_map(q3, q4, dq3, dq4, q2)
            data.qvel[2] = dq3_new
            data.qvel[3] = dq4_new
            mujoco.mj_forward(model, data)
            self._fh.reset(q3); self._fk.reset(q4)
            dq3, dq4 = dq3_new, dq4_new

        # ── FSM ─────────────────────────────────────────────────
        if self.state == "STANCE":
            if (t - self.t_td) >= Tst:
                self.state = "FLIGHT"
        else:
            if ic:
                self.state = "STANCE"
                self.t_td  = t
                self._ptd  = p_toe(q3, q4)   # record touchdown foot pos

        # ── SPRING (exact MATLAB formula: tau_s = -0.0242*q4 + 0.0108, ×2)
        tau_spring = 2.0 * (-0.0242*q4 + 0.0108)

        # ── GRAVITY COMPENSATION ─────────────────────────────────
        Ge3, Ge4 = gravity_comp(q2, q3, q4)

        # ── LEG TORQUES ──────────────────────────────────────────
        J = J_toe(q3, q4)

        if self.state == "STANCE":
            # dyn_stance.m: Bezier GRF feedforward + soft PD feedback
            s = float(np.clip((t - self.t_td) / Tst, 0, 1))
            Fx = polyval_bz(Fx_bz, s)
            Fz = polyval_bz(Fz_bz, s)
            tau_ff = -(J.T @ np.array([Fx, Fz]))
            tau_fb = Kp_st*(q_d - np.array([q3,q4])) + Kd_st*(-np.array([dq3,dq4]))
            tHd = tau_ff[0] + tau_fb[0] + Ge3
            tKd = tau_ff[1] + tau_fb[1] + Ge4 + tau_spring
        else:
            # dyn_aerial.m: Raibert swing foot placement + gravity comp + spring
            vx   = data.qvel[0] * LB
            p_d  = np.array([Krh*vx, -0.15])
            pf   = p_toe(q3, q4)
            vf   = J @ np.array([dq3, dq4])
            F_sw = Kp_sw*(p_d - pf) + Kd_sw*(-vf)
            tau  = J.T @ F_sw
            tHd  = tau[0] + Ge3
            tKd  = tau[1] + Ge4 + tau_spring

        # ── APPLY POLYTOPE LIMITS ────────────────────────────────
        tH = clamp_hip( tHd, dq3)
        tK = clamp_knee(tKd, dq4)
        data.ctrl[0] = tH
        data.ctrl[1] = tK

        # ── YAW PI ──────────────────────────────────────────────
        yaw_err      = YAW_TARGET - data.qvel[0]
        self._yaw_ei = float(np.clip(self._yaw_ei + yaw_err*DT, -5, 5))
        data.ctrl[2] = float(np.clip(KP_YAW*yaw_err + KI_YAW*self._yaw_ei,
                                     -YAW_MAX, YAW_MAX))

        # ── LOG ─────────────────────────────────────────────────
        for k, v in zip(self.log.keys(), [
            t, q3, q4, dq3, dq4, tH, tK,
            float(self.state=="STANCE"), fz, touch, data.qvel[0]
        ]):
            self.log[k].append(v)


# ═══════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════
def plot_results(log, save_path):
    t   = np.array(log['t'])
    sta = np.array(log['state'])
    n_td = int(np.sum(np.diff(sta.astype(int)) > 0))

    def shade(ax):
        d_ = np.diff(sta.astype(int))
        for s in np.where(d_ > 0)[0]:
            ends = np.where(d_ < 0)[0]
            e = ends[ends > s][0] if len(ends[ends > s]) else len(t)-1
            ax.axvspan(t[s], t[e], alpha=0.12, color='#e74c3c')

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"HOPPY — MATLAB dynamics (gravity comp + impact map + Bezier GRF)\n"
        f"tau_max: hip={4.3578:.2f}Nm  knee={4.6656:.2f}Nm  "
        f"Tst={Tst}s  {n_td} TDs = {n_td/t[-1]:.1f} Hz",
        fontsize=11, fontweight='bold')

    def panel(ax, ys, lbls, cols, ylabel, title, hlines=None):
        for y,l,c in zip(ys,lbls,cols): ax.plot(t, y, c, lw=1.3, label=l)
        if hlines:
            for v,c,l in hlines: ax.axhline(v, color=c, ls='--', lw=0.9, label=l)
        shade(ax); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)

    panel(axes[0],
          [np.degrees(log['th']), np.degrees(log['tk'])],
          ['hip [°]','knee [°]'], ['#2980b9','#e74c3c'], '[°]',
          'Joint angles  q_d=[60°, −90°]',
          hlines=[(60,'#2980b9','q_d'), (-90,'#e74c3c','q_d')])

    panel(axes[1],
          [np.array(log['foot_z'])*100],
          ['foot_z [cm]'], ['#27ae60'], '[cm]',
          'Foot height  (red=STANCE)',
          hlines=[(1.8,'#888','foot_r')])

    panel(axes[2],
          [log['tau_h'], log['tau_k']],
          ['τ_hip','τ_knee'], ['#2980b9','#e74c3c'], '[Nm]',
          'Torques (polytope limits from get_params.m)',
          hlines=[(4.3578,'#2980b9','hip_max'),(-4.3578,'#2980b9',''),
                  (4.6656,'#e74c3c','knee_max'),(-4.6656,'#e74c3c','')])

    panel(axes[3],
          [log['yaw_vel']],
          ['yaw [rad/s]'], ['#8e44ad'], '[rad/s]',
          'Orbital yaw velocity',
          hlines=[(YAW_TARGET,'#555',f'target={YAW_TARGET}')])
    axes[3].set_xlabel('Time [s]')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[OK] Plots → {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    here     = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(here, "hoppy.xml")

    print("=" * 65)
    print("  HOPPY — MATLAB dynamics (gravity comp + impact map)")
    print("=" * 65)
    print(f"  Spring : tau_s = -0.0242*q4 + 0.0108  (×2, exact MATLAB)")
    print(f"  Limits : hip ≤ 4.36 Nm  knee ≤ 4.67 Nm  (velocity-dependent)")
    print(f"  GravFF : yes  |  ImpactMap : yes  |  Bezier GRF : yes")
    print("=" * 65)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    data.qpos[0] =  0.0
    data.qpos[1] =  0.243          # arm angle (foot near floor)
    data.qpos[2] =  math.pi/3     # hip = 60°
    data.qpos[3] = -math.pi/2     # knee = −90°
    data.qvel[0] =  YAW_TARGET
    mujoco.mj_forward(model, data)

    ctrl = HoppyController()
    ctrl._fh.reset(math.pi/3);  ctrl._fk.reset(-math.pi/2)

    fz0 = data.sensor('foot_pos').data[2]
    print(f"[INFO] foot_z at IC = {fz0*1e3:.1f} mm")
    print(f"[INFO] Gravity comp at IC: "
          f"hip={gravity_comp(0.243,math.pi/3,-math.pi/2)[0]:.3f}Nm  "
          f"knee={gravity_comp(0.243,math.pi/3,-math.pi/2)[1]:.3f}Nm")

    SIM_DURATION = 15.0
    print(f"[INFO] Simulating {SIM_DURATION} s …")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance  = 1.8
            viewer.cam.azimuth   = 135
            viewer.cam.elevation = -18
            while viewer.is_running() and data.time < SIM_DURATION:
                t0 = time.time()
                ctrl.step(model, data)
                mujoco.mj_step(model, data)
                viewer.sync()
                slack = DT - (time.time() - t0)
                if slack > 0: time.sleep(slack)
    except Exception as e:
        print(f"[Viewer] {e} — headless")
        while data.time < SIM_DURATION:
            ctrl.step(model, data); mujoco.mj_step(model, data)

    sta = np.array(ctrl.log['state'])
    td  = int(np.sum(np.diff(sta.astype(int)) > 0))
    print(f"\n[OK] Touchdowns : {td}  ({td/SIM_DURATION:.1f} Hz)")
    print(f"[OK] foot_z max : {max(ctrl.log['foot_z'])*100:.1f} cm")
    print(f"[OK] τ_hip  max : {max(abs(np.array(ctrl.log['tau_h']))):.4f} Nm")
    print(f"[OK] τ_knee max : {max(abs(np.array(ctrl.log['tau_k']))):.4f} Nm")
    print(f"[OK] Yaw final  : {data.qvel[0]:.3f} rad/s")

    plot_results(ctrl.log, os.path.join(here, "hoppy_results.png"))


if __name__ == "__main__":
    main()