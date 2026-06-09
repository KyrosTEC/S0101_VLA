"""
HOPPY MuJoCo Controller — Official URDF + MATLAB Physics
=========================================================
Features:
  ✓ Video recording (MP4) with --record flag
  ✓ Orbital circular hopping (yaw PI controller)
  ✓ Sensor measurements panel (live + logged plots)
  ✓ 3D Jacobian aerial controller (correct for URDF kinematics)
  ✓ Back-EMF motor model (kT=0.0135, Rw=1.3Ω)

Usage:
  python hoppy_controller.py           # viewer only
  python hoppy_controller.py --record  # viewer + save MP4

Model: HOPPY-E0-final URDF (SolidWorks)
Physics: get_params.m (UIUC RoboDesignLab)
"""

import os, sys, time, math, argparse, platform

# Set GL backend based on OS (EGL = Linux headless, Windows uses native WGL)
if platform.system() == 'Linux' and 'DISPLAY' not in os.environ:
    os.environ.setdefault('MUJOCO_GL', 'egl')
# Windows: don't set MUJOCO_GL — MuJoCo uses WGL automatically

import mujoco, mujoco.viewer
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from collections import deque

# ── Official parameters (get_params.m) ─────────────────────────
Nh=26.9;  Nk=28.8;  Ir=7e-6;  Rw=1.3;  kT=0.0135;  kv=0.0186;  Vmax=12.0
LB = 0.6585                           # boom radius [m]
TAU_HIP  = Nh * kT * (Vmax/Rw)       # 3.352 Nm
TAU_KNEE = Nk * kT * (Vmax/Rw)       # 3.589 Nm
ARM_HIP  = Nh**2 * Ir                 # 0.00507 kg·m²
ARM_KNEE = Nk**2 * Ir                 # 0.00581 kg·m²

Tst     = 0.35      # stance time [s]
Kp_aer  = 200.0     # aerial position gain
Kd_aer  = 8.0       # aerial velocity gain
LAMBDA  = 10.0      # velocity filter [rad/s]
DT      = 0.001     # timestep

YAW_TARGET = 1.2    # orbital velocity [rad/s]
KP_YAW     = 20.0
KI_YAW     = 5.0
YAW_MAX    = 15.0

def back_emf(tHd, tKd, dth, dtk):
    Vh = np.clip((Rw/(kT*Nh))*tHd + kv*Nh*dth, -Vmax, Vmax)
    Vk = np.clip((Rw/(kT*Nk))*tKd + kv*Nk*dtk, -Vmax, Vmax)
    tH = np.clip((kT*Nh/Rw)*(Vh-kv*Nh*dth), -TAU_HIP,  TAU_HIP)
    tK = np.clip((kT*Nk/Rw)*(Vk-kv*Nk*dtk), -TAU_KNEE, TAU_KNEE)
    return tH, tK, Vh, Vk

class VelFilter:
    def __init__(self): self.tp=0.0; self.df=0.0
    def reset(self,p0=0.0): self.tp=p0; self.df=0.0
    def update(self,th):
        r=(th-self.tp)/DT; self.df=(1-LAMBDA*DT)*self.df+LAMBDA*DT*r; self.tp=th
        return self.df

def leg_jacobian(m, d, site_id):
    """3×2 Jacobian mapping [dq_hip, dq_knee] → d(foot_pos) in world frame."""
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, site_id)
    return jp[:,2:4]

# ── Sensor data class ────────────────────────────────────────────
class SensorLog:
    def __init__(self, maxlen=15000):
        n = maxlen
        self.t       = deque(maxlen=n)
        # Joint sensors
        self.q_yaw   = deque(maxlen=n)
        self.q_pitch = deque(maxlen=n)
        self.q_hip   = deque(maxlen=n)
        self.q_knee  = deque(maxlen=n)
        self.dq_yaw  = deque(maxlen=n)
        self.dq_hip  = deque(maxlen=n)
        self.dq_knee = deque(maxlen=n)
        # Foot sensors
        self.foot_x  = deque(maxlen=n)
        self.foot_y  = deque(maxlen=n)
        self.foot_z  = deque(maxlen=n)
        self.touch   = deque(maxlen=n)
        # Motor outputs
        self.tau_h   = deque(maxlen=n)
        self.tau_k   = deque(maxlen=n)
        self.V_h     = deque(maxlen=n)
        self.V_k     = deque(maxlen=n)
        # State
        self.state   = deque(maxlen=n)
        self.pitch   = deque(maxlen=n)
        # Orbital trajectory (for circular plot)
        self.foot_traj_x = deque(maxlen=3000)
        self.foot_traj_y = deque(maxlen=3000)
        # Stats
        self.touchdowns = 0
        self.max_fz     = 0.0

    def record(self, data, m, state, tH, tK, VH, VK):
        t = data.time
        self.t.append(t)
        self.q_yaw.append(data.sensordata[0])
        self.q_pitch.append(data.sensordata[1])
        self.q_hip.append(data.sensordata[2])
        self.q_knee.append(data.sensordata[3])
        self.dq_yaw.append(data.sensordata[4])
        self.dq_hip.append(data.sensordata[6])
        self.dq_knee.append(data.sensordata[7])
        fx,fy,fz = data.sensordata[8], data.sensordata[9], data.sensordata[10]
        touch    = data.sensordata[11]
        self.foot_x.append(fx); self.foot_y.append(fy); self.foot_z.append(fz)
        self.touch.append(touch)
        self.tau_h.append(tH);  self.tau_k.append(tK)
        self.V_h.append(VH);    self.V_k.append(VK)
        self.state.append(float(state=='STANCE'))
        self.pitch.append(data.qpos[1])
        self.foot_traj_x.append(fx); self.foot_traj_y.append(fy)
        if fz > self.max_fz: self.max_fz = fz

# ── Controller ───────────────────────────────────────────────────
class HoppyController:
    def __init__(self):
        self.state='FLIGHT'; self.t_td=0.0; self._yaw_ei=0.0
        self.fh=VelFilter(); self.fk=VelFilter()
        self._site_id=None; self._hip_id=None
        self.sensor_log = SensorLog()
        self._prev_ic = False

    def _ids(self, m):
        self._site_id = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'foot_site')
        self._hip_id  = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Link3')

    def step(self, m, d):
        if self._site_id is None: self._ids(m)
        t=d.time; q3=d.qpos[2]; q4=d.qpos[3]
        dq3=self.fh.update(q3); dq4=self.fk.update(q4)
        fz    = float(d.sensordata[10])
        touch = float(d.sensordata[11])
        ic = (touch > 5.0)

        # FSM
        if self.state=='STANCE':
            if (t-self.t_td) >= Tst: self.state='FLIGHT'
        else:
            if ic:
                self.state='STANCE'; self.t_td=t
                self.sensor_log.touchdowns += 1
        self._prev_ic = ic

        # ── STANCE: saturated torque with Bezier envelope ──────
        if self.state=='STANCE':
            env = np.sin(math.pi*np.clip((t-self.t_td)/Tst,0,1))
            tHd = -TAU_HIP  * env
            tKd = -TAU_KNEE * env

        # ── FLIGHT: 3D Jacobian Raibert foot placement ─────────
        else:
            J = leg_jacobian(m, d, self._site_id)
            hip  = d.xpos[self._hip_id]
            foot = d.site_xpos[self._site_id]
            # Target foot position below hip (leg hangs in Y in URDF frame)
            Krh  = 0.05
            vy   = d.qvel[0] * LB   # orbital speed at hip
            tgt  = np.array([hip[0], hip[1]-0.200+Krh*vy, 0.018])
            err  = tgt - foot
            fv   = J @ np.array([dq3, dq4])
            F    = Kp_aer*err + Kd_aer*(-fv)
            tau  = J.T @ F
            tHd  = np.clip(tau[0], -TAU_HIP,  TAU_HIP)
            tKd  = np.clip(tau[1], -TAU_KNEE, TAU_KNEE)

        # Back-EMF model
        tH,tK,VH,VK = back_emf(tHd, tKd, dq3, dq4)
        d.ctrl[0]=tH; d.ctrl[1]=tK

        # Yaw orbital PI
        ye  = YAW_TARGET - d.qvel[0]
        self._yaw_ei = np.clip(self._yaw_ei + ye*DT, -5, 5)
        d.ctrl[2] = np.clip(KP_YAW*ye + KI_YAW*self._yaw_ei, -YAW_MAX, YAW_MAX)

        # Record all sensor data
        self.sensor_log.record(d, m, self.state, tH, tK, VH, VK)


# ── Result plots ─────────────────────────────────────────────────
def plot_results(log, save_path='hoppy_results.png'):
    t   = np.array(log.t)
    sta = np.array(log.state)

    def shade(ax):
        diff = np.diff(sta.astype(int))
        for s in np.where(diff>0)[0]:
            ends = np.where(diff<0)[0]
            e = ends[ends>s][0] if len(ends[ends>s]) else len(t)-1
            ax.axvspan(t[s],t[e],alpha=0.12,color='#e74c3c')

    td = log.touchdowns
    fig = plt.figure(figsize=(18,14))
    gs  = gridspec.GridSpec(4,3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "HOPPY MuJoCo — Official URDF (SolidWorks HOPPY-E0-final)\n"
        f"Sensor measurements | TDs={td} ({td/max(t[-1],1):.1f}Hz) | "
        f"τ_hip_max={TAU_HIP:.3f}Nm | τ_knee_max={TAU_KNEE:.3f}Nm",
        fontsize=12, fontweight='bold')

    # 1. Joint positions
    ax=fig.add_subplot(gs[0,0:2])
    ax.plot(t,np.degrees(log.q_hip), '#2980b9',lw=1.2,label='s_hip [°]')
    ax.plot(t,np.degrees(log.q_knee),'#e74c3c',lw=1.2,label='s_knee [°]')
    ax.axhline(60, color='#2980b9',ls='--',lw=0.8,label='q_d=60°')
    ax.axhline(-90,color='#e74c3c',ls='--',lw=0.8,label='q_d=-90°')
    shade(ax); ax.set_ylabel('[°]'); ax.legend(ncol=4,fontsize=8)
    ax.set_title('Sensor: s_hip, s_knee (joint positions)'); ax.grid(alpha=0.3)

    # 2. Joint velocities
    ax=fig.add_subplot(gs[1,0:2])
    ax.plot(t,np.array(log.dq_hip), '#2980b9',lw=1.2,label='sv_hip [rad/s]')
    ax.plot(t,np.array(log.dq_knee),'#e74c3c',lw=1.2,label='sv_knee [rad/s]')
    shade(ax); ax.set_ylabel('[rad/s]'); ax.legend(ncol=2,fontsize=8)
    ax.set_title('Sensor: sv_hip, sv_knee (joint velocities)'); ax.grid(alpha=0.3)

    # 3. Torques + voltages
    ax=fig.add_subplot(gs[2,0:2])
    ax.plot(t,log.tau_h,'#2980b9',lw=1.3,label='τ_hip [Nm]')
    ax.plot(t,log.tau_k,'#e74c3c',lw=1.3,label='τ_knee [Nm]')
    ax2=ax.twinx()
    ax2.plot(t,log.V_h,'#2980b9',lw=0.8,alpha=0.4,ls='--',label='V_hip [V]')
    ax2.plot(t,log.V_k,'#e74c3c',lw=0.8,alpha=0.4,ls='--',label='V_knee [V]')
    ax2.set_ylabel('[V]',color='gray'); ax2.set_ylim(-15,15)
    ax.axhline( TAU_HIP,color='k',ls='--',lw=0.8); ax.axhline(-TAU_HIP,color='k',ls='--',lw=0.8)
    shade(ax); ax.set_ylabel('[Nm]'); ax.legend(ncol=2,fontsize=8)
    ax.set_title('Motor: torques (solid) + voltages (dashed, right axis)'); ax.grid(alpha=0.3)

    # 4. Foot height + touch + pitch
    ax=fig.add_subplot(gs[3,0:2])
    ax.plot(t,np.array(log.foot_z)*100,  '#27ae60',lw=1.2,label='foot_z [cm]')
    ax.plot(t,np.array(log.pitch)*100,   '#8e44ad',lw=1.0,alpha=0.7,label='arm pitch [cm]')
    ax2=ax.twinx()
    ax2.plot(t,np.array(log.touch),      '#e67e22',lw=0.8,alpha=0.5,label='touch [N]')
    ax2.set_ylabel('[N]',color='#e67e22')
    ax.axhline(1.8,color='k',ls='--',lw=0.8)
    shade(ax); ax.set_ylabel('[cm]'); ax.legend(ncol=2,fontsize=8)
    ax.set_title('Sensor: foot_pos_z (height) + foot_touch force'); ax.grid(alpha=0.3)
    ax.set_xlabel('Time [s]')

    # 5. Yaw velocity (orbital)
    ax=fig.add_subplot(gs[0,2])
    ax.plot(t,np.array(log.dq_yaw),'#8e44ad',lw=1.2,label='sv_yaw')
    ax.axhline(YAW_TARGET,color='k',ls='--',lw=0.8,label=f'target={YAW_TARGET}')
    shade(ax); ax.set_ylabel('[rad/s]'); ax.legend(fontsize=8)
    ax.set_title('Sensor: sv_yaw\n(orbital velocity)'); ax.grid(alpha=0.3)

    # 6. Foot XY trajectory (circular orbit)
    ax=fig.add_subplot(gs[1,2])
    fx=np.array(log.foot_traj_x); fy=np.array(log.foot_traj_y)
    # Color by time (early=blue, late=red)
    n_pts=len(fx)
    if n_pts > 1:
        colors=plt.cm.plasma(np.linspace(0,1,n_pts))
        ax.scatter(fx,fy,c=np.linspace(0,1,n_pts),cmap='plasma',s=1,alpha=0.6)
    ax.set_xlabel('foot X [m]'); ax.set_ylabel('foot Y [m]')
    ax.set_title('Foot orbital trajectory\n(circular hopping path)'); ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # 7. Phase portrait: hip angle vs velocity
    ax=fig.add_subplot(gs[2,2])
    qh=np.degrees(log.q_hip); dqh=np.array(log.dq_hip)
    ax.scatter(qh,dqh,c=sta,cmap='RdYlGn',s=1,alpha=0.5)
    ax.set_xlabel('hip [°]'); ax.set_ylabel('ω_hip [rad/s]')
    ax.set_title('Phase portrait: hip\n(green=FLIGHT, red=STANCE)'); ax.grid(alpha=0.3)

    # 8. Hop height histogram
    ax=fig.add_subplot(gs[3,2])
    fzv=np.array(log.foot_z)*100
    ax.hist(fzv[fzv>2.5],bins=30,color='#27ae60',alpha=0.7,edgecolor='white')
    ax.axvline(np.mean(fzv[fzv>2.5]),color='k',ls='--',lw=1.5,
               label=f'mean={np.mean(fzv[fzv>2.5]):.1f}cm')
    ax.set_xlabel('foot_z [cm]'); ax.set_ylabel('count')
    ax.set_title('Hop height distribution\n(flight phase only)'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(save_path,dpi=150,bbox_inches='tight')
    print(f"[OK] Plots saved: {save_path}")
    plt.close()


# ── Video recorder ───────────────────────────────────────────────
class VideoRecorder:
    """
    Split-screen video recorder: side view (left) + top view (right).
    Works on Windows (WGL) and Linux (EGL headless).
    """
    def __init__(self, model, path='hoppy_sim.mp4', fps=30, width=1280, height=720):
        self.path     = path
        self.fps      = fps
        self.frame_dt = 1.0 / fps
        self._last_t  = -1.0
        self._frames  = 0
        self._w       = width // 2   # each half-frame width
        self._h       = height

        # Two renderers: left=side view, right=top view
        self._r_side = mujoco.Renderer(model, self._h, self._w)
        self._r_top  = mujoco.Renderer(model, self._h, self._w)

        import imageio
        self.writer = imageio.get_writer(
            path, fps=fps,
            macro_block_size=16
        )
        print(f"[REC] Recording → {path}  ({width}×{height} @ {fps}fps)")

    def maybe_capture(self, data):
        """Capture frame if enough sim-time has passed since last capture."""
        if data.time - self._last_t < self.frame_dt:
            return
        # Left: side view
        self._r_side.update_scene(data, camera='side_view')
        left = self._r_side.render().copy()
        # Right: top view (shows circular orbit)
        self._r_top.update_scene(data, camera='top_view')
        right = self._r_top.render().copy()
        # Combine side-by-side
        frame = np.concatenate([left, right], axis=1)
        self.writer.append_data(frame)
        self._last_t = data.time
        self._frames += 1

    def close(self):
        self.writer.close()
        self._r_side.close()
        self._r_top.close()
        print(f"[REC] Saved {self._frames} frames → {self.path}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='HOPPY MuJoCo Simulation')
    parser.add_argument('--record', action='store_true',
                        help='Record MP4 video to hoppy_sim.mp4')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Simulation duration in seconds (default 15)')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Run headless (no viewer window)')
    args = parser.parse_args()

    xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hoppy.xml')
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*65)
    print("  HOPPY — Official URDF (HOPPY-E0-final) + MATLAB Physics")
    print("="*65)
    print(f"  τ_hip={TAU_HIP:.3f}Nm  τ_knee={TAU_KNEE:.3f}Nm  Tst={Tst}s")
    print(f"  Motor: kT={kT}  Rw={Rw}Ω  Vmax={Vmax}V  LB={LB}m")
    print(f"  Duration: {args.duration}s  Record: {args.record}")
    print("="*65)

    model = mujoco.MjModel.from_xml_path(xml)
    data  = mujoco.MjData(model)

    # Official MATLAB IC + URDF arm angle for foot on floor
    data.qpos[0] =  0.0           # joint1 yaw
    data.qpos[1] =  0.243         # joint2 arm angle
    data.qpos[2] =  1.7128   # joint3 hip = URDF equilibrium (shank down)
    data.qpos[3] = -1.5008   # joint4 knee = URDF equilibrium
    data.qvel[0] =  YAW_TARGET   # orbital velocity
    mujoco.mj_forward(model, data)

    ctrl = HoppyController()
    ctrl.fh.reset(1.7128); ctrl.fk.reset(-1.5008)

    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'foot_site')
    print(f"[INFO] foot_z={data.sensordata[10]*1000:.1f}mm  "
          f"total_mass={sum(model.body_mass):.3f}kg")

    # Set up video recorder
    rec = None
    if args.record:
        vid_path = os.path.join(out_dir, 'hoppy_sim.mp4')
        # Add tracking camera to model if not present
        rec = VideoRecorder(model, vid_path, fps=30, width=1280, height=720)

    # Run simulation
    SIM = args.duration
    t_report = 0.0

    def run_step():
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        if rec: rec.maybe_capture(data)

    if not args.no_viewer:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Add camera tracking
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance = 2.5
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 135

                print(f"[INFO] Viewer open. Simulating {SIM}s ...")
                while viewer.is_running() and data.time < SIM:
                    t0 = time.time()
                    run_step()
                    viewer.sync()
                    dt_rem = DT - (time.time()-t0)
                    if dt_rem > 0: time.sleep(dt_rem)
                    # Print stats every 2s
                    if data.time - t_report >= 2.0:
                        td = ctrl.sensor_log.touchdowns
                        print(f"  t={data.time:.1f}s  TDs={td} ({td/max(data.time,0.1):.1f}Hz)"
                              f"  yaw={data.qvel[0]:.2f}rad/s"
                              f"  fz_max={ctrl.sensor_log.max_fz*100:.1f}cm")
        except Exception as e:
            print(f"[Viewer] {e} — running headless")
            while data.time < SIM: run_step()
    else:
        print(f"[INFO] Headless mode. Simulating {SIM}s ...")
        while data.time < SIM:
            run_step()
            if data.time - t_report >= 2.0:
                td = ctrl.sensor_log.touchdowns
                print(f"  t={data.time:.1f}s  TDs={td}  yaw={data.qvel[0]:.2f}rad/s")
                t_report = data.time

    # Close recorder
    if rec: rec.close()

    # Final stats
    log = ctrl.sensor_log
    td  = log.touchdowns
    print(f"\n{'='*50}")
    print(f"[RESULTS]")
    print(f"  Touchdowns:     {td} ({td/max(data.time,0.1):.1f} Hz)")
    print(f"  Max hop height: {log.max_fz*100:.1f} cm")
    print(f"  τ_hip used:     ±{max(abs(v) for v in log.tau_h):.4f} Nm")
    print(f"  τ_knee used:    ±{max(abs(v) for v in log.tau_k):.4f} Nm")
    print(f"  Yaw final:      {data.qvel[0]:.3f} rad/s")
    print(f"{'='*50}")

    # Save plots
    plot_path = os.path.join(out_dir, 'hoppy_results.png')
    plot_results(log, plot_path)

    if rec:
        print(f"\n[VIDEO] Saved: {os.path.join(out_dir,'hoppy_sim.mp4')}")
    print(f"[PLOTS] Saved: {plot_path}")


if __name__ == '__main__':
    main()