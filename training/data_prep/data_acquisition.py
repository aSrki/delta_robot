import csv
import numpy as np
import mujoco

XML_PATH = "delta_robot/xml/delta_robot.xml"
OUT_CSV = "data/delta_ik_data.csv"

N_EPISODES = 50_000     # number of random targets

SPEED_MIN = 0.5         # rad/s: joint-space speed magnitude, drawn per episode
SPEED_MAX = 3.0

MARGIN = 0.25           # rad: keep targets away from limits
TARGET_TOL = 0.02       # rad: considered "reached"
MAX_EP_TIME = 1.2       # s: safety cap on episode length

MAX_RATIO = 30.0        # rad/s per m/s: |qvel|/|ee_speed| near-singular
MIN_JOINT_SPEED = 0.05  # rad/s: below this the ratio is meaningless

LOG_START = 0.06        # s: skip the actuator transient after the command starts
LOG_EVERY = 50          # log every N-th step
WARMUP_TIME = 0.20      # s: let the model settle under gravity before starting

RESET_HOME = False      # False: next episode starts where the last one ended
                        # True:  reset to the home pose every episode

rng = np.random.default_rng(42)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

joint_names = ["theta1", "theta2", "theta3"]
qpos_adr = np.array([model.joint(n).qposadr[0] for n in joint_names])
dof_adr = np.array([model.joint(n).dofadr[0] for n in joint_names])
nj = len(joint_names)

jnt_lo = np.array([model.joint(n).range[0] for n in joint_names])
jnt_hi = np.array([model.joint(n).range[1] for n in joint_names])
tgt_lo = jnt_lo + MARGIN
tgt_hi = jnt_hi - MARGIN

dt = model.opt.timestep
log_start_step = int(LOG_START / dt)
max_ep_steps = int(MAX_EP_TIME / dt)
warmup_steps = int(WARMUP_TIME / dt)

data.ctrl[:nj] = 0.0
for _ in range(warmup_steps):
    mujoco.mj_step(model, data)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["qpos_1", "qpos_2", "qpos_3",
         "qvel_1", "qvel_2", "qvel_3",
         "ee_vx", "ee_vy", "ee_vz"]
    )

    for ep in range(N_EPISODES):
        if RESET_HOME:
            mujoco.mj_resetData(model, data)
            data.ctrl[:nj] = 0.0
            for _ in range(warmup_steps):
                mujoco.mj_step(model, data)

        q_start = data.qpos[qpos_adr].copy()

        if (not np.all(np.isfinite(q_start))
                or np.any(q_start > tgt_hi) or np.any(q_start < tgt_lo)):
            mujoco.mj_resetData(model, data)
            data.ctrl[:nj] = 0.0
            for _ in range(warmup_steps):
                mujoco.mj_step(model, data)
            q_start = data.qpos[qpos_adr].copy()

        target = rng.uniform(tgt_lo, tgt_hi)

        delta = target - q_start
        dist = np.linalg.norm(delta)
        if dist < TARGET_TOL:
            continue                              
        direction = delta / dist

        # constant velocity command for the whole episode
        speed = rng.uniform(SPEED_MIN, SPEED_MAX)
        data.ctrl[:nj] = speed * direction

        for k in range(max_ep_steps):
            mujoco.mj_step(model, data)

            q = data.qpos[qpos_adr]
            qv = data.qvel[dof_adr]
            ev = data.sensor("ee_vel").data

            # stop - passed the target
            if np.dot(target - q, direction) <= 0.0:
                break

            # stop - any joint approaching a limit
            if np.any(q > jnt_hi - MARGIN) or np.any(q < jnt_lo + MARGIN):
                break

            jspeed = np.linalg.norm(qv)
            espeed = np.linalg.norm(ev)

            # stop - loop closure lost
            if not (np.isfinite(jspeed) and np.isfinite(espeed)):
                break

            # stop - near-singular
            if jspeed > MIN_JOINT_SPEED and espeed * MAX_RATIO < jspeed:
                break

            if k < log_start_step or (k % LOG_EVERY) != 0:
                continue

            writer.writerow(list(q) + list(qv) + list(ev))

        data.ctrl[:nj] = 0.0

