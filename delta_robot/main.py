import time
import numpy as np
import joblib
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import mplcursors

XML_PATH = "delta_robot/xml/delta_robot.xml"
MODEL_PATH = "models/ik_model_dirnorm.joblib"

EE_VEL = np.array([0.1, 0.05, -0.05])   # desired ee velocity [m/s]
TOTAL_TIME = 2.0
WARMUP_TIME = 0.10

bundle = joblib.load(MODEL_PATH)
net, xs, ys = bundle["net"], bundle["x_scaler"], bundle["y_scaler"]

speed = np.linalg.norm(EE_VEL)
ee_dir = EE_VEL / speed

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

qpos_adr = np.array([model.joint(n).qposadr[0]
                     for n in ["theta1", "theta2", "theta3"]])

dt = model.opt.timestep
n_steps = int(TOTAL_TIME / dt)

data.ctrl[:3] = 0.0
for _ in range(int(WARMUP_TIME / dt)):
    mujoco.mj_step(model, data)

log_t, log_meas = [], []

with mujoco.viewer.launch_passive(model, data) as viewer:
    wall0 = time.perf_counter()

    for k in range(n_steps):
        if not viewer.is_running():
            break

        qpos = data.qpos[qpos_adr]
        x = np.hstack([qpos, ee_dir]).reshape(1, -1)
        qvel = speed * ys.inverse_transform(net.predict(xs.transform(x))).ravel()

        data.ctrl[:3] = qvel
        mujoco.mj_step(model, data)

        log_t.append((k + 1) * dt)
        log_meas.append(data.sensor("ee_vel").data.copy())

        if k % 10 == 0:
            viewer.sync()
        lag = (k + 1) * dt - (time.perf_counter() - wall0)
        if lag > 0:
            time.sleep(lag)

    data.ctrl[:3] = 0.0

t = np.array(log_t)
meas = np.array(log_meas)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, label in enumerate(["vx", "vy", "vz"]):
    axes[i].axhline(EE_VEL[i], color="k", ls="--", lw=1.5, label="desired")
    axes[i].plot(t, meas[:, i], lw=1.2, label="measured")
    axes[i].set_ylabel(f"ee {label} [m/s]")
    axes[i].grid(alpha=0.3)
    axes[i].legend(loc="upper right")

axes[-1].set_xlabel("time [s]")
fig.suptitle("Desired vs measured end-effector velocity")
fig.tight_layout()
fig.savefig("ee_velocity_tracking.png", dpi=150)
cursor = mplcursors.cursor(axes, hover=True)
cursor.connect(
    "add",
    lambda sel: sel.annotation.set_text(
        f"t = {sel.target[0]:.3f} s\nv = {sel.target[1]:.4f} m/s"
    ),
)
plt.show()


