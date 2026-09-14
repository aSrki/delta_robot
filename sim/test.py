import time

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

XML_PATH = "delta_robot/xml/delta_robot.xml"
JOINT_VEL = [0.5, 0.1, 0.6]     # commanded joint velocities (rad/s)
DURATION = 15.0                 # seconds

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
ee_site = model.site("ee").id
mujoco.mj_forward(model, data)

dt = model.opt.timestep
n_steps = int(DURATION / dt)

times = np.zeros(n_steps)
ee_vel = np.zeros((n_steps, 3))
vel6 = np.zeros(6)

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(1.0)                      
    data.ctrl[:] = JOINT_VEL             

    start = time.perf_counter()
    for k in range(n_steps):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()

        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                 ee_site, vel6, 0)   
        times[k] = (k + 1) * dt
        ee_vel[k] = vel6[3:6]            

        lag = start + (k + 1) * dt - time.perf_counter()
        if lag > 0:
            time.sleep(lag)

plt.figure(figsize=(8, 5))
plt.plot(times, ee_vel[:, 0], label="vx")
plt.plot(times, ee_vel[:, 1], label="vy")
plt.plot(times, ee_vel[:, 2], label="vz")
plt.xlabel("time (s)")
plt.ylabel("EE velocity (m/s)")
plt.title(f"EE velocity, joint cmd = {JOINT_VEL} rad/s")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()