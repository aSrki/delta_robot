# Learned Inverse Velocity Kinematics for a Delta Robot

MSc project. An MLP trained to replace the analytical inverse Jacobian of a
parallel delta robot. Input: current joint positions and a desired
end-effector velocity. Output: the joint velocities that produce that motion.

All simulation, data collection, evaluation, and the closed-loop demo run in
MuJoCo.

---

## Motivation

`ee_vel = J(qpos) · q_vel` is exact and cheaper than a network forward pass;
this is not an attempt to beat it on accuracy. Deriving the Jacobian by hand
for a closed-loop mechanism like a delta robot is impractical, and available
Python robotics libraries do not handle closed-loop kinematic chains well.
MuJoCo is used to solve the loop-closure constraints, and a network is trained
to learn the resulting input/output mapping from data.

---

## Method

### Direction normalization

The relation `qvel = J⁻¹(qpos) · ee_vel` is linear in `ee_vel`. To avoid
requiring the network to learn that linearity from scratch, it is factored out
by hand:

```
training:   input  = [qpos, ee_vel / |ee_vel|]
            target = qvel / |ee_vel|
inference:  qvel   = |ee_vel| · net(qpos, ee_vel / |ee_vel|)
```

The network only ever sees unit direction vectors, so the same model works at
any commanded speed, including speeds not present in the training data, since
speed scaling is applied outside the net. Inputs are 6-D (3 joint angles + 3
direction components), output is 3-D.

### Data collection

Data is collected episodically: each episode picks a random target inside the
joint limits and applies a constant velocity command until the target is
reached. The command does not change mid-episode, which keeps the
qpos/qvel/ee_vel rows labeled consistently instead of mixing transients from
different commands.

Rows only get logged once the actuator transient has settled down, and an
episode is cut short on any of:

| condition | why |
|---|---|
| target reached | projection onto the travel direction goes negative |
| joint limit approached | stop `MARGIN` before the limit, never park against it |
| near-singular pose | `\|qvel\| / \|ee_vel\| > MAX_RATIO`, platform basically stops responding |
| non-finite state | loop closure got lost, sim diverged |
| timeout | safety cap on episode length |

The singular-pose check is the most significant of these. Near a singularity
the normalized targets blow up, so a handful of extreme rows can dominate
training if left in. Rejecting them at collection time is more effective than
filtering afterward, since post-hoc filtering still leaves the near-bad
neighbors of the rejected rows in the dataset.

### Training

`MLPRegressor` from scikit-learn is used, with `StandardScaler` on both inputs
and outputs — a full deep learning framework is unnecessary for a 6-in/3-out
regression. `RandomizedSearchCV` searches over architecture, regularization,
learning rate, batch size, and activation on a subsample of the data with
shortened training; the best configuration is then retrained on the full
training set.

---

## Results

These are the numbers from the run I currently have saved in `models/`:

| metric | value |
|---|---|
| Test R² (normalized targets) | 0.99995 |
| Per-joint RMSE | 0.024 rad/s per m/s of ee speed |
| Relative RMS error | 0.68 % |
| Inference, single call | 0.29 ms (≈3.4 kHz) |
| Inference, batched | 8.5 µs/sample |

End-effector error was also checked dynamically — commanding the predicted
joint velocities to the simulator and reading the ee velocity sensor back,
rather than comparing numbers on paper alone:

| commanded | ee error [m/s per m/s] |
|---|---|
| ground-truth qvel | 0.16331 |
| predicted qvel | 0.16413 |

The gap between the two is **0.0008**, indicating the network adds negligible
end-effector error beyond what the simulator/actuators already contribute.

---

## Files

| file | role |
|---|---|
| `delta_robot/xml/delta_robot.xml` | MuJoCo model: 3 actuated shoulders, parallelogram forearms, freejoint platform, 6 `connect` loop closures |
| `training/data_prep/data_acquisition.py` | episodic data collection → `data/delta_ik_data.csv` |
| `training/train_model.py` | filtering, hyperparameter search, training, evaluation → `models/ik_model_dirnorm.joblib` |
| `delta_robot/main.py` | closed-loop demo with live viewer and desired-vs-measured tracking plots |
| `sim/test.py` | quick sanity check script — commands a fixed joint velocity and plots the resulting ee velocity, used this while building the MuJoCo model |

## Usage

```bash
pip install -r requirements.txt

python training/data_prep/data_acquisition.py
# writes data/delta_ik_data.csv

python training/train_model.py
# writes models/ik_model_dirnorm.joblib

python delta_robot/main.py
# opens the viewer + desired vs measured plots
```

The demo holds one desired end-effector velocity for two seconds. Each
timestep it reads `qpos` from the simulator, runs it through the network, and
sends the predicted joint velocities to the actuators. Change `EE_VEL` at the
top of `delta_robot/main.py` to use a different direction.

This README and some parts of the code were written using AI. All core concepts and ideas are my own.