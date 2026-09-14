import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

CSV_PATH = "data/delta_ik_data.csv"
MODEL_PATH = "models/ik_model_dirnorm.joblib"

MIN_SPEED = 0.05     
SINGULAR_PCT = 99.9
TEST_SIZE = 0.15
SEARCH_ROWS = 50000
SEARCH_ITERS = 20
CV_FOLDS = 3

# load data
df = pd.read_csv(CSV_PATH, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]

qpos = df[["qpos_1", "qpos_2", "qpos_3"]].values
qvel = df[["qvel_1", "qvel_2", "qvel_3"]].values
ee_vel = df[["ee_vx", "ee_vy", "ee_vz"]].values

# drop slow rows, normalize by ee speed
speed = np.linalg.norm(ee_vel, axis=1)
keep = speed > MIN_SPEED
qpos = qpos[keep]
qvel = qvel[keep]
ee_vel = ee_vel[keep]
speed = speed[keep]

ee_dir = ee_vel / speed[:, None]
qvel_n = qvel / speed[:, None]

# drop rows near singularities
mag = np.linalg.norm(qvel_n, axis=1)
cut = np.percentile(mag, SINGULAR_PCT)
ok = mag < cut

X = np.hstack([qpos[ok], ee_dir[ok]])
y = qvel_n[ok]
print(f"final dataset: {len(X)} rows")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
Xtr = x_scaler.transform(X_train)
ytr = y_scaler.transform(y_train)
Xte = x_scaler.transform(X_test)

# hyperparameter search on a subsample
param_space = {
    "hidden_layer_sizes": [(64, 64), (128, 128), (256, 256),
                           (128, 128, 128), (256, 256, 256), (64, 64, 64, 64)],
    "alpha": [1e-6, 1e-5, 1e-4, 1e-3],
    "learning_rate_init": [3e-4, 1e-3, 3e-3],
    "batch_size": [128, 256, 512],
    "activation": ["relu", "tanh"],
}

base = MLPRegressor(solver="adam", max_iter=200, early_stopping=True,
                    n_iter_no_change=15, random_state=42)

rng = np.random.default_rng(42)
sub = rng.choice(len(Xtr), size=min(SEARCH_ROWS, len(Xtr)), replace=False)

search = RandomizedSearchCV(base, param_space, n_iter=SEARCH_ITERS, cv=CV_FOLDS,
                            scoring="r2", n_jobs=-1, random_state=42, verbose=2)
print("\n--- hyperparameter search ---")
t0 = time.perf_counter()
search.fit(Xtr[sub], ytr[sub])
print(f"search took {time.perf_counter() - t0:.1f} s")
print("best params:", search.best_params_)
print(f"best CV R^2: {search.best_score_:.5f}")

# final training
net = MLPRegressor(solver="adam", max_iter=800, early_stopping=True,
                   n_iter_no_change=40, verbose=True, random_state=42,
                   **search.best_params_)
print("\n--- final training on full training set ---")
net.fit(Xtr, ytr)

# evaluation
y_pred = y_scaler.inverse_transform(net.predict(Xte))
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))
print(f"\nTest R^2: {r2:.5f}")
print("Per-joint RMSE [rad/s per m/s]:", rmse)
print(f"Relative RMS error: {np.sqrt(max(0.0, 1.0 - r2)) * 100:.2f} %")

# inference timing
n_timing = 1000
t0 = time.perf_counter()
for i in range(n_timing):
    x = X_test[i % len(X_test)].reshape(1, -1)
    y_scaler.inverse_transform(net.predict(x_scaler.transform(x)))
t_single = (time.perf_counter() - t0) / n_timing

n_batch = min(10000, len(X_test))
t0 = time.perf_counter()
y_scaler.inverse_transform(net.predict(x_scaler.transform(X_test[:n_batch])))
t_batch = (time.perf_counter() - t0) / n_batch

print(f"\nInference time:")
print(f"  single call: {t_single * 1e3:.3f} ms  ({1 / t_single:.0f} Hz)")
print(f"  batched:     {t_batch * 1e6:.2f} us/sample")

joblib.dump({"net": net, "x_scaler": x_scaler, "y_scaler": y_scaler}, MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")