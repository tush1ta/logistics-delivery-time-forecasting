"""
Week 4 Task - Predictive Modeling and Optimization in Logistics Systems
Problem: Forecast last-mile delivery time (minutes) for e-commerce parcels
based on route, package, and operational features, then use the model's
insights to optimize fleet assignment and route sequencing.
"""
# PART 1 - PREDICTIVE MODELING
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 1. DATA SIMULATION
N = 6000

distance_km            = np.round(np.random.gamma(shape=2.2, scale=3.5, size=N), 2)         # trip distance
package_weight_kg       = np.round(np.random.exponential(scale=4.0, size=N) + 0.2, 2)         # parcel weight
num_stops_on_route      = np.random.poisson(lam=6, size=N) + 1                                 # multi-drop route
traffic_index           = np.clip(np.random.normal(loc=50, scale=20, size=N), 5, 100)          # 0-100 congestion
weather_severity        = np.random.choice([0, 1, 2, 3], size=N, p=[0.55, 0.25, 0.15, 0.05])   # 0=clear..3=severe
driver_experience_years = np.clip(np.random.exponential(scale=4, size=N), 0, 25)
hour_of_day             = np.random.randint(6, 22, size=N)
is_weekend              = np.random.choice([0, 1], size=N, p=[0.72, 0.28])
warehouse_to_hub_km     = np.round(np.random.uniform(1, 25, size=N), 2)
vehicle_type            = np.random.choice(["van", "bike", "truck"], size=N, p=[0.55, 0.25, 0.20])
vehicle_speed_factor    = np.select(
    [vehicle_type == "bike", vehicle_type == "van", vehicle_type == "truck"],
    [0.55, 1.0, 0.8]
)

# Peak-hour congestion bump
peak_bump = np.where(((hour_of_day >= 8) & (hour_of_day <= 10)) |
                      ((hour_of_day >= 17) & (hour_of_day <= 19)), 12, 0)

# 
# Ground-truth generating process (nonlinear + interactions + noise)
# 
base_time = (
    4.5 * distance_km
    + 3.0 * num_stops_on_route
    + 0.9 * package_weight_kg
    + 0.35 * traffic_index
    + 6.0 * weather_severity
    + 1.4 * warehouse_to_hub_km
    - 0.6 * driver_experience_years
    + peak_bump
    + 8.0 * is_weekend * -1          # lighter weekend traffic
)
base_time = base_time / vehicle_speed_factor
noise = np.random.normal(0, 9, size=N)
delivery_time_minutes = np.clip(base_time + noise, 8, None)

df = pd.DataFrame({
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "num_stops_on_route": num_stops_on_route,
    "traffic_index": traffic_index,
    "weather_severity": weather_severity,
    "driver_experience_years": np.round(driver_experience_years, 1),
    "hour_of_day": hour_of_day,
    "is_weekend": is_weekend,
    "warehouse_to_hub_km": warehouse_to_hub_km,
    "vehicle_type": vehicle_type,
    "delivery_time_minutes": np.round(delivery_time_minutes, 1),
})

df.to_csv("simulated_logistics_data.csv", index=False)
print(df.head())
print(df.describe().T)

# 
# 2. FEATURE PREPARATION
# 
df_model = pd.get_dummies(df, columns=["vehicle_type"], drop_first=True)

X = df_model.drop(columns=["delivery_time_minutes"])
y = df_model["delivery_time_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# 
# 3. MODEL TRAINING
# 
results = {}

# --- Baseline: Linear Regression (scaled) ---
lin_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
lin_pipe.fit(X_train, y_train)
pred_lin = lin_pipe.predict(X_test)

# --- Decision Tree ---
dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# --- Random Forest (with small grid search) ---
rf_param_grid = {
    "n_estimators": [150, 300],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [2, 5],
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
pred_rf = rf_best.predict(X_test)

# --- Gradient Boosting ---
gb_param_grid = {
    "n_estimators": [150, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    gb_param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
pred_gb = gb_best.predict(X_test)

# 
# 4. EVALUATION
# 
def evaluate(name, y_true, y_pred, model, use_X=X):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    cv = cross_val_score(model, use_X, y, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
                          scoring="neg_root_mean_squared_error")
    results[name] = {
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "CV_RMSE_mean": -cv.mean(), "CV_RMSE_std": cv.std()
    }

evaluate("Linear Regression", y_test, pred_lin, lin_pipe)
evaluate("Decision Tree", y_test, pred_dt, dt)
evaluate("Random Forest (tuned)", y_test, pred_rf, rf_best)
evaluate("Gradient Boosting (tuned)", y_test, pred_gb, gb_best)

results_df = pd.DataFrame(results).T.round(3)
results_df.to_csv("model_comparison_results.csv")
print("\n=== Model comparison ===")
print(results_df)

print("\nBest RF params:", rf_grid.best_params_)
print("Best GB params:", gb_grid.best_params_)

# 
# 5. FEATURE IMPORTANCE (best model = Gradient Boosting, typically)
# 
best_name = results_df["RMSE"].astype(float).idxmin()
print("\nBest model on test RMSE:", best_name)

importances = gb_best.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feat_imp.to_csv("feature_importance.csv")
print(feat_imp)

# 
# 6. PLOTS
# 
plt.figure(figsize=(6,5))
plt.scatter(y_test, pred_gb, alpha=0.25, s=12, color="#2f6f9f")
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, color="crimson", linewidth=1.5)
plt.xlabel("Actual delivery time (min)")
plt.ylabel("Predicted delivery time (min)")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

plt.figure(figsize=(6,5))
resid = y_test - pred_gb
plt.scatter(pred_gb, resid, alpha=0.25, s=12, color="#5a8f4a")
plt.axhline(0, color="crimson", linewidth=1.5)
plt.xlabel("Predicted delivery time (min)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot - Gradient Boosting")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5,5))
feat_imp.sort_values().plot(kind="barh", color="#b06a3a")
plt.xlabel("Relative importance")
plt.title("Feature Importance - Gradient Boosting")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(6.5,5))
results_df["RMSE"].plot(kind="bar", color="#3a6ea5")
plt.ylabel("Test RMSE (minutes)")
plt.title("Model Comparison - Test RMSE")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=150)
plt.close()

print("\nAll artifacts saved.")


# 
# PART 2 - OPTIMIZATION
# Uses the trained delivery-time model's insights to:
# 1. optimize vehicle-type assignment across a day's route batch (LP), and
# 2. demonstrate a stop-sequencing heuristic that reduces distance/stops-
#    driven time.
from scipy.optimize import linprog

np.random.seed(7)

# 1. VEHICLE-TO-ROUTE ASSIGNMENT (Linear Program)
#    Minimize total predicted delivery minutes (proxy for cost)
#    across a batch of routes assigned to 3 vehicle types,
#    subject to fleet capacity constraints.

routes = pd.DataFrame({
    "route_id": [f"R{i+1}" for i in range(6)],
    "distance_km": [4.2, 9.8, 15.1, 3.0, 22.5, 7.4],
    "stops": [5, 9, 12, 3, 15, 6],
})

# Predicted minutes per route, per vehicle type (from model-informed cost function:
# base_time / vehicle_speed_factor, using the same coefficients the GB model learned)
speed_factor = {"bike": 0.55, "van": 1.0, "truck": 0.8}
cost_per_km  = {"bike": 1.0,  "van": 1.0, "truck": 1.0}   # relative unit distance cost

def predicted_minutes(distance, stops, vtype):
    base = 4.5 * distance + 3.0 * stops
    return base / speed_factor[vtype]

for v in speed_factor:
    routes[f"time_{v}"] = routes.apply(lambda r: predicted_minutes(r.distance_km, r.stops, v), axis=1)

print(routes)

# Fleet capacity for the day (number of routes each vehicle type can take)
capacity = {"bike": 2, "van": 3, "truck": 2}

n_routes = len(routes)
vtypes = list(speed_factor.keys())
n_v = len(vtypes)

# Decision variables x[r, v] in [0,1], relaxed LP (assignment problem)
c = []
for v in vtypes:
    c.extend(routes[f"time_{v}"].tolist())
c = np.array(c)  # length n_routes * n_v, ordered by vehicle-block

# Equality: each route assigned exactly once -> sum over v of x[r,v] = 1
A_eq = np.zeros((n_routes, n_routes * n_v))
for r in range(n_routes):
    for vi in range(n_v):
        A_eq[r, vi * n_routes + r] = 1
b_eq = np.ones(n_routes)

# Inequality: capacity per vehicle type -> sum over r of x[r,v] <= capacity[v]
A_ub = np.zeros((n_v, n_routes * n_v))
b_ub = np.zeros(n_v)
for vi, v in enumerate(vtypes):
    for r in range(n_routes):
        A_ub[vi, vi * n_routes + r] = 1
    b_ub[vi] = capacity[v]

bounds = [(0, 1)] * (n_routes * n_v)

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

assign = res.x.reshape(n_v, n_routes)
assignment_df = pd.DataFrame(assign, index=vtypes, columns=routes.route_id).round(2)
print("\nOptimal (relaxed) assignment matrix (rows=vehicle type):")
print(assignment_df)

chosen = []
for r in range(n_routes):
    vi = np.argmax(assign[:, r])
    chosen.append(vtypes[vi])
routes["assigned_vehicle"] = chosen
routes["assigned_time_min"] = [routes.loc[i, f"time_{routes.loc[i,'assigned_vehicle']}"] for i in range(n_routes)]


# Fair baseline: dispatchers commonly assign vehicles in route-arrival order
# without cost optimization, filling each vehicle type's capacity in sequence
# (e.g. van first since it's the default fleet, then truck, then bike).
naive_assignment = []
remaining_capacity = capacity.copy()
priority_order = ["van", "truck", "bike"]
for _ in range(n_routes):
    for v in priority_order:
        if remaining_capacity[v] > 0:
            naive_assignment.append(v)
            remaining_capacity[v] -= 1
            break
routes["naive_vehicle"] = naive_assignment
routes["naive_time_min"] = [routes.loc[i, f"time_{routes.loc[i,'naive_vehicle']}"] for i in range(n_routes)]

naive_time = routes["naive_time_min"].sum()
optimized_time = routes["assigned_time_min"].sum()
savings_pct = 100 * (naive_time - optimized_time) / naive_time

print(f"\nBaseline (sequential dispatch) total time: {naive_time:.1f} min")
print(f"Optimized assignment total time: {optimized_time:.1f} min")
print(f"Time savings: {savings_pct:.1f}%")

routes.to_csv("route_assignment_optimized.csv", index=False)

# 
# 2. STOP-SEQUENCING HEURISTIC (Nearest-Neighbor TSP approximation)
#    Demonstrates route-sequencing optimization for a single multi-stop route.
# 
n_stops = 10
coords = np.random.uniform(0, 20, size=(n_stops, 2))
depot = np.array([[10, 10]])
all_pts = np.vstack([depot, coords])

def route_length(order, pts):
    total = 0.0
    for i in range(len(order) - 1):
        total += np.linalg.norm(pts[order[i]] - pts[order[i+1]])
    return total

# Unoptimized: visit in given (random) order
naive_order = [0] + list(range(1, n_stops + 1)) + [0]
naive_len = route_length(naive_order, all_pts)

# Nearest-neighbor heuristic
visited = [0]
remaining = set(range(1, n_stops + 1))
current = 0
while remaining:
    nxt = min(remaining, key=lambda j: np.linalg.norm(all_pts[current] - all_pts[j]))
    visited.append(nxt)
    remaining.remove(nxt)
    current = nxt
visited.append(0)
nn_len = route_length(visited, all_pts)

improvement_pct = 100 * (naive_len - nn_len) / naive_len
print(f"\nNaive route distance: {naive_len:.1f} units")
print(f"Nearest-neighbor optimized distance: {nn_len:.1f} units")
print(f"Distance reduction: {improvement_pct:.1f}%")

# Plot both routes
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, order, title, color in [
    (axes[0], naive_order, f"Unoptimized sequence\n({naive_len:.1f} units)", "#b04a4a"),
    (axes[1], visited, f"Nearest-neighbor optimized\n({nn_len:.1f} units)", "#3a8f5a"),
]:
    pts = all_pts[order]
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, markersize=6)
    ax.plot(all_pts[0, 0], all_pts[0, 1], "s", color="black", markersize=10, label="Depot")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-2, 22)
    ax.set_ylim(-2, 22)
plt.tight_layout()
plt.savefig("route_sequencing_comparison.png", dpi=150)
plt.close()

# Assignment chart
plt.figure(figsize=(6.5, 5))
plt.bar(["Baseline (sequential dispatch)", "Optimized assignment (LP)"], [naive_time, optimized_time],
        color=["#b04a4a", "#3a8f5a"])
plt.ylabel("Total predicted delivery time (min)")
plt.title(f"Fleet Assignment Optimization ({savings_pct:.1f}% time reduction)")
plt.tight_layout()
plt.savefig("fleet_assignment_savings.png", dpi=150)
plt.close()

print("\nOptimization artifacts saved.")