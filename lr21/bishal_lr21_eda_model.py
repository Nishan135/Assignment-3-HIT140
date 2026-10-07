"""
Linear Regression 2.1 - Bishal's part: EDA, modelling, evaluation
Target: goal_diff (Team A goals - Team B goals), 104 rows
Features: exactly 8 pre-match variables (edit FEATURES below)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- 1. Load data ----------
df = pd.read_csv("lr21_dataset.csv")
TARGET = "goal_diff"
FEATURES = [
    "elo_diff", "fifa_rank_diff", "avg_goals_scored_diff",
    "avg_goals_conceded_diff", "squad_value_diff", "avg_age_diff",
    "recent_form_diff", "rest_days_diff",
]
assert len(FEATURES) == 8 and len(df) == 104, "Need 8 features and 104 rows"
print(df[FEATURES + [TARGET]].describe().T)
print("Missing values:\n", df[FEATURES + [TARGET]].isna().sum())

# ---------- 2. EDA ----------
for col in FEATURES + [TARGET]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"{col}: {n_out} outliers")

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for ax, col in zip(axes.ravel(), FEATURES + [TARGET]):
    sns.boxplot(y=df[col], ax=ax)
plt.tight_layout(); plt.savefig("eda_boxplots.png", dpi=200); plt.close()

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.regplot(x=df[col], y=df[TARGET], ax=ax, scatter_kws={"s": 15})
    r = df[col].corr(df[TARGET])
    ax.set_title(f"{col} (r = {r:.2f})")
plt.tight_layout(); plt.savefig("eda_scatter.png", dpi=200); plt.close()

plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.tight_layout(); plt.savefig("eda_heatmap.png", dpi=200); plt.close()

# ---------- 3. Train/test split ----------
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- 4. Evaluation helper ----------
def evaluate(name, y_true, y_pred, y_range):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / y_range
    print(f"{name:22s} MAE={mae:.3f} MSE={mse:.3f} RMSE={rmse:.3f} NRMSE={nrmse:.3f}")
    return dict(model=name, MAE=mae, MSE=mse, RMSE=rmse, NRMSE=nrmse)

y_range = y.max() - y.min()
results = []

# ---------- 5. Baseline ----------
baseline_pred = np.full(len(y_test), y_train.mean())
results.append(evaluate("Baseline (train mean)", y_test, baseline_pred, y_range))

# ---------- 6. Multiple linear regression ----------
lr = LinearRegression().fit(X_train, y_train)
results.append(evaluate("Linear Regression", y_test, lr.predict(X_test), y_range))
coefs = pd.Series(lr.coef_, index=FEATURES).sort_values()
print("Intercept:", lr.intercept_)
print(coefs)
coefs.plot.barh(title="LR coefficients"); plt.tight_layout()
plt.savefig("coefficients.png", dpi=200); plt.close()

# ---------- 7. Competing algorithms ----------
others = {
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
for name, model in others.items():
    model.fit(X_train, y_train)
    results.append(evaluate(name, y_test, model.predict(X_test), y_range))

# ---------- 8. Residual diagnostics ----------
pred = lr.predict(X_test)
resid = y_test - pred
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(pred, resid); ax[0].axhline(0, color="red")
ax[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs predicted")
ax[1].scatter(y_test, pred); ax[1].plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax[1].set(xlabel="Actual", ylabel="Predicted", title="Actual vs predicted")
plt.tight_layout(); plt.savefig("residuals.png", dpi=200); plt.close()

# ---------- 9. Cross-validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(LinearRegression(), X, y, cv=kf,
                           scoring="neg_root_mean_squared_error")
print(f"5-fold CV RMSE: {cv_rmse.mean():.3f} +/- {cv_rmse.std():.3f}")

pd.DataFrame(results).to_csv("model_comparison.csv", index=False)