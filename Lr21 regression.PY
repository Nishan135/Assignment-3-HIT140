#Linear Regression 2.1 - Predicting goal difference in FIFA World Cup 2026 matches.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
FEATURES = [
    "value_ratio_log", "caps_per_million_diff", "xg_matchup_edge",
    "clinical_finishing_diff", "shot_volume_diff", "set_piece_edge",
    "keeper_workload_diff", "continental_familiarity_diff",
]
TARGET = "goal_diff"


def evaluate(y_true, y_pred, label):
    """Return MAE, MSE, RMSE, NRMSE (range-normalised) and R2 for one set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (y_true.max() - y_true.min())
    r2 = r2_score(y_true, y_pred)
    return {"Model": label, "MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R2": r2}


df = pd.read_csv("lr21_creative_dataset.csv")
print("Shape:", df.shape)
print("Missing values:", int(df[FEATURES + [TARGET]].isna().sum().sum()))
print(df[FEATURES + [TARGET]].describe().T.round(3), "\n")

X = df[FEATURES]
y = df[TARGET]

# 2a. response distribution of data
plt.figure(figsize=(6, 4))
sns.histplot(y, bins=range(-5, 8), discrete=True, color="steelblue")
plt.title("Distribution of goal difference")
plt.xlabel("goal_diff (home - away)")
plt.tight_layout()
plt.savefig("fig1_goal_diff_distribution.png", dpi=150)
plt.close()

# 2b. linearity: each explanatory variable vs goal_diff
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.ravel(), FEATURES):
    ax.scatter(df[col], y, alpha=0.6, color="seagreen")
    slope, intercept = np.polyfit(df[col], y, 1)
    xs = np.linspace(df[col].min(), df[col].max(), 50)
    ax.plot(xs, slope * xs + intercept, color="red")
    r = df[col].corr(y)
    ax.set_title(f"{col}\nr = {r:.2f}", fontsize=10)
    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("goal_diff")
plt.tight_layout()
plt.savefig("fig2_linearity_scatterplots.png", dpi=150)
plt.close()

# 2c. correlation heatmap (multicollinearity check)
plt.figure(figsize=(9, 7))
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig("fig3_correlation_matrix.png", dpi=150)
plt.close()

# 2d. variance inflation factors (VIF > 5 suggests problematic collinearity)
Xc = sm.add_constant(X)
vif = pd.DataFrame({
    "variable": FEATURES,
    "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(len(FEATURES))],
})
print("Variance inflation factors:\n", vif.round(2).to_string(index=False), "\n")

#  3. SPLIT + TRAIN (sklearn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

equation = f"goal_diff = {model.intercept_:.4f}"
for name, coef in zip(FEATURES, model.coef_):
    equation += f" {'+' if coef >= 0 else '-'} {abs(coef):.4f} ({name})"
print("\nRegression equation (sklearn, training set):\n", equation, "\n")

#  4. EVALUATE + BASELINE
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# baseline: always predict the mean goal_diff of the TRAINING set
baseline_pred_test = np.full(len(y_test), y_train.mean())

results = pd.DataFrame([
    evaluate(y_train, y_pred_train, "Linear regression - train"),
    evaluate(y_test, y_pred_test, "Linear regression - test"),
    evaluate(y_test, baseline_pred_test, "Baseline (train mean) - test"),
])
print("Performance:\n", results.round(4).to_string(index=False), "\n")
results.to_csv("model_performance.csv", index=False)

#  5. STATSMODELS
ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(ols.summary(), "\n")

# standardised coefficients (z-scored X) so effect sizes are comparable across variables
Xz = (X_train - X_train.mean()) / X_train.std()
ols_z = sm.OLS((y_train - y_train.mean()) / y_train.std(), sm.add_constant(Xz)).fit()
coef_table = pd.DataFrame({
    "coef": ols.params, "p_value": ols.pvalues,
    "std_coef": ols_z.params,
}).round(4)
print("Coefficient table:\n", coef_table, "\n")
coef_table.to_csv("coefficients.csv")

#  6. CROSS-VALIDATION
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
cv_rmse = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_root_mean_squared_error")
print(f"5-fold CV R2   : {cv_r2.round(3)}  mean = {cv_r2.mean():.3f}")
print(f"5-fold CV RMSE : {cv_rmse.round(3)}  mean = {cv_rmse.mean():.3f}\n")

#  7. PLOTS + RESIDUALS
# predicted vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, color="darkorange", alpha=0.8)
lims = [min(y_test.min(), y_pred_test.min()) - 0.5, max(y_test.max(), y_pred_test.max()) + 0.5]
plt.plot(lims, lims, "k--", label="perfect prediction")
plt.xlabel("Actual goal_diff")
plt.ylabel("Predicted goal_diff")
plt.title("Predicted vs actual (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_predicted_vs_actual.png", dpi=150)
plt.close()

# residuals vs fitted + residual histogram (training set, from statsmodels)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(ols.fittedvalues, ols.resid, alpha=0.7)
ax[0].axhline(0, color="red")
ax[0].set_xlabel("Fitted values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residuals vs fitted")
sns.histplot(ols.resid, kde=True, ax=ax[1], color="purple")
ax[1].set_title("Residual distribution")
plt.tight_layout()
plt.savefig("fig5_residual_diagnostics.png", dpi=150)
plt.close()

# standardised coefficients bar chart.
plt.figure(figsize=(8, 4.5))
sc = coef_table["std_coef"].drop("const").sort_values()
plt.barh(sc.index, sc.values, color=["tomato" if v < 0 else "steelblue" for v in sc.values])
plt.axvline(0, color="black", lw=0.8)
plt.title("Standardised coefficients (effect size per 1 SD)")
plt.tight_layout()
plt.savefig("fig6_standardised_coefficients.png", dpi=150)
plt.close()

print("Saved figures fig1-fig6, model_performance.csv, coefficients.csv")