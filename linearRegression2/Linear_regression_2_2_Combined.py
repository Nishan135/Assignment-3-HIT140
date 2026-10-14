# Objective 2.2 - Linear Regression: predicting goals scored by a team in a match

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, PoissonRegressor, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

script_dir = Path(__file__).resolve().parent
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CANDIDATES = [
    BASE_DIR / "Match data.csv",
    BASE_DIR / "Match_data.csv",
    BASE_DIR / "Match_data_Copy.csv",
]
SOURCE_FILE = next((p for p in SOURCE_CANDIDATES if p.exists()), None)
OUTPUT_FILE = BASE_DIR / "linear_regression_2_2_dataset.csv"


# Actual feature names used in Objective 2.1 (Lr21 regression.PY).
# Compare concepts as well as names when checking the <=4 shared-predictor rule.
OBJECTIVE_2_1_PREDICTORS = [
    "value_ratio_log", "caps_per_million_diff", "xg_matchup_edge",
    "clinical_finishing_diff", "shot_volume_diff", "set_piece_edge",
    "keeper_workload_diff", "continental_familiarity_diff",
]

PREDICTORS = [
    # Objective 2.2 predictors (verify conceptual overlap with 2.1)
    "Team_ELO", "Opponent_ELO",
    "Team_Prev_Avg_Goals_Scored", "Opponent_Prev_Avg_Goals_Conceded",
    # additional Objective 2.2 predictors
    "Host_Nation", "Knockout_Stage",
    "Team_Squad_Market_Value_M_EUR", "Team_Avg_Squad_Age",
]
TARGET = "Goals_Scored"


def read_source(path: Path) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
    header = [h if h.strip() else "blank_col" for h in header]

    # Rows WC26_103 / WC26_104 store the home/away values as pairs
    # (rank, rank, elo, elo, host, host, ...) instead of the normal layout.
    tail = header[header.index("home_rest_days"):]
    alt_cols = [
        "Match_ID", "date", "kickoff_time_utc", "stage_id", "is_knockout",
        "home_team_id", "home_team_name", "home_fifa_code", "home_confederation",
        "away_team_id", "away_team_name", "away_fifa_code", "away_confederation",
        "venue_id", "stadium_name", "venue_city", "venue_country",
        "venue_capacity", "venue_elevation_meters", "referee_id", "referee_name",
        "referee_avg_cards",
        "home_fifa_rank", "away_fifa_rank", "home_elo", "away_elo",
        "home_is_host", "away_is_host", "home_squad_avg_age", "away_squad_avg_age",
        "home_squad_total_caps", "away_squad_total_caps",
        "home_squad_total_value_eur", "away_squad_total_value_eur",
        "home_squad_avg_value_eur", "away_squad_avg_value_eur",
    ] + tail

    elo_idx = header.index("home_elo")
    normal, paired = {}, {}
    for r in body:
        try:
            float(r[elo_idx])
            normal[r[0]] = dict(zip(header, r))
        except ValueError:
            r = r[: len(alt_cols)]
            r += [""] * (len(alt_cols) - len(r))
            paired[r[0]] = dict(zip(alt_cols, r))

    
    check_cols = ["home_elo", "away_elo", "home_fifa_rank", "away_fifa_rank",
                  "home_is_host", "away_is_host", "home_squad_total_value_eur",
                  "away_squad_total_value_eur", "home_squad_avg_age",
                  "away_squad_avg_age", "home_rest_days", "away_rest_days",
                  "home_prev_avg_goals_scored", "away_prev_avg_goals_scored",
                  "home_prev_avg_goals_conceded", "away_prev_avg_goals_conceded",
                  "home_score", "away_score"]
    mismatches = 0
    for mid in set(normal) & set(paired):
        for c in check_cols:
            if abs(float(normal[mid][c]) - float(paired[mid][c])) > 1e-6:
                mismatches += 1
    print(f"Cross-check of the two layouts on {len(set(normal) & set(paired))} matches: "
          f"{mismatches} mismatching values")

    only_paired = sorted(set(paired) - set(normal))
    if only_paired:
        print("Matches available only in the paired layout (parsed from it):",
              ", ".join(only_paired))
    merged = {**paired, **normal}
    return pd.DataFrame([merged[k] for k in sorted(merged)])


NUMERIC = [
    "is_knockout", "home_elo", "away_elo", "home_is_host", "away_is_host",
    "home_squad_total_value_eur", "away_squad_total_value_eur",
    "home_squad_avg_age", "away_squad_avg_age",
    "home_prev_avg_goals_scored", "away_prev_avg_goals_scored",
    "home_prev_avg_goals_conceded", "away_prev_avg_goals_conceded",
    "home_score", "away_score",
]


def build_rows(m):
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "Match_ID": m["Match_ID"],
            "Match_Date": m["date"],
            "Team": m[f"{side}_team_name"],
            "Opponent": m[f"{other}_team_name"],
            "Venue_Role": side.capitalize(),
            "Team_ELO": m[f"{side}_elo"],
            "Opponent_ELO": m[f"{other}_elo"],
            "Team_Prev_Avg_Goals_Scored": m[f"{side}_prev_avg_goals_scored"],
            "Opponent_Prev_Avg_Goals_Conceded": m[f"{other}_prev_avg_goals_conceded"],
            "Host_Nation": m[f"{side}_is_host"],
            "Knockout_Stage": m["is_knockout"],
            "Team_Squad_Market_Value_M_EUR": m[f"{side}_squad_total_value_eur"] / 1e6,
            "Team_Avg_Squad_Age": m[f"{side}_squad_avg_age"],
            "Goals_Scored": m[f"{side}_score"],
        }))
    out = pd.concat(rows).sort_values(["Match_ID", "Venue_Role"], ascending=[True, True])
    # Home row first within each match
    out["_o"] = (out["Venue_Role"] == "Away").astype(int)
    return out.sort_values(["Match_ID", "_o"]).drop(columns="_o").reset_index(drop=True)


def main():
    # --- overlap check against Objective 2.1 ---
    shared = sorted(set(PREDICTORS) & set(OBJECTIVE_2_1_PREDICTORS))
    print(f"Predictors: {len(PREDICTORS)} | shared with 2.1: {len(shared)} -> {shared}")
    if len(PREDICTORS) != 8 or len(set(PREDICTORS)) != 8:
        raise ValueError("Exactly 8 distinct explanatory variables are required.")
    if len(shared) > 4:
        raise ValueError(f"{len(shared)} variables shared with 2.1; the limit is 4.")

    if SOURCE_FILE is None:
        if not OUTPUT_FILE.exists():
            raise FileNotFoundError(
                "No source match dataset was found. Expected one of: "
                + ", ".join(str(p) for p in SOURCE_CANDIDATES)
            )
        print(f"Source file not found. Validating existing dataset: {OUTPUT_FILE.resolve()}")
        result = pd.read_csv(OUTPUT_FILE)
        expected_cols = (
            ["Match_ID", "Match_Date", "Team", "Opponent", "Venue_Role"]
            + PREDICTORS + [TARGET]
        )
        missing_cols = sorted(set(expected_cols) - set(result.columns))
        if missing_cols:
            raise ValueError(
                "Existing dataset is missing required columns: "
                + ", ".join(missing_cols)
            )
        result = result[expected_cols]
    else:
        print(f"Using source file: {SOURCE_FILE.resolve()}")
        src = read_source(SOURCE_FILE)
        src[NUMERIC] = src[NUMERIC].apply(pd.to_numeric, errors="coerce")
        src = src.dropna(subset=NUMERIC).reset_index(drop=True)
        if src["Match_ID"].duplicated().any():
            raise ValueError("Duplicate Match_ID values in source.")

        result = build_rows(src)
        for c in ["Host_Nation", "Knockout_Stage"]:
            result[c] = result[c].astype(int)
        cols = ["Match_ID", "Match_Date", "Team", "Opponent", "Venue_Role"] + PREDICTORS + [TARGET]
        result = result[cols]

        if len(result) != 2 * len(src):
            raise ValueError("Expected two rows per match.")

    result[PREDICTORS + [TARGET]] = result[PREDICTORS + [TARGET]].apply(
        pd.to_numeric, errors="coerce"
    )
    if result[PREDICTORS + [TARGET]].isna().any().any():
        raise ValueError("Dataset contains missing or non-numeric predictor/target values.")
    assert (result.groupby("Match_ID").size() == 2).all()
    assert not result.duplicated(["Match_ID", "Team"]).any()

    if SOURCE_FILE is not None:
        result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("\nDataset created:", OUTPUT_FILE.resolve())
    else:
        print("\nExisting dataset is valid:", OUTPUT_FILE.resolve())
    print("Matches:", result["Match_ID"].nunique(), "| team-match rows:", len(result))
    print("\nDescriptive statistics:")
    print(result[PREDICTORS + [TARGET]].describe().round(3).to_string())


if __name__ == "__main__":
    main()


# 1. Load and validate

candidates = [
    script_dir / "linear_regression_2_2_dataset.csv",
    script_dir / "Objective_2_2_Linear_Regression_Dataset.csv",
]
dataset_path = next((p for p in candidates if p.is_file()), None)
if dataset_path is None:
    raise FileNotFoundError("Dataset not found. Checked:\n" + "\n".join(map(str, candidates)))
df = pd.read_csv(dataset_path)

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nRows per Match_ID:\n", df["Match_ID"].value_counts().value_counts())


# 2. Predictors and target (8 predictors; verify conceptual overlap)

predictors = [
    "Team_ELO",                          # shared with 2.1
    "Opponent_ELO",                      # shared with 2.1
    "Team_Prev_Avg_Goals_Scored",        # shared with 2.1
    "Opponent_Prev_Avg_Goals_Conceded",  # shared with 2.1
    "Host_Nation",
    "Knockout_Stage",
    "Team_Squad_Market_Value_M_EUR",
    "Team_Avg_Squad_Age",
]
target = "Goals_Scored"

objective_2_1 = OBJECTIVE_2_1_PREDICTORS

shared = sorted(set(predictors) & set(objective_2_1))
print(f"\nPredictors: {len(predictors)} | shared with Objective 2.1: {len(shared)}")
assert len(predictors) == 8 and len(shared) <= 4, "Predictor rules violated."

X, y = df[predictors], df[target]


# 3. Train-test split (70:30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))

# 4. Evaluation helper
def evaluate_model(name, model, Xtr=None, Xte=None, ytr=None, yte=None):
    Xtr = X_train if Xtr is None else Xtr
    Xte = X_test if Xte is None else Xte
    ytr = y_train if ytr is None else ytr
    yte = y_test if yte is None else yte
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mse = mean_squared_error(yte, pred)
    return {"Model": name, "R2": r2_score(yte, pred),
            "MAE": mean_absolute_error(yte, pred), "MSE": mse, "RMSE": np.sqrt(mse)}



# 5. Linear Regression

linear_model = LinearRegression()
linear_results = evaluate_model("Linear Regression", linear_model)
print("\nLinear Regression:\n", pd.Series(linear_results))

coef_table = pd.DataFrame({"Predictor": predictors, "Coefficient": linear_model.coef_}
                          ).sort_values("Coefficient", ascending=False)
print("\nLinear Regression Coefficients:\n", coef_table)
print("\nIntercept:", linear_model.intercept_)


# 6. Correlation and multicollinearity

print("\nPredictor Correlation Matrix:\n", X.corr().round(3))


def calculate_vif(frame, col):
    """VIF = 1 / (1 - R2) from regressing one predictor on the others (with intercept)."""
    others = frame.drop(columns=col)
    r2 = LinearRegression().fit(others, frame[col]).score(others, frame[col])
    return 1 / (1 - r2)


vif = pd.DataFrame({"Predictor": predictors,
                    "VIF": [calculate_vif(X, c) for c in predictors]})
print("\nVariance Inflation Factors:\n", vif.sort_values("VIF", ascending=False).round(3))


# 7. Ridge Regression
ridge_model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
ridge_results = evaluate_model("Ridge Regression", ridge_model)
print("\nRidge Regression:\n", pd.Series(ridge_results))

# 8. Random Forest benchmark
rf_model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
rf_results = evaluate_model("Random Forest", rf_model)
print("\nRandom Forest:\n", pd.Series(rf_results))


# 9. Final comparison

comparison = pd.DataFrame([linear_results, ridge_results, rf_results])
print("\nFinal Model Comparison:\n", comparison.round(3))
comparison.to_csv(script_dir / "Objective_2_2_Model_Comparison.csv", index=False)


# 10. Robustness checks (primary result above stays the prescribed 70:30 split)

# (a) Grouped split: both teams from a match stay together in train or test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr, te = next(gss.split(X, y, groups=df["Match_ID"]))
g = evaluate_model("Linear Regression (grouped by Match_ID)", LinearRegression(),
                   X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te])
print("\nGrouped Match_ID check (train/test rows: %d/%d):" % (len(tr), len(te)))
print(pd.Series({k: v for k, v in g.items() if k != 'Model'}).round(4))

# (b) Drop each team's first appearance: its "previous form" values are placeholders
# because the team has no earlier matches.
df_sorted = df.sort_values("Match_ID")
first_rows = df_sorted.groupby("Team").head(1).index
sub = df.drop(index=first_rows)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(sub[predictors], sub[target],
                                              test_size=0.30, random_state=42)
s = evaluate_model("Linear Regression (first appearances removed)", LinearRegression(),
                   Xs_tr, Xs_te, ys_tr, ys_te)
print("\nFirst appearances removed (rows used: %d):" % len(sub))
print(pd.Series({k: v for k, v in s.items() if k != 'Model'}).round(4))


# PART B - Systematic comparison of competing algorithms
# Repeated grouped cross-validation (5 folds x 20 repeats, grouped by Match_ID so the
# two teams of one match are never split between training and test data)

groups = df["Match_ID"]

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=42, max_iter=20000)),
    "Poisson Regression": make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=2,
                                           random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                                   learning_rate=0.05, random_state=42),
}

# ---------- repeated grouped cross-validation ----------
match_ids = groups.unique()
N_FOLDS, N_REPEATS = 5, 20
rng = np.random.default_rng(42)
records, oof_linear = [], None

for rep in range(N_REPEATS):
    shuffled = rng.permutation(match_ids)
    fold_of = {m: i % N_FOLDS for i, m in enumerate(shuffled)}
    folds = groups.map(fold_of).to_numpy()
    for name, model in models.items():
        oof = np.zeros(len(df))
        for k in range(N_FOLDS):
            tr, te = folds != k, folds == k
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            oof[te] = pred
            records.append({"Model": name, "Repeat": rep, "Fold": k,
                            "R2": r2_score(y[te], pred),
                            "MAE": mean_absolute_error(y[te], pred),
                            "RMSE": np.sqrt(mean_squared_error(y[te], pred))})
        if rep == 0 and name == "Linear Regression":
            oof_linear = oof.copy()

res = pd.DataFrame(records)
# per-repeat mean, then spread across repeats
per_rep = res.groupby(["Model", "Repeat"])[["R2", "MAE", "RMSE"]].mean().reset_index()
summary = per_rep.groupby("Model").agg(
    R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
    MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"))
order = ["Mean baseline", "Linear Regression", "Ridge", "Lasso",
         "Poisson Regression", "Random Forest", "Gradient Boosting"]
summary = summary.loc[order]
base_rmse = summary.loc["Mean baseline", "RMSE_mean"]
summary["RMSE_gain_vs_baseline_%"] = (1 - summary["RMSE_mean"] / base_rmse) * 100
summary.round(4).to_csv(script_dir / "Objective_2_2_CV_Comparison.csv")
print(summary.round(3).to_string())

# Is Poisson/Ridge better than Linear on the same folds? paired RMSE differences
lin = per_rep[per_rep.Model == "Linear Regression"].set_index("Repeat")["RMSE"]
for m in ["Ridge", "Lasso", "Poisson Regression", "Random Forest", "Gradient Boosting"]:
    d = lin - per_rep[per_rep.Model == m].set_index("Repeat")["RMSE"]
    print(f"{m}: RMSE improvement over Linear = {d.mean():+.4f} (sd {d.std():.4f}), "
          f"better in {int((d > 0).sum())}/{N_REPEATS} repeats")


# PART C - Figures

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
NAVY, ORANGE, GREY, TEAL = "#1F3864", "#E07B39", "#8C8C8C", "#2A9D8F"
labels = {"Team_ELO": "Team Elo", "Opponent_ELO": "Opponent Elo",
          "Team_Prev_Avg_Goals_Scored": "Team recent goals scored",
          "Opponent_Prev_Avg_Goals_Conceded": "Opponent recent goals conceded",
          "Host_Nation": "Host nation", "Knockout_Stage": "Knockout stage",
          "Team_Squad_Market_Value_M_EUR": "Squad market value",
          "Team_Avg_Squad_Age": "Squad average age"}

# (a) standardised coefficients: goals per 1 SD change in predictor
scaler = StandardScaler().fit(X)
lr_std = LinearRegression().fit(scaler.transform(X), y)
coef = pd.Series(lr_std.coef_, index=predictors).sort_values()

# (b) calibration of out-of-fold predictions
cal = pd.DataFrame({"pred": oof_linear, "actual": y.to_numpy()})
cal["bin"] = pd.qcut(cal["pred"], 6, labels=False, duplicates="drop")
g = cal.groupby("bin").agg(pred=("pred", "mean"), actual=("actual", "mean"),
                           se=("actual", lambda s: s.std() / np.sqrt(len(s))))

# (c) goals are counts: observed distribution vs Poisson with the same mean
from scipy.stats import poisson
ks = np.arange(0, 8)
obs = y.value_counts(normalize=True).reindex(ks, fill_value=0).to_numpy()
pois = poisson.pmf(ks, y.mean())

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
ax = axes[0]
cols = [ORANGE if v < 0 else NAVY for v in coef.values]
ax.barh([labels[i] for i in coef.index], coef.values, color=cols)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in goals per 1 SD increase")
ax.set_title("(a) What moves a team's goals?", loc="left", fontweight="bold")

ax = axes[1]
lim = [g.pred.min() - 0.15, g.pred.max() + 0.15]
ax.plot(lim, lim, "--", color=GREY, label="Perfect calibration")
ax.errorbar(g.pred, g.actual, yerr=1.96 * g.se, fmt="o-", color=NAVY, capsize=3, label="Observed (95% CI)")
ax.set_xlabel("Predicted goals (cross-validated)")
ax.set_ylabel("Average actual goals")
ax.set_title("(b) Predictions track reality", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

ax = axes[2]
w = 0.38
ax.bar(ks - w / 2, obs, w, color=NAVY, label="Observed")
ax.bar(ks + w / 2, pois, w, color=ORANGE, label="Poisson (same mean)")
ax.set_xlabel("Goals scored by a team in a match")
ax.set_ylabel("Share of team-matches")
ax.set_title("(c) Goals are rare events", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(script_dir / "Fig1_Story.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.1))
s = summary.iloc[::-1]
colors = [GREY if m == "Mean baseline" else (TEAL if m == "Poisson Regression" else NAVY) for m in s.index]
for i, (m, r) in enumerate(s.iterrows()):
    ax.errorbar(r.RMSE_mean, i, xerr=r.RMSE_sd, fmt="o", color=colors[i], ecolor=colors[i],
                capsize=3, markersize=7)
ax.set_yticks(range(len(s)))
ax.set_yticklabels(s.index)
ax.grid(axis="x", alpha=0.25)
ax.axvline(base_rmse, color=ORANGE, ls="--", lw=1.2)
ax.text(base_rmse, len(s) - 0.35, " baseline", color=ORANGE, fontsize=8, va="bottom")
ax.set_xlim(s["RMSE_mean"].min() - 0.05, base_rmse + 0.08)
for i, (m, r) in enumerate(s.iterrows()):
    ax.text(r.RMSE_mean + r.RMSE_sd + 0.004, i, f"R² {r.R2_mean:.2f}", va="center", fontsize=8)
ax.set_xlabel("Cross-validated RMSE in goals (lower is better; bars = spread over 20 repeats)")
fig.tight_layout()
fig.savefig(script_dir / "Fig2_Model_Comparison.png", dpi=200)
plt.close(fig)
print("figures saved")
