# Objective 2.2 - Linear Regression: predicting goals scored by a team in a match

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

script_dir = Path(__file__).resolve().parent
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


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


# 2. Predictors and target 

predictors = [
    "Team_ELO",                         
    "Opponent_ELO",                     
    "Team_Prev_Avg_Goals_Scored",        
    "Opponent_Prev_Avg_Goals_Conceded",  
    "Host_Nation",
    "Knockout_Stage",
    "Team_Squad_Market_Value_M_EUR",
    "Team_Avg_Squad_Age",
]
target = "Goals_Scored"

objective_2_1 = [
    "Team_FIFA_Ranking", "Opponent_FIFA_Ranking", "Team_ELO", "Opponent_ELO",
    "Team_Rest_Days", "Team_Prev_Avg_Goals_Scored",
    "Opponent_Prev_Avg_Goals_Conceded", "Team_Prev_Avg_Shots_On_Target",
]
shared = sorted(set(predictors) & set(objective_2_1))
print(f"\nPredictors: {len(predictors)} | shared with Objective 2.1: {len(shared)}")
assert len(predictors) == 8 and len(shared) <= 4, "Predictor rules violated."

X, y = df[predictors], df[target]

# 3.  Train-test split (70:30, random_state=42)

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
