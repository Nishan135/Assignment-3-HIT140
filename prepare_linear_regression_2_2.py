
import csv
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = next(
    (p for p in [BASE_DIR / "Match data.csv",
                 BASE_DIR / "Match_data.csv",
                 BASE_DIR / "Match_data_Copy.csv"] if p.exists()),
    BASE_DIR / "Match data.csv",
)
OUTPUT_FILE = BASE_DIR / "linear_regression_2_2_dataset.csv"


OBJECTIVE_2_1_PREDICTORS = [
    "Team_FIFA_Ranking", "Opponent_FIFA_Ranking", "Team_ELO", "Opponent_ELO",
    "Team_Rest_Days", "Team_Prev_Avg_Goals_Scored",
    "Opponent_Prev_Avg_Goals_Conceded", "Team_Prev_Avg_Shots_On_Target",
]
PREDICTORS = [
    # shared with 2.1 (4)
    "Team_ELO", "Opponent_ELO",
    "Team_Prev_Avg_Goals_Scored", "Opponent_Prev_Avg_Goals_Conceded",
    # new in 2.2 (4)
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

    assert len(result) == 2 * len(src), "Expected two rows per match."
    assert (result.groupby("Match_ID").size() == 2).all()
    assert not result.duplicated(["Match_ID", "Team"]).any()
    assert result[PREDICTORS + [TARGET]].isna().sum().sum() == 0

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("\nDataset created:", OUTPUT_FILE.resolve())
    print("Source matches:", len(src), "| team-match rows:", len(result))
    print("\nDescriptive statistics:")
    print(result[PREDICTORS + [TARGET]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
