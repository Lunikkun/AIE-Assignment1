import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_overall_table(results: dict) -> pd.DataFrame:
    rows = []
    for ranker_name, vals in results.items():
        rows.append(
            {
                "Ranker": ranker_name,
                "Recall@10": vals["Recall@10"],
                "NDCG@10": vals["NDCG@10"],
                "Diversity@10": vals["Diversity@10"],
                "Users": vals["evaluated_users"],
            }
        )
    return pd.DataFrame(rows)


def build_mmr_table(results: dict) -> pd.DataFrame:
    rows = []
    for ranker_name, vals in results.items():
        if ranker_name.startswith("R5_mmr_on_sgd_alpha_"):
            alpha = float(ranker_name.split("_")[-1])
            rows.append(
                {
                    "Ranker": ranker_name,
                    "Alpha": alpha,
                    "Recall@10": vals["Recall@10"],
                    "NDCG@10": vals["NDCG@10"],
                    "Diversity@10": vals["Diversity@10"],
                    "Users": vals["evaluated_users"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Ranker", "Alpha", "Recall@10", "NDCG@10", "Diversity@10", "Users"])
    return pd.DataFrame(rows).sort_values(by="Alpha").reset_index(drop=True)


def build_best_table(df_overall: pd.DataFrame) -> pd.DataFrame:
    best_recall = df_overall.loc[df_overall["Recall@10"].idxmax()]
    best_ndcg = df_overall.loc[df_overall["NDCG@10"].idxmax()]
    best_diversity = df_overall.loc[df_overall["Diversity@10"].idxmax()]

    return pd.DataFrame(
        [
            {"Metric": "Recall@10", "Best Ranker": best_recall["Ranker"], "Value": best_recall["Recall@10"]},
            {"Metric": "NDCG@10", "Best Ranker": best_ndcg["Ranker"], "Value": best_ndcg["NDCG@10"]},
            {
                "Metric": "Diversity@10",
                "Best Ranker": best_diversity["Ranker"],
                "Value": best_diversity["Diversity@10"],
            },
        ]
    )


def parse_discussion_line(line: str):
    line = " ".join(line.strip().split())
    if ": overlap top10 round1->round5 = " not in line or ", rho=" not in line or ", drift medio stato=" not in line:
        raise ValueError(f"Invalid format: {line}")

    session, rest = line.split(": overlap top10 round1->round5 = ", 1)
    overlap, rest = rest.split(", rho=", 1)
    rho_str, drift_str = rest.split(", drift medio stato=", 1)

    rho = float(rho_str.strip())
    drift = float(drift_str.strip().rstrip("."))
    representation = "MF q_i" if "_mf_" in session else "Genre vector"

    return session, overlap, rho, drift, representation


def build_personalization_table(results_dir: Path) -> pd.DataFrame:
    rows = []
    for user_id in [1, 2, 3]:
        summary_path = results_dir / "personalization" / f"user_{user_id}" / f"summary_user{user_id}.json"
        if not summary_path.exists():
            continue

        summary = load_json(summary_path)
        for line in summary.get("discussion", []):
            session, overlap, rho, drift, representation = parse_discussion_line(line)
            rows.append(
                {
                    "User": user_id,
                    "Session": session,
                    "Representation": representation,
                    "Rho": rho,
                    "Overlap R1->R5": overlap,
                    "Avg State Drift": drift,
                }
            )

    columns = ["User", "Session", "Representation", "Rho", "Overlap R1->R5", "Avg State Drift"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Generate non-text summary tables from recommender results.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="results/metrics/metrics_at_10.json",
        help="Path to metrics JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory that contains personalization folders.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/results_tables.xlsx",
        help="Output Excel file path.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found {metrics_path}")

    metrics_data = load_json(metrics_path)
    results = metrics_data.get("results", {})
    if not results:
        raise ValueError("Missing results.")

    df_overall = build_overall_table(results)
    df_mmr = build_mmr_table(results)
    df_best = build_best_table(df_overall)
    df_personalization = build_personalization_table(results_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_overall.to_excel(writer, sheet_name="performance_overall", index=False)
        df_mmr.to_excel(writer, sheet_name="mmr_tuning", index=False)
        df_best.to_excel(writer, sheet_name="best_metrics", index=False)
        df_personalization.to_excel(writer, sheet_name="personalization", index=False)

    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
