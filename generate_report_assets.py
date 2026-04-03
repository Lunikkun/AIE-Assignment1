import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


METRICS_PATH = "results/metrics/metrics_at_10.json"
OUT_DIR = "results/report_assets"


def _ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def _load_metrics_report():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def _sorted_ranker_items(results_dict):
    # Keep deterministic order by ranker id, e.g. R1_..., R2_..., ..., R5_...
    return sorted(results_dict.items(), key=lambda kv: kv[0])


def plot_ranker_metrics(results_dict):
    items = _sorted_ranker_items(results_dict)
    labels = [name for name, _ in items]
    recall = [vals.get("Recall@10", 0.0) for _, vals in items]
    ndcg = [vals.get("NDCG@10", 0.0) for _, vals in items]
    diversity = [vals.get("Diversity@10", 0.0) for _, vals in items]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w, recall, width=w, label="Recall@10")
    ax.bar(x, ndcg, width=w, label="NDCG@10")
    ax.bar(x + w, diversity, width=w, label="Diversity@10")

    ax.set_title("Ranker Metrics @10")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "ranker_metrics.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_mmr_tradeoff(results_dict):
    mmr_points = []
    for key, vals in results_dict.items():
        if key.startswith("R5_mmr_on_sgd_alpha_"):
            alpha = float(key.split("alpha_")[-1])
            mmr_points.append(
                (
                    alpha,
                    vals.get("Recall@10", 0.0),
                    vals.get("NDCG@10", 0.0),
                    vals.get("Diversity@10", 0.0),
                )
            )

    if not mmr_points:
        return

    mmr_points.sort(key=lambda t: t[0])
    alphas = [p[0] for p in mmr_points]
    recall = [p[1] for p in mmr_points]
    ndcg = [p[2] for p in mmr_points]
    diversity = [p[3] for p in mmr_points]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(alphas, recall, marker="o", label="Recall@10")
    ax.plot(alphas, ndcg, marker="o", label="NDCG@10")
    ax.plot(alphas, diversity, marker="o", label="Diversity@10")

    for a, r, n, d in mmr_points:
        ax.annotate(f"a={a}", (a, d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ax.set_title("MMR Trade-off on MF-SGD (varying alpha)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Metric value")
    ax.set_xticks(alphas)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(OUT_DIR, "mmr_tradeoff.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _collect_personalization_summary():
    rows = []
    session_files = glob("results/personalization/user_*/session_*.json")
    for path in session_files:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            user_id = int(data.get("user_id"))
            summary = data.get("summary", {})
            session_name = summary.get("session_name")
            overlap = float(summary.get("first_vs_last_slate_overlap_top10", 0.0))
            avg_drift = float(summary.get("avg_state_drift_l2", 0.0))
            rows.append((user_id, session_name, overlap, avg_drift))
        except Exception:
            # Skip malformed files to keep the script robust.
            continue
    return rows


def plot_personalization_summary():
    rows = _collect_personalization_summary()
    if not rows:
        return

    # Aggregate by session type across users.
    by_session = {}
    for _, session_name, overlap, avg_drift in rows:
        by_session.setdefault(session_name, {"overlap": [], "drift": []})
        by_session[session_name]["overlap"].append(overlap)
        by_session[session_name]["drift"].append(avg_drift)

    sessions = sorted(by_session.keys())
    overlap_mean = [np.mean(by_session[s]["overlap"]) for s in sessions]
    drift_mean = [np.mean(by_session[s]["drift"]) for s in sessions]

    x = np.arange(len(sessions))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(x, overlap_mean, color="#4C78A8", alpha=0.85, label="Top10 overlap (round1->round5)")
    ax1.set_ylabel("Overlap count on Top10")
    ax1.set_ylim(0, 10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sessions, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.plot(x, drift_mean, color="#F58518", marker="o", linewidth=2, label="Avg state drift L2")
    ax2.set_ylabel("Average state drift (L2)")

    ax1.set_title("Personalization Session Summary (mean over users)")

    labels = [b.get_label() for b in [bars]] + [line[0].get_label()]
    handles = [bars, line[0]]
    ax1.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "personalization_summary.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    _ensure_out_dir()
    report = _load_metrics_report()
    results_dict = report.get("results", {})

    if not results_dict:
        raise ValueError("No ranker results found in metrics report.")

    plot_ranker_metrics(results_dict)
    plot_mmr_tradeoff(results_dict)
    plot_personalization_summary()

    print(f"Generated plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
