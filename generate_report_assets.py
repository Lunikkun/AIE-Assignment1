from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics" / "metrics_at_10.json"
PERSONALIZATION_DIR = RESULTS_DIR / "personalization"
ASSETS_DIR = RESULTS_DIR / "report_assets"
REPORT_PATH = RESULTS_DIR / "FINAL_REPORT.md"


RANKER_LABELS = {
    "R1_popularity": "Popularity",
    "R2_mf_sgd": "MF-SGD",
    "R3_mf_als": "MF-ALS",
    "R4_pairwise_mf_sgd": "Pairwise MF-SGD",
    "R4_pairwise_mf_als": "Pairwise MF-ALS",
    "R5_mmr_on_sgd_alpha_0.1": "MMR on SGD (alpha=0.1)",
    "R5_mmr_on_sgd_alpha_0.4": "MMR on SGD (alpha=0.4)",
    "R5_mmr_on_sgd_alpha_0.7": "MMR on SGD (alpha=0.7)",
}


SESSION_REGEX = re.compile(
    r"session_(?P<session>\d+)_(?P<representation>mf|genre)_rho_(?P<rho>[0-9.]+): "
    r"overlap top10 round1->round5 = (?P<overlap>\d+)/10, rho=(?P<rho_check>[0-9.]+), "
    r"(?:mean state drift|drift medio stato)=(?P<drift>[0-9.]+)\."
)


@dataclass(frozen=True)
class PersonalizationObservation:
    user_id: int
    session_id: int
    representation: str
    rho: float
    overlap: int
    drift: float

    @property
    def configuration_label(self) -> str:
        return f"{self.representation.upper()} rho={self.rho:.1f}"


def load_metrics() -> tuple[dict, dict[str, dict[str, float]]]:
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, payload["results"]


def parse_personalization() -> list[PersonalizationObservation]:
    observations: list[PersonalizationObservation] = []
    for summary_path in sorted(PERSONALIZATION_DIR.glob("user_*/summary_user*.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        user_id = int(payload["user_id"])
        for line in payload["discussion"]:
            match = SESSION_REGEX.fullmatch(line)
            if not match:
                raise ValueError(f"Unable to parse personalization summary line: {line}")
            observations.append(
                PersonalizationObservation(
                    user_id=user_id,
                    session_id=int(match.group("session")),
                    representation=match.group("representation"),
                    rho=float(match.group("rho")),
                    overlap=int(match.group("overlap")),
                    drift=float(match.group("drift")),
                )
            )
    return observations


def build_metrics_table(results: dict[str, dict[str, float]]) -> str:
    headers = ["Ranker", "Recall@10", "NDCG@10", "Diversity@10", "Users"]
    lines = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---:|"]
    for ranker_key, values in sorted(results.items()):
        label = RANKER_LABELS.get(ranker_key, ranker_key)
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{values['Recall@10']:.4f}",
                    f"{values['NDCG@10']:.4f}",
                    f"{values['Diversity@10']:.4f}",
                    str(values["evaluated_users"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_personalization_table(observations: list[PersonalizationObservation]) -> str:
    headers = ["User", "Session", "Configuration", "Overlap top10", "Mean drift"]
    lines = ["| " + " | ".join(headers) + " |", "|---:|---:|---|---:|---:|"]
    for obs in observations:
        lines.append(
            f"| {obs.user_id} | {obs.session_id} | {obs.configuration_label} | {obs.overlap}/10 | {obs.drift:.4f} |"
        )
    return "\n".join(lines)


def compute_best_rankers(results: dict[str, dict[str, float]]) -> dict[str, tuple[str, float]]:
    return {
        metric: max(results.items(), key=lambda item: item[1][metric])
        for metric in ("Recall@10", "NDCG@10", "Diversity@10")
    }


def summarize_personalization(observations: list[PersonalizationObservation]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[PersonalizationObservation]] = {}
    for obs in observations:
        grouped.setdefault(obs.configuration_label, []).append(obs)

    summary: dict[str, dict[str, float]] = {}
    for label, configs in grouped.items():
        summary[label] = {
            "mean_overlap": float(np.mean([obs.overlap for obs in configs])),
            "mean_drift": float(np.mean([obs.drift for obs in configs])),
        }
    return summary


def plot_ranker_metrics(results: dict[str, dict[str, float]]) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [RANKER_LABELS.get(key, key) for key in results]
    recall = [results[key]["Recall@10"] for key in results]
    ndcg = [results[key]["NDCG@10"] for key in results]
    diversity = [results[key]["Diversity@10"] for key in results]

    positions = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(positions - width, recall, width=width, label="Recall@10", color="#264653")
    ax.bar(positions, ndcg, width=width, label="NDCG@10", color="#2A9D8F")
    ax.bar(positions + width, diversity, width=width, label="Diversity@10", color="#E9C46A")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Comparison across available rankers")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "ranker_metrics.png", dpi=220)
    plt.close(fig)


def plot_mmr_tradeoff(results: dict[str, dict[str, float]]) -> None:
    mmr_keys = [
        "R5_mmr_on_sgd_alpha_0.1",
        "R5_mmr_on_sgd_alpha_0.4",
        "R5_mmr_on_sgd_alpha_0.7",
    ]
    alphas = [0.1, 0.4, 0.7]
    recall = [results[key]["Recall@10"] for key in mmr_keys]
    ndcg = [results[key]["NDCG@10"] for key in mmr_keys]
    diversity = [results[key]["Diversity@10"] for key in mmr_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, recall, marker="o", label="Recall@10", color="#264653")
    ax.plot(alphas, ndcg, marker="o", label="NDCG@10", color="#2A9D8F")
    ax.plot(alphas, diversity, marker="o", label="Diversity@10", color="#E76F51")
    ax.set_xlabel("MMR alpha")
    ax.set_ylabel("Metric value")
    ax.set_title("MMR on MF-SGD: relevance-diversity trade-off")
    ax.set_xticks(alphas)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "mmr_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_personalization_summary(observations: list[PersonalizationObservation]) -> None:
    summary = summarize_personalization(observations)
    labels = list(summary)
    overlap = [summary[label]["mean_overlap"] for label in labels]
    drift = [summary[label]["mean_drift"] for label in labels]

    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(positions, overlap, color="#577590")
    axes[0].set_title("Average overlap between round 1 and round 5")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Overlap out of 10")
    axes[0].set_ylim(0, 10)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(positions, drift, color="#F4A261")
    axes[1].set_title("Average latent-state drift")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Mean drift")
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "personalization_summary.png", dpi=220)
    plt.close(fig)


def build_report_text(
    payload: dict,
    results: dict[str, dict[str, float]],
    observations: list[PersonalizationObservation],
) -> str:
    best_rankers = compute_best_rankers(results)
    personalization_summary = summarize_personalization(observations)
    metrics_table = build_metrics_table(results)
    personalization_table = build_personalization_table(observations)

    best_recall_name = RANKER_LABELS.get(best_rankers["Recall@10"][0], best_rankers["Recall@10"][0])
    best_ndcg_name = RANKER_LABELS.get(best_rankers["NDCG@10"][0], best_rankers["NDCG@10"][0])
    best_diversity_name = RANKER_LABELS.get(best_rankers["Diversity@10"][0], best_rankers["Diversity@10"][0])
    mf_low = personalization_summary["MF rho=0.2"]
    mf_high = personalization_summary["MF rho=0.5"]
    genre_mid = personalization_summary["GENRE rho=0.4"]

    return f"""# MovieLens 100K Recommender System Report

## Abstract

This report documents the implementation and evaluation of a recommender-system pipeline on the MovieLens 100K dataset using the standard split {payload['dataset_split']}. The project compares popularity-based ranking, matrix-factorization models trained with SGD and ALS, pairwise learning-to-rank, and a diversification stage based on Maximal Marginal Relevance (MMR). In addition, it studies a simple conversational personalization loop in which a user state is updated through an exponential moving average over selected item representations. Results show that the popularity baseline remains the strongest method on Recall@10 and NDCG@10 in the current configuration, while MMR substantially increases Diversity@10 without materially damaging relevance. The personalization study confirms that higher rho values produce stronger state movement and lower overlap between the first and last ranking rounds.

## 1. Objective and scope

The goal of the assignment is to build a reproducible recommendation pipeline that includes ranking, diversification, evaluation, and a multi-round personalization simulation. The codebase covers five conceptual stages:

1. Training latent-factor models with MF-SGD and MF-ALS.
2. Building base rankings with popularity and latent-factor predictions.
3. Training a pairwise ranker on preference differences.
4. Applying MMR to trade off relevance and diversity.
5. Simulating iterative personalization for three users across three sessions each.

The report focuses on the experiments that are actually present in the workspace outputs. At the time of writing, the evaluation file contains eight rankers: five base methods and three MMR variants built on the MF-SGD ranking.

## 2. Dataset and experimental setup

The experiments use MovieLens 100K, with training interactions loaded from u1.base and test interactions from u1.test. An item is treated as relevant if its rating in the test set is at least 4. Metrics are computed only on users who have at least one relevant item in the test split; this yields 456 evaluated users.

The ranking depth is fixed to K = 10. Three metrics are used:

- Recall@10, averaged over evaluated users.
- Binary NDCG@10, averaged over evaluated users.
- Diversity@10, computed from pairwise cosine distance between genre vectors in the recommended list.

The latent models use 10 factors. MMR is evaluated with alpha values 0.1, 0.4, and 0.7. The personalization loop uses the update rule $u_{{t+1}} = \\text{{normalize}}((1-\\rho)u_t + \\rho v_i)$ with three session configurations: MF with rho = 0.2, MF with rho = 0.5, and genre-based vectors with rho = 0.4.

## 3. Methods

### 3.1 Popularity baseline

The popularity ranker orders items by their interaction frequency in the training set. Even though it is simple, it often performs competitively on sparse recommendation benchmarks because popular items are more likely to match held-out positives.

### 3.2 Matrix factorization

MF-SGD and MF-ALS both learn user and item latent factors. MF-SGD optimizes parameters through stochastic gradient updates, while MF-ALS alternates closed-form updates over user and item blocks. The resulting models provide predicted ratings used to sort candidate items.

### 3.3 Pairwise learning to rank

The pairwise model learns item preferences from feature differences between preferred and non-preferred items. In this project, LogisticRegression is trained on pairwise signals derived from MF representations, producing two variants: one on top of the SGD model and one on top of the ALS model.

### 3.4 MMR reranking

MMR reranks an initial ranking by balancing relevance and novelty. The implemented form chooses the next item according to

$$
c^* = \\arg\\max_{{c \\in C \\setminus S}} \\left[(1-\\alpha)\\,\\mathrm{{rel}}(c) - \\alpha\\max_{{s \\in S}} \\mathrm{{sim}}(c,s)\\right].
$$

In this setting, relevance comes from the base ranking position and similarity is derived from L2-normalized genre vectors.

### 3.5 Iterative personalization

The personalization stage simulates a short interaction loop. After each round, the user state is updated toward the chosen item representation. This allows the ranking to adapt over five rounds and makes it possible to observe how fast the recommendation list changes as rho varies.

## 4. Quantitative results

### 4.1 Overall ranking performance

{metrics_table}

Figure 1 compares all available rankers over the three evaluation metrics.

![Ranker comparison](report_assets/ranker_metrics.png)

The strongest method on Recall@10 is **{best_recall_name}** with a score of **{best_rankers['Recall@10'][1]['Recall@10']:.4f}**. The same method also achieves the best NDCG@10 with **{best_rankers['NDCG@10'][1]['NDCG@10']:.4f}**. In contrast, the strongest method on Diversity@10 is **{best_diversity_name}** with **{best_rankers['Diversity@10'][1]['Diversity@10']:.4f}**.

Two observations stand out. First, the popularity baseline is unexpectedly difficult to beat on relevance, which suggests that the current latent models and pairwise rankers are under-tuned relative to the dataset split. Second, MMR is highly effective at increasing diversity: moving from the base MF-SGD ranking to MMR with alpha = 0.7 raises Diversity@10 from {results['R2_mf_sgd']['Diversity@10']:.4f} to {results['R5_mmr_on_sgd_alpha_0.7']['Diversity@10']:.4f}, while Recall@10 remains essentially stable.

### 4.2 MMR trade-off analysis

Figure 2 isolates the three MMR configurations built on the MF-SGD ranking.

![MMR tradeoff](report_assets/mmr_tradeoff.png)

The figure shows a clear trade-off. As alpha grows from 0.1 to 0.7, Diversity@10 increases monotonically from {results['R5_mmr_on_sgd_alpha_0.1']['Diversity@10']:.4f} to {results['R5_mmr_on_sgd_alpha_0.7']['Diversity@10']:.4f}. Meanwhile Recall@10 fluctuates only slightly around 0.031, and NDCG@10 remains in a narrow band around 0.076. In practical terms, alpha = 0.4 appears to be a good compromise because it already delivers a large diversity jump relative to pure MF-SGD and even gives the best NDCG@10 among the three MMR settings.

## 5. Personalization analysis

### 5.1 Session-level observations

{personalization_table}

Figure 3 aggregates the personalization behavior by configuration.

![Personalization summary](report_assets/personalization_summary.png)

The patterns are coherent across users:

- MF rho = 0.2 yields moderate drift ({mf_low['mean_drift']:.4f} on average) and preserves more of the original ranking ({mf_low['mean_overlap']:.2f}/10 overlap).
- MF rho = 0.5 yields the strongest adaptation ({mf_high['mean_drift']:.4f}) and the lowest overlap ({mf_high['mean_overlap']:.2f}/10), showing that a larger update rate makes the recommendation list react more aggressively.
- Genre rho = 0.4 yields stable recommendation lists with full overlap ({genre_mid['mean_overlap']:.2f}/10) despite non-zero state movement ({genre_mid['mean_drift']:.4f}). This suggests that the lower-dimensional genre representation changes the internal state without changing the top-10 set enough to alter the ranking.

### 5.2 Interpretation

The personalization loop is useful because it demonstrates that the system can react differently depending on how user feedback is embedded. Latent MF vectors are more expressive and therefore more sensitive to the update coefficient. Genre vectors are more interpretable, but they appear too coarse to generate visibly different top-10 lists over only five rounds. This is a meaningful design insight: if the goal is rapid conversational adaptation, latent representations combined with a calibrated rho are more promising than sparse genre vectors.

## 6. Discussion

The project reveals three central trade-offs.

First, model complexity does not automatically improve top-k relevance. The popularity baseline is still the best ranker on both Recall@10 and NDCG@10, which is a useful reminder that strong baselines matter and that matrix-factorization models require careful tuning.

Second, diversification via MMR works well even when the base ranker is weak. Because the genre representation captures broad content differences, MMR can generate substantially more diverse recommendation lists while preserving almost the same relevance level.

Third, personalization strength must be controlled. A small rho preserves stability, while a larger rho reacts faster but can drastically reshape the ranking. The best setting depends on whether the application prioritizes consistency or responsiveness.

## 7. Limitations and future work

The current workspace outputs also reveal some limitations:

1. Only the MMR variants built on MF-SGD are present in the final metrics file, so the report cannot compare MMR over popularity, ALS, or pairwise bases.
2. The latent models appear under-optimized relative to the popularity baseline; future work should tune the number of factors, regularization, and optimization schedule.
3. The personalization study is intentionally small, with only three users and five rounds per session. A larger user sample would make the conclusions more robust.
4. The report uses binary relevance from ratings >= 4. Alternative thresholds or graded relevance could offer a richer picture.

## 8. Conclusion

This assignment delivers a complete recommendation workflow with training, ranking, diversification, evaluation, and conversational personalization. The main empirical result is that MMR is the clearest improvement available in the current setup: it dramatically boosts diversity while keeping Recall@10 and NDCG@10 nearly unchanged. The personalization experiment complements this by showing that the update coefficient rho governs the balance between stability and adaptation. Overall, the project is reproducible, analytically coherent, and provides a solid basis for further tuning of the ranking models.

## 9. Reproducibility

The full pipeline can be reproduced with the following commands:

```bash
/Users/lunikk/Desktop/AIE-Assignment1/venv/bin/python cocreation_loop1.py
/Users/lunikk/Desktop/AIE-Assignment1/venv/bin/python report_metrics.py
/Users/lunikk/Desktop/AIE-Assignment1/venv/bin/python generate_results_tables.py
/Users/lunikk/Desktop/AIE-Assignment1/venv/bin/python generate_report_assets.py
```

The generated deliverables are:

- results/FINAL_REPORT.md
- results/report_assets/ranker_metrics.png
- results/report_assets/mmr_tradeoff.png
- results/report_assets/personalization_summary.png
- results/results_tables.xlsx
"""


def main() -> None:
    payload, results = load_metrics()
    observations = parse_personalization()
    plot_ranker_metrics(results)
    plot_mmr_tradeoff(results)
    plot_personalization_summary(observations)
    REPORT_PATH.write_text(build_report_text(payload, results, observations), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    print(f"Assets written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()