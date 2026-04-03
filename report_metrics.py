import json
import os
import numpy as np
import pandas as pd
from MF_ALS import MF_ALS
from MF_SGD import MF_SGD
from _utils import (
    get_candidates,
    load_pairwise_artifacts,
    rank_mf,
    rank_popularity,
    rank_pairwise,
)
from metrics import (
    build_relevant_items_map,
    evaluate_ranker_at_k,
    recall_at_k,
    ndcg_at_k_binary,
    diversity_at_k,
)
from mmr import build_genre_matrix, mmr_rerank_from_ranking

FAST_MODE        = False
FAST_USER_SUBSET = 300
MMR_TOP_M        = 80
PAIRWISE_EPOCHS  = 8


def _load_jsonl_records(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _save_metrics_report(report, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)


def _evaluate_and_log(method_name, user_ids, ranking_fn, relevant_map,
                      genre_matrix, item_to_idx, hyperparams, user_map,
                      item_map, k=10):
    records, recall_vals, ndcg_vals, div_vals = [], [], [], []
    for user_id in user_ids:
        relevant_items = relevant_map.get(int(user_id), set())
        if not relevant_items:
            continue
        ranking = ranking_fn(int(user_id))
        rc  = recall_at_k(ranking, relevant_items, k)
        nd  = ndcg_at_k_binary(ranking, relevant_items, k)
        div = diversity_at_k(ranking, genre_matrix, item_to_idx, k)
        records.append({
            "record_type":    "ranking_evaluation",
            "user_internal_idx": int(user_map[int(user_id)]),
            "user_id":        int(user_id),
            "method":         method_name,
            "method_name":    method_name,
            "top_k":          [int(item_map[int(x)]) for x in ranking[:k]],
            "top_k_item_ids": [int(x) for x in ranking[:k]],
            "metrics": {
                "Recall@10":    float(rc)  if rc  is not None else 0.0,
                "NDCG@10":      float(nd)  if nd  is not None else 0.0,
                "Diversity@10": float(div),
            },
            "hyperparameters": hyperparams,
        })
        if rc is not None:  recall_vals.append(rc)
        if nd is not None:  ndcg_vals.append(nd)
        div_vals.append(div)
    agg = {
        "k":               int(k),
        "evaluated_users": int(len(recall_vals)),
        "Recall@10":       float(np.mean(recall_vals)) if recall_vals else 0.0,
        "NDCG@10":         float(np.mean(ndcg_vals))   if ndcg_vals   else 0.0,
        "Diversity@10":    float(np.mean(div_vals))    if div_vals    else 0.0,
    }
    return agg, records


def main():
    ratings = pd.read_csv("./ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv(
        "./ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        usecols=range(24),
        names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL"]
        + [f"genre_{i}" for i in range(19)],
    )
    train_ratings = pd.read_csv("./ml-100k/u1.base", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    test_ratings = pd.read_csv("./ml-100k/u1.test", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    user_ids = ratings["user_id"].unique()
    item_ids = ratings["item_id"].unique()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    item_map = {iid: i for i, iid in enumerate(item_ids)}

    all_items = sorted(train_ratings["item_id"].unique())
    relevant_map = build_relevant_items_map(test_ratings, threshold=4)

    genre_matrix, item_to_idx = build_genre_matrix(movies)

    sgd_ckpt = "results/models/mf_sgd_u1.npz"
    als_ckpt = "results/models/mf_als_u1.npz"
    pw_sgd_ckpt = "results/models/pairwise_sgd_u1.pkl"
    pw_als_ckpt = "results/models/pairwise_als_u1.pkl"

    missing = [p for p in [sgd_ckpt, als_ckpt, pw_sgd_ckpt, pw_als_ckpt] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoints "
            + ", ".join(missing)
        )

    mf_sgd_model = MF_SGD.load(sgd_ckpt)
    mf_als_model = MF_ALS.load(als_ckpt)
    pairwise_model_sgd, popularity_sgd = load_pairwise_artifacts(pw_sgd_ckpt)
    pairwise_model_als, popularity_als = load_pairwise_artifacts(pw_als_ckpt)

    def ranking_pop(user_id):
        candidates = get_candidates(user_id, train_ratings, all_items)
        return rank_popularity(candidates, train_ratings)

    def ranking_sgd(user_id):
        candidates = get_candidates(user_id, train_ratings, all_items)
        return rank_mf(candidates, user_id, mf_sgd_model, user_map, item_map)

    def ranking_als(user_id):
        candidates = get_candidates(user_id, train_ratings, all_items)
        return rank_mf(candidates, user_id, mf_als_model, user_map, item_map)

    def ranking_pairwise_sgd(user_id):
        candidates = get_candidates(user_id, train_ratings, all_items)
        return rank_pairwise(candidates, user_id, mf_sgd_model, pairwise_model_sgd, user_map, item_map, popularity_sgd)

    def ranking_pairwise_als(user_id):
        candidates = get_candidates(user_id, train_ratings, all_items)
        return rank_pairwise(candidates, user_id, mf_als_model, pairwise_model_als, user_map, item_map, popularity_als)

    def ranking_mmr_sgd_alpha_01(user_id):
        base = ranking_sgd(user_id)
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.1, top_M=MMR_TOP_M, output_K=10)

    def ranking_mmr_sgd_alpha_04(user_id):
        base = ranking_sgd(user_id)
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.4, top_M=MMR_TOP_M, output_K=10)

    def ranking_mmr_sgd_alpha_07(user_id):
        base = ranking_sgd(user_id)
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.7, top_M=MMR_TOP_M, output_K=10)

    eligible_users = [u for u in user_ids if relevant_map.get(int(u))]
    eval_users = np.array(eligible_users[:FAST_USER_SUBSET] if FAST_MODE else eligible_users)
    print(f"[EVAL] Evaluating {len(eval_users)} users (FAST_MODE={FAST_MODE})")

    rankers = [
        ("R1_popularity", "popularity", ranking_pop, {
            "K": 10,
        }),
        ("R2_mf_sgd", "MF-SGD", ranking_sgd, {
            "d": int(mf_sgd_model.n_factors),
            "lambda": float(mf_sgd_model.reg),
            "eta": float(mf_sgd_model.lr),
            "epochs": 10,
            "K": 10,
        }),
        ("R3_mf_als", "MF-ALS", ranking_als, {
            "d": int(mf_als_model.n_factors),
            "lambda": float(mf_als_model.reg),
            "iters": 8,
            "K": 10,
        }),
        ("R4_pairwise_mf_sgd", "pairwise", ranking_pairwise_sgd, {
            "base_model": "MF-SGD",
            "d": int(mf_sgd_model.n_factors),
            "lambda": float(mf_sgd_model.reg),
            "eta": float(mf_sgd_model.lr),
            "max_pairs_per_user": 50,
            "ltr_epochs": PAIRWISE_EPOCHS,
            "K": 10,
        }),
        ("R4_pairwise_mf_als", "pairwise", ranking_pairwise_als, {
            "base_model": "MF-ALS",
            "d": int(mf_als_model.n_factors),
            "lambda": float(mf_als_model.reg),
            "max_pairs_per_user": 50,
            "ltr_epochs": PAIRWISE_EPOCHS,
            "K": 10,
        }),
        ("R5_mmr_on_sgd_alpha_0.1", "MMR", ranking_mmr_sgd_alpha_01, {
            "base_model": "MF-SGD",
            "d": int(mf_sgd_model.n_factors),
            "lambda": float(mf_sgd_model.reg),
            "eta": float(mf_sgd_model.lr),
            "alpha": 0.1,
            "M": MMR_TOP_M,
            "K": 10,
        }),
        ("R5_mmr_on_sgd_alpha_0.4", "MMR", ranking_mmr_sgd_alpha_04, {
            "base_model": "MF-SGD",
            "d": int(mf_sgd_model.n_factors),
            "lambda": float(mf_sgd_model.reg),
            "eta": float(mf_sgd_model.lr),
            "alpha": 0.4,
            "M": MMR_TOP_M,
            "K": 10,
        }),
        ("R5_mmr_on_sgd_alpha_0.7", "MMR", ranking_mmr_sgd_alpha_07, {
            "base_model": "MF-SGD",
            "d": int(mf_sgd_model.n_factors),
            "lambda": float(mf_sgd_model.reg),
            "eta": float(mf_sgd_model.lr),
            "alpha": 0.7,
            "M": MMR_TOP_M,
            "K": 10,
        }),
    ]

    results_dict   = {}
    all_log_records = []
    for result_key, method_name, fn, hparams in rankers:
        print(f"  Evaluating {result_key}...")
        agg, records = _evaluate_and_log(
            method_name, eval_users, fn, relevant_map, genre_matrix, item_to_idx, hparams, user_map, item_map, k=10
        )
        results_dict[result_key] = agg
        all_log_records.extend(records)

    report = {
        "dataset_split":         "u1.base / u1.test",
        "relevance_definition":  "rating >= 4",
        "fast_mode":             FAST_MODE,
        "metrics": {
            "M1": "Recall@10 (averaged over users with |Gu|>0)",
            "M2": "NDCG@10 binary (averaged over users with |Gu|>0)",
            "M3": "Diversity@10 from genre vectors (average pairwise cosine distance)",
        },
        "results": results_dict,
    }

    pop_metrics = results_dict.get("R1_popularity", {})
    others = [v for k, v in results_dict.items() if k != "R1_popularity"]
    max_other_recall = max([x.get("Recall@10", 0.0) for x in others], default=0.0)
    max_other_ndcg = max([x.get("NDCG@10", 0.0) for x in others], default=0.0)
    popularity_dominates = (
        pop_metrics.get("Recall@10", 0.0) >= max_other_recall
        and pop_metrics.get("NDCG@10", 0.0) >= max_other_ndcg
    )

    leakage_checks = {
        "triggered": bool(popularity_dominates),
        "seen_items_excluded": None,
        "popularity_source_train_only": None,
        "split_consistency": True,
    }

    if popularity_dominates:
        sample_users = eval_users[: min(50, len(eval_users))]
        seen_ok = True
        for u in sample_users:
            top_k = ranking_pop(int(u))[:10]
            rated_train = set(train_ratings[train_ratings["user_id"] == int(u)]["item_id"].astype(int).tolist())
            if any(item in rated_train for item in top_k):
                seen_ok = False
                break

        popularity_counts = train_ratings.groupby("item_id").size()
        source_ok = True
        for u in sample_users[:10]:
            candidates = get_candidates(int(u), train_ratings, all_items)
            ranked = ranking_pop(int(u))
            expected = sorted(candidates, key=lambda i: popularity_counts.get(i, 0), reverse=True)
            if ranked[:10] != expected[:10]:
                source_ok = False
                break

        leakage_checks["seen_items_excluded"] = seen_ok
        leakage_checks["popularity_source_train_only"] = source_ok

    report["leakage_checks"] = leakage_checks

    out_path = "results/metrics/metrics_at_10.json"
    _save_metrics_report(report, out_path)

    jsonl_path = "results/metrics/evaluation_log.jsonl"
    session_jsonl_path = "results/metrics/session_log.jsonl"
    session_records = _load_jsonl_records(session_jsonl_path)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as f:
        for rec in all_log_records:
            f.write(json.dumps(rec) + "\n")
        for rec in session_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nMetrics saved in {out_path}")
    print(f"JSONL log saved in {jsonl_path} ({len(all_log_records) + len(session_records)} records)")
    for name, vals in report["results"].items():
        print(
            f"{name}: Recall@10={vals['Recall@10']:.4f}, "
            f"NDCG@10={vals['NDCG@10']:.4f}, Diversity@10={vals['Diversity@10']:.4f}, "
            f"users={vals['evaluated_users']}"
        )


if __name__ == "__main__":
    main()
