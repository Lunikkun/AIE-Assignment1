import json
import os
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
from metrics import build_relevant_items_map, evaluate_ranker_at_k
from mmr import build_genre_matrix, mmr_rerank_from_ranking

def _save_metrics_report(report, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)


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
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.1, top_M=50, output_K=10)

    def ranking_mmr_sgd_alpha_04(user_id):
        base = ranking_sgd(user_id)
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.4, top_M=50, output_K=10)

    def ranking_mmr_sgd_alpha_07(user_id):
        base = ranking_sgd(user_id)
        return mmr_rerank_from_ranking(base, genre_matrix, item_to_idx, alpha=0.7, top_M=50, output_K=10)

    report = {
        "dataset_split": "u1.base / u1.test",
        "relevance_definition": "rating >= 4",
        "metrics": {
            "M1": "Recall@10 (media su utenti con |Gu|>0)",
            "M2": "NDCG@10 binaria (media su utenti con |Gu|>0)",
            "M3": "Diversity@10 da generi (media distanza coseno pairwise)",
        },
        "results": {
            "R1_popularity": evaluate_ranker_at_k(user_ids, ranking_pop, relevant_map, genre_matrix, item_to_idx, k=10),
            "R2_mf_sgd": evaluate_ranker_at_k(user_ids, ranking_sgd, relevant_map, genre_matrix, item_to_idx, k=10),
            "R3_mf_als": evaluate_ranker_at_k(user_ids, ranking_als, relevant_map, genre_matrix, item_to_idx, k=10),
            "R4_pairwise_mf_sgd": evaluate_ranker_at_k(user_ids, ranking_pairwise_sgd, relevant_map, genre_matrix, item_to_idx, k=10),
            "R4_pairwise_mf_als": evaluate_ranker_at_k(user_ids, ranking_pairwise_als, relevant_map, genre_matrix, item_to_idx, k=10),
            "R5_mmr_on_sgd_alpha_0.1": evaluate_ranker_at_k(user_ids, ranking_mmr_sgd_alpha_01, relevant_map, genre_matrix, item_to_idx, k=10),
            "R5_mmr_on_sgd_alpha_0.4": evaluate_ranker_at_k(user_ids, ranking_mmr_sgd_alpha_04, relevant_map, genre_matrix, item_to_idx, k=10),
            "R5_mmr_on_sgd_alpha_0.7": evaluate_ranker_at_k(user_ids, ranking_mmr_sgd_alpha_07, relevant_map, genre_matrix, item_to_idx, k=10),
        },
    }

    out_path = "results/metrics/metrics_at_10.json"
    _save_metrics_report(report, out_path)

    print("\Metrics saved in", out_path)
    for name, vals in report["results"].items():
        print(
            f"{name}: Recall@10={vals['Recall@10']:.4f}, "
            f"NDCG@10={vals['NDCG@10']:.4f}, Diversity@10={vals['Diversity@10']:.4f}, "
            f"users={vals['evaluated_users']}"
        )


if __name__ == "__main__":
    main()
