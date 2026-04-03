import math
from itertools import combinations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_relevant_items_map(test_ratings, threshold=4):
    """Map user_id -> set of relevant items (rating >= threshold [4])."""
    relevant = {}
    grouped = test_ratings.groupby("user_id")
    for user_id, rows in grouped:
        rel_items = set(rows[rows["rating"] >= threshold]["item_id"].astype(int).tolist())
        relevant[int(user_id)] = rel_items
    return relevant


def recall_at_k(recommended, relevant_items, k=10):
    if not relevant_items:
        return None
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / float(len(relevant_items))


def ndcg_at_k_binary(recommended, relevant_items, k=10):
    if not relevant_items:
        return None

    top_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k, start=1):
        rel = 1.0 if item in relevant_items else 0.0
        if rel > 0:
            dcg += rel / math.log2(idx + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def diversity_at_k(recommended, genre_matrix, item_to_idx, k=10):
    top_k = [item for item in recommended[:k] if item in item_to_idx]
    if len(top_k) < 2:
        return 0.0

    distances = []
    for i_item, j_item in combinations(top_k, 2):
        i_vec = genre_matrix[item_to_idx[i_item]].reshape(1, -1)
        j_vec = genre_matrix[item_to_idx[j_item]].reshape(1, -1)
        sim = float(cosine_similarity(i_vec, j_vec)[0, 0])
        distances.append(1.0 - sim)

    return float(np.mean(distances)) if distances else 0.0


def evaluate_ranker_at_k(user_ids, ranking_fn, relevant_map, genre_matrix, item_to_idx, k=10):
    recall_values = []
    ndcg_values = []
    diversity_values = []

    for user_id in user_ids:
        relevant_items = relevant_map.get(int(user_id), set())
        if len(relevant_items) == 0:
            continue

        ranking = ranking_fn(int(user_id))
        recall = recall_at_k(ranking, relevant_items, k=k)
        ndcg = ndcg_at_k_binary(ranking, relevant_items, k=k)
        div = diversity_at_k(ranking, genre_matrix, item_to_idx, k=k)

        if recall is not None:
            recall_values.append(recall)
        if ndcg is not None:
            ndcg_values.append(ndcg)
        diversity_values.append(div)

    return {
        "k": int(k),
        "evaluated_users": int(len(recall_values)),
        "Recall@10": float(np.mean(recall_values)) if recall_values else 0.0,
        "NDCG@10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "Diversity@10": float(np.mean(diversity_values)) if diversity_values else 0.0,
    }
